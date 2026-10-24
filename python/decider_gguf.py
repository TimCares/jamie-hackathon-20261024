"""pplx-decider-flash-v1.2 on llama.cpp (Q8_0 GGUF, Metal / CUDA / CPU). Text-only.

Same API as decider.py (decide / choose / ask / rate), different engine:
  - prompt + tokenization: Perplexity's own protocol-v3 encoder (pplx_decider.protocol), so prompts are identical
  - forward pass: llama.cpp, whole sequence in one microbatch, non-causal full attention
  - readout: logits at each <|fim_suffix|> marker, restricted to the question's answer codes, / temperature, softmax

Stock llama-server does NOT implement this protocol; that's why this file drives llama.cpp's low-level API itself.
Needs:  uv sync --extra gguf
Downloads: the 5.2 GB GGUF plus the small tokenizer/config/source files of the base repo (not its 9 GB weights).
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import sys
from functools import cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".hf"))

GGUF_REPO = "perplexity-ai/pplx-decider-flash-gguf-v1.2"
GGUF_FILE = "decider-flash-v1.2-q8_0.gguf"
BASE_REPO = "perplexity-ai/pplx-decider-flash-v1.2"
GGUF_DIR = ROOT / ".models" / "pplx-decider-flash-gguf-v1.2"
BASE_DIR = ROOT / ".models" / "pplx-decider-flash-v1.2"  # shared with decider.py; only small files needed here


_log_callback = None  # kept alive for llama.cpp


def download() -> tuple[Path, Path]:
    from huggingface_hub import hf_hub_download, snapshot_download

    if not (GGUF_DIR / GGUF_FILE).exists():
        hf_hub_download(GGUF_REPO, GGUF_FILE, local_dir=GGUF_DIR)
    if not (BASE_DIR / "decision_config.json").exists():
        snapshot_download(BASE_REPO, local_dir=BASE_DIR, allow_patterns=["*.json", "*.jinja", "source/**"])
    return GGUF_DIR / GGUF_FILE, BASE_DIR


class GGUFDecider:
    def __init__(
        self, max_length: int = 8192, gpu_layers: int = -1, threads: int | None = None, verbose: bool = False
    ) -> None:
        """max_length: longest sequence per forward pass; Metal reserves ~1 GB of compute buffer per 1k tokens."""
        import llama_cpp as ll

        global _log_callback
        if not verbose and _log_callback is None:
            _log_callback = ll.llama_log_callback(lambda level, text, user_data: None)
            ll.llama_log_set(_log_callback, ctypes.c_void_p())

        gguf, base = download()
        sys.path.insert(0, str(base / "source" / "src"))
        from pplx_decider.protocol import Encoder, answer
        from transformers import AutoProcessor

        self.ll, self.answer, self.max_length = ll, answer, max_length
        config = json.loads((base / "decision_config.json").read_text())
        self.temperature = float(config["temperature"])
        self.token_ids: list[int] = list(config["token_ids"])
        processor = AutoProcessor.from_pretrained(str(base))
        self.encoder = Encoder(processor, config["codes"])

        ll.llama_backend_init()
        mparams = ll.llama_model_default_params()
        mparams.n_gpu_layers = 999 if gpu_layers < 0 else gpu_layers
        self.model = ll.llama_model_load_from_file(str(gguf).encode(), mparams)
        if not self.model:
            raise RuntimeError(f"llama.cpp could not load {gguf}")

        cparams = ll.llama_context_default_params()
        # the whole request must be evaluated in a single microbatch (non-causal attention spans all of it)
        cparams.n_ctx = cparams.n_batch = cparams.n_ubatch = max_length
        cparams.n_seq_max = 1
        if threads:
            cparams.n_threads = cparams.n_threads_batch = threads
        self.ctx = ll.llama_init_from_model(self.model, cparams)
        if not self.ctx:
            raise RuntimeError("llama.cpp could not create a context")
        ll.llama_set_causal_attn(self.ctx, False)  # GGUF metadata says the same; linear-attn layers stay causal
        self.batch = ll.llama_batch_init(max_length, 0, 1)

    def __del__(self) -> None:
        try:
            self.ll.llama_batch_free(self.batch)
            self.ll.llama_free(self.ctx)
            self.ll.llama_model_free(self.model)
        except Exception:
            pass

    def _logits(self, chunk) -> list[list[float]]:
        """Readout logits (valid codes only) at each marker of one encoded chunk."""
        ll, batch = self.ll, self.batch
        ids = chunk.input_ids.tolist()
        markers = set(chunk.markers.tolist())
        ll.llama_memory_clear(ll.llama_get_memory(self.ctx), True)
        batch.n_tokens = len(ids)
        for i, token in enumerate(ids):
            batch.token[i] = token
            batch.pos[i] = i
            batch.n_seq_id[i] = 1
            batch.seq_id[i][0] = 0
            batch.logits[i] = i in markers
        if (status := ll.llama_decode(self.ctx, batch)) != 0:
            raise RuntimeError(f"llama_decode failed with status {status}")
        out = []
        for position, count in zip(chunk.markers.tolist(), chunk.counts, strict=True):
            row = ll.llama_get_logits_ith(self.ctx, position)
            out.append([row[token] for token in self.token_ids[:count]])
        return out

    def logits(self, request: dict[str, Any]) -> tuple[dict[str, list[float]], int]:
        """Raw readout logits per question (before temperature) and the number of input tokens."""
        result, tokens = {}, 0
        for chunk in self.encoder.plan(request, self.max_length):
            result.update(zip(chunk.keys, self._logits(chunk), strict=True))
            tokens += chunk.length
        return result, tokens

    def decide(self, requests, *, temperature: float | None = None):
        """One request -> one response; a list -> a list (run one after another)."""
        if isinstance(requests, list):
            return [self.decide(r, temperature=temperature) for r in requests]
        if requests.get("images"):
            raise ValueError("decider_gguf is text-only; use decider.py for images")
        scale = temperature or self.temperature
        logits, tokens = self.logits(requests)
        answers = {}
        for key, question in requests["questions"].items():
            values = [v / scale for v in logits[key]]
            top = max(values)
            exp = [math.exp(v - top) for v in values]
            answers[key] = self.answer(question, [e / sum(exp) for e in exp])
        return {"answers": answers, "usage": {"input_tokens": tokens, "output_tokens": 0}}


@cache
def get_decider() -> GGUFDecider:
    return GGUFDecider()


# --- Convenience helpers (same signatures as decider.py) ---------------------------
def decide(request):
    return get_decider().decide(request)


def choose(state: Any, options: dict[str, str | None], instructions: str | None = None) -> str:
    question = {"type": "choice", "criteria": options, "instructions": instructions}
    return decide({"state": state, "questions": {"q": question}})["answers"]["q"]["choice"]


def ask(state: Any, question: str) -> float:
    return decide({"state": state, "questions": {"q": {"type": "noul", "instructions": question}}})["answers"]["q"]["noul"]


def rate(state: Any, levels: list[str], instructions: str | None = None) -> float:
    question = {"type": "score", "criteria": levels, "instructions": instructions}
    return decide({"state": state, "questions": {"q": question}})["answers"]["q"]["score"]


if __name__ == "__main__":
    response = decide({
        "state": "My integration keeps failing with a 401 since we rotated our API key. Please help.",
        "questions": {
            "team": {
                "type": "choice",
                "instructions": "Which team should handle this request?",
                "criteria": {
                    "billing": "Charges and refunds",
                    "support": "Technical integration errors",
                    "sales": "Questions about buying a product",
                },
            },
            "urgent": {"type": "noul", "instructions": "Is the customer blocked right now?"},
            "frustration": {"type": "score", "criteria": ["calm", "mildly annoyed", "frustrated", "angry"]},
        },
    })
    for name, answer in response["answers"].items():
        print(name, answer)
    print("usage", response["usage"])
