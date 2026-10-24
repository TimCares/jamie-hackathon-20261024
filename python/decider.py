"""Local "System One" decisions with Perplexity's pplx-decider-flash-v1.2.

Instead of generating text, the model reads a state and answers typed questions in one forward pass:
  - choice: pick one of N named options     -> {"choice": "support", "probabilities": {...}, "confidence": ...}
  - noul:   yes/no                          -> {"noul": 0.93}   (P(true))
  - score:  ordered scale, lowest level first -> {"score": 2.4, "probabilities": {...}, ...}

Usage:
    from decider import choose, ask, decide

    choose("401 since we rotated our API key", {"billing": "Charges", "support": "Integration errors"})
    ask("401 since we rotated our API key", "Is the customer blocked right now?")
    decide({"state": ..., "questions": {...}})   # full request, several questions at once (cheapest)

Everything (weights, HF cache) lives under this folder, so deleting the project removes it.
Runs on CUDA (bf16, optional packed fast path), Apple Silicon (MPS) or CPU.
"""

from __future__ import annotations

import os
import sys
from functools import cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".hf"))  # keep any HF cache project-local

REPO = "perplexity-ai/pplx-decider-flash-v1.2"
MODEL_DIR = ROOT / ".models" / "pplx-decider-flash-v1.2"


def best_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def download() -> Path:
    """Fetch the checkpoint (~9 GB) into MODEL_DIR if it isn't there yet."""
    if not (MODEL_DIR / "readout.safetensors").exists():
        from huggingface_hub import snapshot_download

        snapshot_download(REPO, local_dir=MODEL_DIR)
    return MODEL_DIR


def load(device: str | None = None, *, fast: bool | None = None, bf16: bool | None = None):
    """Load a Decider.

    device: "cuda" | "mps" | "cpu"; defaults to the best available.
    fast:   packed fast path (CUDA + flash-linear-attention only); defaults to on when available.
    bf16:   run in bfloat16. The upstream code uses bf16 on CUDA and fp32 elsewhere; on MPS bf16 halves memory
            (~9 GB instead of ~18 GB) and is usually faster, at the cost of slight probability noise. Text only.
    """
    import torch

    checkpoint = download()
    sys.path.insert(0, str(checkpoint / "source" / "src"))
    from pplx_decider import Decider

    device = device or best_device()
    if fast is None:
        fast = device == "cuda" and _has_fla()
    decider = Decider.from_pretrained(checkpoint, device=device, fast=fast)
    if bf16 is None:
        bf16 = device == "mps"
    if bf16 and device != "cuda":  # CUDA already loads in bf16
        decider.backbone.to(torch.bfloat16)
        decider.readout.to(torch.bfloat16)
    decider.warmup()
    return decider


def _has_fla() -> bool:
    try:
        import fla  # noqa: F401
    except ImportError:
        return False
    return True


@cache
def get_decider():
    """Process-wide Decider, loaded on first use."""
    return load()


# --- Convenience helpers --------------------------------------------------------
def decide(request: dict[str, Any] | list[dict[str, Any]]) -> Any:
    """Full request (or a list of them): {"state": ..., "questions": {name: question}}. See the module docstring."""
    return get_decider().decide(request)


def choose(state: Any, options: dict[str, str | None], instructions: str | None = None) -> str:
    """Pick one option name. `options` maps name -> description (or None)."""
    question = {"type": "choice", "criteria": options, "instructions": instructions}
    return decide({"state": state, "questions": {"q": question}})["answers"]["q"]["choice"]


def ask(state: Any, question: str) -> float:
    """Yes/no question; returns P(yes)."""
    return decide({"state": state, "questions": {"q": {"type": "noul", "instructions": question}}})["answers"]["q"]["noul"]


def rate(state: Any, levels: list[str], instructions: str | None = None) -> float:
    """Expected level on an ordered scale (0 = levels[0])."""
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
