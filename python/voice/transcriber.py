"""Local speech-to-text: Silero VAD finds speech, Parakeet (MLX) transcribes it.

While someone talks, the whole utterance so far is re-transcribed every `interim_every` seconds (is_final=False).
At a pause, Smart Turn judges from the audio whether the speaker is done: if so the utterance ends right away
(is_final=True), otherwise it waits up to `turn_wait` seconds for them to go on.
"""

import asyncio
import os
import re
import time
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / ".hf"))  # model cache in the project
os.environ.setdefault("HF_TOKEN_PATH", str(Path.home() / ".cache/huggingface/token"))  # login from `hf auth login`

import mlx.core as mx
import numpy as np
import torch
from parakeet_mlx import from_pretrained
from parakeet_mlx.audio import get_logmel
from silero_vad import load_silero_vad

warnings.filterwarnings("ignore", message=".*torch.jit.load.*")

MODEL = "mlx-community/parakeet-tdt_ctc-110m"  # English, ~460 MB
SAMPLE_RATE = 16000
VAD_FRAME = 512  # samples per Silero frame (32 ms)
FRAME_S = VAD_FRAME / SAMPLE_RATE


@dataclass
class Transcript:
    text: str
    is_final: bool


class TurnDetector:
    """Smart Turn v3 (Pipecat): P(the speaker finished their turn) from the last 8 s of audio; ~11 ms on CPU."""

    def __init__(self, repo: str = "pipecat-ai/smart-turn-v3", file: str = "smart-turn-v3.2-cpu.onnx"):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import WhisperFeatureExtractor

        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(hf_hub_download(repo, file), sess_options=options)
        self._features = WhisperFeatureExtractor(chunk_length=8)

    def complete(self, audio: np.ndarray) -> float:
        n = 8 * SAMPLE_RATE
        audio = np.pad(audio[-n:], (max(0, n - len(audio)), 0))  # the last 8 s, padded at the start
        features = self._features(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="np", padding="max_length", max_length=n,
            truncation=True, do_normalize=True,
        ).input_features
        return float(self._session.run(None, {"input_features": features.astype(np.float32)})[0][0].item())


class ParakeetTranscriber:
    """16 kHz mono int16 audio goes into `audio_in`, Transcripts come out of `transcripts`."""

    def __init__(
        self,
        model: str = MODEL,
        *,
        endpoint_silence: float = 0.5,  # seconds of silence that end an utterance (without Smart Turn)
        smart_turn: bool = True,  # ask Smart Turn at each pause instead of waiting a fixed time
        turn_check: float = 0.2,  # pause before asking Smart Turn
        turn_wait: float = 1.0,  # longest pause when Smart Turn says the speaker isn't done
        interim_every: float = 0.2,  # seconds of new speech between interim results
        max_utterance: float = 30.0,  # force a final result after this long
        speech_threshold: float = 0.5,  # Silero speech probability
        pre_roll: float = 0.3,  # audio kept from before speech was detected
        device: str = "gpu",  # "cpu" leaves the GPU to the decider
    ):
        self.model_id = model
        self.device = device
        self.endpoint_silence = endpoint_silence
        self.smart_turn = smart_turn
        self.turn_check = turn_check
        self.turn_wait = turn_wait
        self.interim_every = interim_every
        self.max_utterance = max_utterance
        self.speech_threshold = speech_threshold
        self.audio_in: asyncio.Queue[bytes] = asyncio.Queue()
        self.transcripts: asyncio.Queue[Transcript] = asyncio.Queue()

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="parakeet")
        self._model = None
        self._vad = None
        self._turn_model: TurnDetector | None = None
        self._turn: float | None = None  # Smart Turn's verdict for the current pause
        self._text_hint: tuple[str, float] | None = None  # (words, P(finished)) from a text model, see set_text_hint
        self._pending = np.zeros(0, np.float32)  # samples not yet run through the VAD
        self._pre_roll: deque[np.ndarray] = deque(maxlen=round(pre_roll / FRAME_S))
        self._utterance: list[np.ndarray] | None = None  # frames of the current utterance
        self._silence = 0.0
        self._interim_at = 0.0  # utterance length at the last interim
        self._interim_cost = 0.0  # how long the last transcription took
        self._last_text = ""

    def set_text_hint(self, text: str, finished: float) -> None:
        """P(finished) for the interim `text`, judged from the words (the decider's). When Smart Turn thinks the
        speaker isn't done, a confident text verdict still ends the utterance ("Actually, never mind.")."""
        self._text_hint = (_words(text), finished)

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._load)
        while True:
            chunks = [await self.audio_in.get()]
            while not self.audio_in.empty():  # catch up in one step if we fell behind
                chunks.append(self.audio_in.get_nowait())
            for transcript in await loop.run_in_executor(self._executor, self._step, b"".join(chunks)):
                await self.transcripts.put(transcript)

    def _load(self) -> None:
        t = time.perf_counter()
        mx.set_default_device(mx.cpu if self.device == "cpu" else mx.gpu)
        self._model = from_pretrained(self.model_id)
        self._vad = load_silero_vad()
        if self.smart_turn:
            self._turn_model = TurnDetector()
        self._transcribe(np.zeros(SAMPLE_RATE, np.float32))  # warm up
        print(f"[parakeet] {self.model_id} ({self.device}) ready in {time.perf_counter() - t:.1f}s")

    def _transcribe(self, samples: np.ndarray) -> str:
        t = time.perf_counter()
        mel = get_logmel(mx.array(samples), self._model.preprocessor_config)
        text = self._model.generate(mel)[0].text.strip()
        self._interim_cost = time.perf_counter() - t
        return text

    def _step(self, audio: bytes) -> list[Transcript]:
        samples = np.frombuffer(audio, np.int16).astype(np.float32) / 32768
        self._pending = np.concatenate([self._pending, samples])
        out = []
        while len(self._pending) >= VAD_FRAME:
            frame, self._pending = self._pending[:VAD_FRAME], self._pending[VAD_FRAME:]
            prob = self._vad(torch.from_numpy(frame), SAMPLE_RATE).item()
            if self._utterance is None:
                self._pre_roll.append(frame)
                if prob >= self.speech_threshold:
                    self._utterance = list(self._pre_roll)
                    self._silence = self._interim_at = 0.0
                    self._turn = None
                    self._last_text = ""
                    self._pre_roll.clear()
                continue
            self._utterance.append(frame)
            # A lower threshold to stay in speech, so short dips don't end the utterance.
            if prob >= self.speech_threshold - 0.15:
                self._silence, self._turn = 0.0, None
            else:
                self._silence += FRAME_S
            if self._ended() or len(self._utterance) * FRAME_S >= self.max_utterance:
                out += self._finish()

        if self._utterance is not None:
            length = len(self._utterance) * FRAME_S
            # Never spend more than about half the time transcribing, so we can't fall behind.
            if length - self._interim_at >= max(self.interim_every, 2 * self._interim_cost):
                self._interim_at = length
                text = self._transcribe(np.concatenate(self._utterance))
                if text and text != self._last_text:
                    self._last_text = text
                    out.append(Transcript(text, is_final=False))
        return out

    def _ended(self) -> bool:
        """Whether the current pause ends the utterance."""
        if self._turn_model is None:
            return self._silence >= self.endpoint_silence
        if self._silence < self.turn_check:
            return False
        if self._turn is None:  # once per pause
            self._turn = self._turn_model.complete(np.concatenate(self._utterance))
        if self._turn > 0.5 or self._silence >= self.turn_wait:
            return True
        hint = self._text_hint
        return hint is not None and hint[1] >= 0.8 and hint[0] == _words(self._last_text)

    def _finish(self) -> list[Transcript]:
        frames, self._utterance = self._utterance, None
        speech = frames[: len(frames) - round(self._silence / FRAME_S)] or frames  # without the trailing silence
        text = self._transcribe(np.concatenate(speech))
        return [Transcript(text, is_final=True)] if text else []


async def stream_microphone(audio_in: asyncio.Queue[bytes]) -> None:
    """Push mic audio (100 ms chunks) into `audio_in` until cancelled."""
    import sounddevice as sd

    loop = asyncio.get_running_loop()

    def callback(indata, frames, time, status):  # runs on the audio thread
        loop.call_soon_threadsafe(audio_in.put_nowait, bytes(indata))

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=SAMPLE_RATE // 10, callback=callback
    ):
        await asyncio.Event().wait()


def _words(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()
