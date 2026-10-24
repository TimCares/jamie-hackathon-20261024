"""Mic -> transcriber -> decider -> coder -> file, everything printed.

uv run --extra gguf python -m voice path/to/file
"""

import asyncio
import sys

from .coder import Coder
from .tasks import TaskDecider
from .transcriber import ParakeetTranscriber, stream_microphone
from .writer import FileWriter

GREY, YELLOW, GREEN, CYAN, RED, RESET = "\033[90m", "\033[33m", "\033[1;32m", "\033[1;36m", "\033[1;31m", "\033[0m"
COLORS = {"wait": YELLOW, "new": GREEN, "update": CYAN, "cancel": RED}


def show(line: str, *, keep: bool = True) -> None:
    """Print over the current interim line; keep=False leaves the cursor on it."""
    print(f"\r\033[K{line}", end="\n" if keep else "", flush=True)


def show_diff(diff: str) -> None:
    for line in diff.splitlines():
        color = GREEN if line.startswith("+") else RED if line.startswith("-") else CYAN if line.startswith("@@") else ""
        show(f"      {color}{line}{RESET}")


async def main(path: str):
    stt = ParakeetTranscriber(device="cpu")  # the GPU is for the decider
    decider = TaskDecider()
    writer = FileWriter(Coder(path))

    async def forward_speech():
        while True:
            t = await stt.transcripts.get()
            await decider.speech_in.put(t)
            show(f"> {t.text}" if t.is_final else f"~ {t.text}", keep=t.is_final)

    async def print_decisions():
        while True:
            d = await decider.decisions.get()
            probs = " ".join(f"{k} {v:.2f}" for k, v in d.probabilities.items()) + f" finished {d.finished:.2f}"
            target = f" #{d.target}" if d.target else ""
            color = COLORS.get(d.choice, GREY)
            show(f"  {color}[{d.choice}{target}]{RESET} {GREY}{probs} ({d.seconds * 1000:.0f} ms) | {d.text}{RESET}")

    async def print_events():
        while True:
            e = await decider.events.get()
            early = "" if e.is_final else " (early)"
            what = f"undo: {e.text}" if e.kind == "cancel" else e.said
            show(f"  {COLORS[e.kind]}=> {e.kind.upper()} #{e.task_id}{early} ({e.confidence:.2f}): {what}{RESET}")
            writer.handle(e)

    async def print_file_updates():
        while True:
            u = await writer.updates.get()
            label = f"<> {u.kind.upper()}" + (f" #{u.task_id}" if u.task_id else "") + ("" if u.is_final else " (early)")
            color = {"new": GREEN, "update": CYAN, "undo": RED, "failed": RED}.get(u.kind, GREY)
            slow = f", {u.speed} speed" if u.speed and u.speed != "fast" else ""
            timing = f" ({u.seconds:.2f}s{slow})" if u.seconds else ""
            show(f"  {color}{label}{timing}: {u.summary}{RESET}")
            show_diff(u.diff)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(stt.run())
        tg.create_task(decider.run())
        tg.create_task(stream_microphone(stt.audio_in))
        tg.create_task(forward_speech())
        tg.create_task(print_decisions())
        tg.create_task(print_events())
        tg.create_task(writer.run())
        tg.create_task(print_file_updates())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: uv run --extra gguf python -m voice path/to/file")
    print("Loading models, then listening... (Ctrl+C to stop)")
    try:
        asyncio.run(main(sys.argv[1]))
    except KeyboardInterrupt:
        pass
