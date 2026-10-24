# Voice coding

Talk while you code; small spoken requests become edits to the file in front of you.

Typing a prompt and waiting for a full answer breaks your flow. Here you just keep talking while looking at the
code ("rename this", "also handle None", "no, revert that"): each request becomes an edit a second or two later,
while you're already on to the next one. No pausing, no waiting, and corrections are as cheap as saying them.

## How it works

```
mic -> speech-to-text -> decider -> coder -> file
```

- **Speech-to-text** (Parakeet, local): partial text every ~0.2 s while you talk, the full sentence after a pause.
- **Decider** (pplx-decider, local): for each bit of speech: ignore, wait, new task, update a task, or undo. It
  already decides on partial text, so work starts before you finish the sentence.
- **Coder** (Claude Opus 5.5, fast mode): turns the request and the current file into small search/replace edits.
- **Writer**: applies edits one at a time. Each spoken request is an undoable step; "revert it" restores the file
  exactly, without an API call.

## Requirements

Built and tested on an Apple Silicon Mac. MLX and llama.cpp also run on Linux and Windows, but that's untested,
and the decider needs a GPU to stay fast. Plus [uv](https://docs.astral.sh/uv/) and `ANTHROPIC_API_KEY` in `.env`
(see `.env.example`). Models download on first run.

## Run

```bash
uv run --extra gguf python -m voice sample/user_service.py   # talk to the file
uv run python -m voice.live_view sample/user_service.py      # watch it change at http://localhost:8765
```

## Files

- `voice/transcriber.py` speech-to-text
- `voice/tasks.py` the decider: ignore, wait, new task, update, undo
- `voice/coder.py` Claude turns a request into edits (prompt: `voice/coder_system.md`)
- `voice/writer.py` applies edits to the file one at a time, undo per request
- `voice/live_view.py` browser view of the file
- `decider.py`, `decider_gguf.py` the decider model (PyTorch / llama.cpp), `bench_decider*.py` benchmarks
- `agent.py` minimal Claude agent loop
