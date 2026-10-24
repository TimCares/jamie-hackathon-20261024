# Voice coding

Talk while you code; small spoken requests become edits to one file.

```
mic -> speech-to-text (Parakeet, local) -> decider (pplx-decider, local) -> coder (Claude) -> file
```

Needs macOS on Apple Silicon, [uv](https://docs.astral.sh/uv/), and `ANTHROPIC_API_KEY` in `.env` (see `.env.example`).
Models download on first run.

## Run

```bash
uv run --extra gguf python -m voice sample/user_service.py   # talk to the file
uv run python -m voice.live_view sample/user_service.py      # watch it change at http://localhost:8765
uv run python -m voice.coder sample/user_service.py "rename get user to fetch user"   # one request, no mic
```

Say things like "add a docstring to parse config", "also mention it raises a ValueError", "revert it".

## Files

- `voice/transcriber.py` speech-to-text
- `voice/tasks.py` decides per sentence: ignore, wait, new task, update, undo
- `voice/coder.py` Claude turns a request into edits (prompt: `voice/coder_system.md`)
- `voice/writer.py` applies edits to the file one at a time, undo per request
- `voice/live_view.py` browser view of the file
- `decider.py`, `decider_gguf.py` the decider model (PyTorch / llama.cpp), `bench_decider*.py` benchmarks
- `agent.py` minimal Claude agent loop
