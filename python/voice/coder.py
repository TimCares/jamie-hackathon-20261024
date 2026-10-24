"""Coder: a spoken (imprecise) task + the file -> Claude -> search/replace edits -> git diff.

Claude Opus 5.5 in fast mode at low effort, with the file prompt-cached. Edits come back through a strict tool
because the cache warm-up (max_tokens=0) can't be combined with structured output.
"""

import difflib
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = Path(__file__).with_name("coder_system.md").read_text()
MODEL = "claude-opus-5-5"
BETAS = [
    "fast-mode-2026-02-01",  # speed="fast"
    "server-side-fallback-2026-07-01",  # fallbacks="default": a safety decline is retried on another model
]
EARLIER_CHARS = 800  # only the end of long earlier requests goes into the prompt
EDIT_TOOL = {
    "name": "submit_edits",
    "description": "Submit the edits for the spoken task; an empty list when the file should not change.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "edits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"old": {"type": "string"}, "new": {"type": "string"}},
                    "required": ["old", "new"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["edits"],
        "additionalProperties": False,
    },
}


class CoderError(Exception):
    pass


@dataclass
class CoderResult:
    summary: str  # from the diff, e.g. "+3 -1 lines"
    edits: list[dict]
    diff: str  # git-style diff; "" when nothing changed
    after: str
    seconds: float
    usage: dict = field(default_factory=dict)


class Coder:
    def __init__(self, path: str | Path, *, model: str = MODEL, effort: str = "low", fast: bool = True):
        self.path = Path(path)
        self.diff_path = diff_path(self.path)
        self.model = model
        self.effort = effort
        self.fast = fast
        self.client = anthropic.AsyncAnthropic()
        # No retries in fast mode: on a rate limit or error, switch to standard speed instead of backing off.
        self._fast_client = self.client.with_options(max_retries=0)

    async def edit(
        self, content: str, task: str, previous: str | None = None, earlier: str | None = None
    ) -> CoderResult:
        """Edits that make `content` do `task`. `previous`: an earlier wording of the same request that is already
        in `content` (only the difference is returned). `earlier`: other requests already done, as context."""
        t = time.perf_counter()
        prompt = f"<spoken_task>\n{task}\n</spoken_task>"
        if previous is not None:
            prompt = f"<previous_task>\n{previous}\n</previous_task>\n{prompt}"
        if earlier:
            prompt = f"<earlier_requests>\n{earlier[-EARLIER_CHARS:]}\n</earlier_requests>\n{prompt}"
        messages = [{"role": "user", "content": [self._file_block(content), {"type": "text", "text": prompt}]}]
        response = await self._request(messages)
        call = self._tool_call(response)
        try:
            after = apply_edits(content, call.input["edits"])
        except CoderError as error:  # one retry with the reason, usually an `old` that wasn't copied exactly
            messages += [
                {"role": "assistant", "content": response.content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": call.id,
                            "is_error": True,
                            "content": f"Those edits could not be applied: {error}. Call submit_edits again.",
                        }
                    ],
                },
            ]
            response = await self._request(messages)
            call = self._tool_call(response)
            after = apply_edits(content, call.input["edits"])

        usage = response.usage
        return CoderResult(
            summary=describe(content, after),
            edits=call.input["edits"],
            diff=git_diff(self.diff_path, content, after),
            after=after,
            seconds=time.perf_counter() - t,
            usage={
                "input": usage.input_tokens,
                "cache_read": usage.cache_read_input_tokens or 0,
                "cache_write": usage.cache_creation_input_tokens or 0,
                "output": usage.output_tokens,
                "speed": getattr(usage, "speed", None),
            },
        )

    async def warm(self, content: str) -> None:
        """Write the prompt cache for `content` now (max_tokens=0: no output, only the cache write is billed)."""
        placeholder = {"type": "text", "text": "warm up"}  # after the cache breakpoint
        messages = [{"role": "user", "content": [self._file_block(content), placeholder]}]
        await self._request(messages, max_tokens=0, fallback=False)

    def _file_block(self, content: str) -> dict:
        return {
            "type": "text",
            "text": f'<file path="{self.diff_path}">\n{content}\n</file>',
            "cache_control": {"type": "ephemeral"},
        }

    async def _request(self, messages: list[dict], max_tokens: int = 16000, fallback: bool = True):
        params = dict(
            model=self.model,
            max_tokens=max_tokens,
            betas=BETAS,
            fallbacks="default",
            system=SYSTEM_PROMPT,
            tools=[EDIT_TOOL],
            tool_choice={"type": "auto"},  # forced tool choice is a 400 on this model; the prompt asks for it
            output_config={"effort": self.effort},
            messages=messages,
        )
        if not self.fast:
            return await self.client.beta.messages.create(**params)
        try:
            return await self._fast_client.beta.messages.create(**params, speed="fast")
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as error:
            status = getattr(error, "status_code", None)
            if not fallback or (status is not None and status != 429 and status < 500):
                raise
            return await self.client.beta.messages.create(**params)  # usage.speed then says "standard"

    @staticmethod
    def _tool_call(response):
        if response.stop_reason == "refusal":
            raise CoderError(f"model declined: {response.stop_details}")
        if response.stop_reason == "max_tokens":
            raise CoderError("model output was cut off (max_tokens)")
        for block in response.content:
            if block.type == "tool_use" and block.name == EDIT_TOOL["name"]:
                return block
        raise CoderError("model answered without calling submit_edits")


def apply_edits(content: str, edits: list[dict]) -> str:
    """Apply search/replace edits in order; each `old` must occur exactly once (empty `old` appends)."""
    for i, edit in enumerate(edits, 1):
        old, new = edit["old"], edit["new"]
        if not old:
            content = content + ("" if not content or content.endswith("\n") else "\n") + new
            continue
        count = content.count(old)
        if count != 1:
            problem = "was not found in the file" if count == 0 else f"occurs {count} times, so it is ambiguous"
            raise CoderError(f"edit {i}: `old` {problem}: {old[:200]!r}")
        content = content.replace(old, new)
    return content


def describe(before: str, after: str) -> str:
    """e.g. "+3 -1 lines"."""
    if before == after:
        return "no changes"
    added = removed = 0
    for line in difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm="", n=0):
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return f"+{added} -{removed} lines"


def diff_path(path: Path) -> str:
    """Path relative to the git repository (else to the working directory), as git shows it."""
    path = path.resolve()
    try:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], cwd=path.parent, capture_output=True, text=True, check=True
        ).stdout.strip()
        return path.relative_to(root).as_posix()
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return os.path.relpath(path).replace(os.sep, "/")


def git_diff(path: str, before: str, after: str) -> str:
    """Unified diff in `git diff` format (works with `git apply`)."""
    if before == after:
        return ""
    lines = []
    for line in difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True), f"a/{path}", f"b/{path}"
    ):
        lines.append(line)
        if not line.endswith("\n"):  # last line without a trailing newline
            lines.append("\n\\ No newline at end of file\n")
    return f"diff --git a/{path} b/{path}\n" + "".join(lines)
