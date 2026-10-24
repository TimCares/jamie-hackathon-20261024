"""Applies task events to one file, one at a time, so the file stays consistent.

new/update: the coder edits the current file (a sentence's final wording only as a delta to its early one).
cancel: put back the file from before that step and replay later changes; one that no longer applies is
regenerated from its request. Requests are the source of truth; edits can always be regenerated.
"""

import asyncio
import dataclasses
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .coder import Coder, CoderError, CoderResult, apply_edits, git_diff

if TYPE_CHECKING:
    from .tasks import TaskEvent


@dataclass
class FileUpdate:
    kind: str  # "new" | "update" | "undo" | "noop" | "failed" | "external"
    task_id: int | None
    summary: str
    diff: str = ""
    seconds: float = 0.0
    is_final: bool = True  # False when made from an early (mid-sentence) event
    speed: str | None = None  # "standard" when fast mode was unavailable


@dataclass
class Change:
    task_id: int
    step: int
    request: str  # to regenerate the change if it no longer applies
    edits: list[dict]
    before: str  # the whole file before the change, so undo is exact


@dataclass
class Step:
    id: int
    request: str


@dataclass
class Task:
    steps: list[Step] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)


class FileWriter:
    """TaskEvents go into handle(), FileUpdates come out of `updates`."""

    def __init__(self, coder: Coder, *, recent_tasks: int = 3, task_ttl: float = 120.0, prewarm: bool = True):
        self.coder = coder
        self.path = coder.path
        self.recent_tasks = recent_tasks
        self.task_ttl = task_ttl
        self.prewarm = prewarm
        self.content = self.path.read_text()  # what we last wrote (or read)
        self.tasks: dict[int, Task] = {}
        self.log: list[Change] = []  # undoable changes, oldest first
        self.updates: asyncio.Queue[FileUpdate] = asyncio.Queue()

        self._pending: dict[tuple[int, int], "TaskEvent"] = {}  # latest event per (task, step)
        self._wake = asyncio.Event()
        self._running: tuple[int, asyncio.Task, str] | None = None  # step, its coder call, the request
        self._warmed: str | None = None
        self._warm_job: asyncio.Task | None = None

    def handle(self, event: "TaskEvent") -> None:
        """Queue an event. A cancel or a new wording for the step being coded stops that call."""
        if self._running and self._running[0] == event.step:
            _, job, request = self._running
            if event.kind == "cancel" or not _same(request, event.said):
                job.cancel()
        self._pending[(event.task_id, event.step)] = event
        self._wake.set()

    async def run(self) -> None:
        self._warm()
        while True:
            await self._wake.wait()
            self._wake.clear()
            while self._pending:
                await self._process(self._next())
                self._forget_old()
            self._warm()

    def _next(self) -> "TaskEvent":
        """Undos first (instant), then requests not applied yet, then corrections of applied ones."""
        applied = {s.id for task in self.tasks.values() for s in task.steps}

        def rank(item):
            (_, step), event = item
            return (0 if event.kind == "cancel" else 2 if step in applied else 1, step)

        key = min(self._pending.items(), key=rank)[0]
        return self._pending.pop(key)

    async def _process(self, event: "TaskEvent") -> None:
        self._sync_disk()
        if event.kind == "cancel":
            await self._undo(event)
            return
        task = self.tasks.get(event.task_id)
        current = next((s for s in task.steps if s.id == event.step), None) if task else None  # same step again
        if current and _same(current.request, event.said):
            return
        earlier = " ".join(self._wording(event.task_id, s) for s in task.steps if s is not current) if task else ""
        previous = current.request if current else None
        result = await self._code(event.task_id, event.step, self.content, event.said, previous, earlier)
        if result is None:
            return
        if self._sync_disk():  # edited on disk during the call: redo it on that version
            self._pending.setdefault((event.task_id, event.step), event)
            return

        task = self.tasks.setdefault(event.task_id, Task())
        if current:
            current.request = event.said
        else:
            task.steps.append(Step(event.step, event.said))
        task.updated_at = time.time()
        kind = "update" if current else event.kind
        speed = result.usage.get("speed")
        if result.after == self.content:
            self._emit(FileUpdate("noop", event.task_id, result.summary, "", result.seconds, event.is_final, speed))
            return
        self.log.append(Change(event.task_id, event.step, event.said, result.edits, self.content))
        self._write(result.after)
        self._emit(FileUpdate(kind, event.task_id, result.summary, result.diff, result.seconds, event.is_final, speed))

    def _wording(self, task_id: int, step: Step) -> str:
        """A step's latest wording; a queued correction already counts."""
        queued = self._pending.get((task_id, step.id))
        return queued.said if queued and queued.kind != "cancel" else step.request

    async def _undo(self, event: "TaskEvent") -> None:
        if task := self.tasks.get(event.task_id):
            task.steps = [s for s in task.steps if s.id != event.step]
            if not task.steps:
                del self.tasks[event.task_id]
        first = next((i for i, change in enumerate(self.log) if change.step == event.step), None)
        if first is None:
            self._emit(FileUpdate("noop", event.task_id, f"nothing to undo for: {_short(event.text)}"))
            return
        t = time.perf_counter()
        undone = next(c.request for c in reversed(self.log) if c.step == event.step)
        later = [change for change in self.log[first:] if change.step != event.step]
        content = self.log[first].before
        del self.log[first:]

        regenerated: list[int] = []
        for change in later:
            if change.step in regenerated:
                continue
            try:  # edits are anchored by content, so they usually still apply
                after = apply_edits(content, change.edits)
                self.log.append(dataclasses.replace(change, before=content))
                content = after
                continue
            except CoderError:
                pass
            regenerated.append(change.step)
            request = next(c.request for c in reversed(later) if c.step == change.step)
            owner = self.tasks.get(change.task_id)
            earlier = " ".join(s.request for s in owner.steps if s.id < change.step) if owner else ""
            result = await self._code(change.task_id, change.step, content, request, None, earlier)
            if result is not None and result.after != content:
                self.log.append(Change(change.task_id, change.step, request, result.edits, content))
                content = result.after

        if self._sync_disk():
            return  # edited on disk meanwhile: that version wins
        diff = git_diff(self.coder.diff_path, self.content, content)
        self._write(content)
        summary = f"undid: {_short(undone)}"
        if regenerated:
            summary += f" (redid {len(regenerated)} later change{'s' if len(regenerated) > 1 else ''})"
        self._emit(FileUpdate("undo", event.task_id, summary, diff, time.perf_counter() - t, event.is_final))

    async def _code(
        self, task_id: int, step: int, content: str, request: str, previous: str | None, earlier: str
    ) -> CoderResult | None:
        """One coder call; None if it was stopped or failed."""
        job = asyncio.create_task(self.coder.edit(content, request, previous, earlier or None))
        self._running = (step, job, request)
        try:
            result = await job
        except asyncio.CancelledError:
            if asyncio.current_task().cancelling():  # the writer itself is being stopped
                raise
            return None
        except Exception as error:
            self._emit(FileUpdate("failed", task_id, f"{type(error).__name__}: {error}"))
            return None
        finally:
            self._running = None
        return result

    def _forget_old(self) -> None:
        """Drop undo info of tasks that are no longer recent (same rule as the decider)."""
        now = time.time()
        newest = sorted(self.tasks, key=lambda i: max(s.id for s in self.tasks[i].steps))[-self.recent_tasks :]
        recent = {i for i in newest if now - self.tasks[i].updated_at < self.task_ttl}
        while self.log and self.log[0].task_id not in recent:
            self.log.pop(0)

    def _sync_disk(self) -> bool:
        """Take over changes someone else made to the file; returns whether there were any."""
        disk = self.path.read_text()
        if disk == self.content:
            return False
        self.content = disk
        self.log.clear()
        self._emit(FileUpdate("external", None, "file changed on disk; earlier changes can no longer be undone"))
        return True

    def _write(self, content: str) -> None:
        tmp = self.path.with_name(f".{self.path.name}.voice-tmp")  # rename over the file: never half-written
        tmp.write_text(content)
        shutil.copymode(self.path, tmp)
        os.replace(tmp, self.path)
        self.content = content

    def _warm(self) -> None:
        if self.prewarm and self._warmed != self.content:
            self._warmed = self.content
            self._warm_job = asyncio.create_task(self._warm_call(self.content))

    async def _warm_call(self, content: str) -> None:
        try:
            await self.coder.warm(content)
        except Exception as error:
            self._emit(FileUpdate("failed", None, f"cache warm-up: {type(error).__name__}: {error}"))

    def _emit(self, update: FileUpdate) -> None:
        self.updates.put_nowait(update)


def _same(a: str, b: str) -> bool:
    """Same words, ignoring case and punctuation."""
    return _words(a) == _words(b)


def _words(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def _short(text: str, limit: int = 80) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"
