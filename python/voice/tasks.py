"""Decides what speech means for the code: discard, wait, new task, update a task, or cancel (undo).

Runs the local pplx-decider (GGUF) on the speech not consumed yet, also mid-sentence, so an event can fire before
the speaker pauses (is_final=False) and fires again with the full sentence (is_final=True). Each spoken request
is one step; cancel undoes only the last step of a task, and only once the sentence is finished.
"""

import asyncio
import dataclasses
import itertools
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .transcriber import Transcript

# Kept short: the decider reads ~400 prompt tokens/s, so every word costs latency.
INSTRUCTIONS = "A developer talks to a coding voice assistant. What to do with this speech?"
ACTIONS = {
    "discard": "filler (okay, hmm, yeah), small talk, not about coding",
    "wait": "incomplete, more coming",
    "new": "a new coding task",
    "update": "adds to, continues or changes a task",
    "cancel": "never mind / undo or revert the last change",
}
TASK_ACTIONS = ("new", "update", "cancel")
TASK_NAMES = ("newest", "previous", "earlier", "oldest")  # picked the right task more often than ids


@dataclass
class Task:
    id: int
    parts: list[str]  # the request, then each update
    steps: list[int] = field(default_factory=list)  # step id of each part
    updated_at: float = field(default_factory=time.time)

    @property
    def text(self) -> str:
        return " ".join(self.parts)


@dataclass
class TaskEvent:
    kind: str  # "new" | "update" | "cancel"
    task_id: int
    text: str  # new/update: the whole task so far; cancel: the request being undone
    said: str  # the speech behind this event (new/update: this step's request)
    confidence: float
    is_final: bool  # False: decided mid-sentence; repeats with the full sentence
    step: int  # one spoken request; cancel: the step to undo


@dataclass
class Decision:
    choice: str  # an ACTIONS key, or "dropped" when waiting text went stale
    probabilities: dict[str, float]
    finished: float  # P(the sentence is complete)
    target: int | None  # task id for update/cancel
    text: str
    seconds: float


class TaskDecider:
    """Transcripts go into `speech_in`, TaskEvents come out of `events` (every decision also out of `decisions`)."""

    def __init__(
        self,
        *,
        interval: float = 0.2,  # how often to check for new speech
        max_wait: float = 15.0,  # drop waiting text after this long without new speech
        max_chars: int = 300,  # only the most recent speech goes to the model
        max_task_chars: int = 150,  # shorter cuts broke the "finished" check
        recent_tasks: int = 3,  # tasks update/cancel can refer to
        task_ttl: float = 120.0,  # a task stops being recent after this long without changes
        min_finished: float = 0.4,  # new/update/cancel need P(sentence complete) above this
    ):
        self.interval = interval
        self.max_wait = max_wait
        self.max_chars = max_chars
        self.max_task_chars = max_task_chars
        self.recent_tasks = recent_tasks
        self.task_ttl = task_ttl
        self.min_finished = min_finished
        self.speech_in: asyncio.Queue[Transcript] = asyncio.Queue()
        self.events: asyncio.Queue[TaskEvent] = asyncio.Queue()
        self.decisions: asyncio.Queue[Decision] = asyncio.Queue()
        self.tasks: dict[int, Task] = {}
        self.pending: list[str] = []  # final utterances not consumed yet
        self.interim = ""  # the utterance still being spoken
        self.on_finished = None  # optional callback(interim text, P(finished)) after each decision on it

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="decider")
        self._decider = None
        self._ids = itertools.count(1)
        self._steps = itertools.count(1)
        self._open: tuple[TaskEvent, str] | None = None  # event decided mid-sentence, speech before that sentence
        self._last = ("", "")  # (text, choice) of the last decision
        self._held: Decision | None = None  # a cancel decided mid-sentence, waiting for the sentence to end

    def recent(self) -> list[Task]:
        """Recent tasks by latest change, so "revert it" hits the most recent change, not the newest task."""
        now = time.time()
        live = [t for t in self.tasks.values() if now - t.updated_at < self.task_ttl]
        live.sort(key=lambda t: t.steps[-1])
        return live[-self.recent_tasks :]

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._load)
        last_speech = time.monotonic()
        while True:
            await asyncio.sleep(self.interval)
            new = False
            while not self.speech_in.empty():
                new |= self._take(self.speech_in.get_nowait())
                last_speech = time.monotonic()

            text = _tail(" ".join(self.pending + [self.interim]).strip(), self.max_chars)
            if new and text == self._last[0]:  # e.g. the final of an interim we already decided on
                if self._last[1] == "discard":
                    self.pending.clear()
                elif self._held and not self.interim:  # the held cancel's sentence is now finished
                    held, self._held = self._held, None
                    self._last = (text, held.choice)
                    await self.decisions.put(held)
                    await self._act(held)
                continue
            if not new:  # the same text would get the same answer
                if self.pending and time.monotonic() - last_speech > self.max_wait:
                    await self.decisions.put(Decision("dropped", {}, 1.0, None, " ".join(self.pending), 0.0))
                    self.pending.clear()
                continue

            interim = self.interim
            decision = await loop.run_in_executor(self._executor, self._decide, text, self.recent())
            if interim and self.on_finished:
                self.on_finished(interim, decision.finished)
            self._held = None
            if decision.choice == "cancel" and self.interim:
                # Undo only on a finished sentence ("okay no, remove it from here and..." is an update).
                self._held = decision
                decision = dataclasses.replace(decision, choice="wait")
            self._last = (text, decision.choice)
            await self.decisions.put(decision)
            await self._act(decision)

    async def _act(self, decision: Decision) -> None:
        if decision.choice == "wait":
            return
        prefix = " ".join(self.pending)
        self.pending.clear()
        if decision.choice == "discard":
            return  # the utterance still being spoken stays: it may turn into something
        event = self._apply(decision, decision.text)
        if self.interim:  # mid-sentence: the rest of this utterance completes the event
            self._open = (event, prefix)
            self.interim = ""
        await self.events.put(event)

    def _apply(self, decision: Decision, text: str) -> TaskEvent:
        confidence = decision.probabilities[decision.choice]
        step = next(self._steps)
        if decision.choice == "cancel":
            task = self.tasks[decision.target]
            undone, undone_step = task.parts.pop(), task.steps.pop()
            if not task.parts:
                del self.tasks[task.id]
            return TaskEvent("cancel", task.id, undone, text, confidence, is_final=True, step=undone_step)
        if decision.choice == "new":
            task = Task(next(self._ids), [text], [step])
            self.tasks[task.id] = task
        else:
            task = self.tasks[decision.target]
            task.parts.append(text)
            task.steps.append(step)
            task.updated_at = time.time()
        return TaskEvent(decision.choice, task.id, task.text, text, confidence, is_final=not self.interim, step=step)

    def _take(self, t: Transcript) -> bool:
        """Add one transcript; returns whether the text to decide on changed."""
        if self._open:  # this utterance was already acted on mid-sentence
            if t.is_final:
                event, prefix = self._open
                self._open = None
                said = f"{prefix} {t.text}".strip()
                text = event.text
                if task := self.tasks.get(event.task_id):
                    task.parts[-1] = said
                    text = task.text
                self.events.put_nowait(dataclasses.replace(event, text=text, said=said, is_final=True))
            return False
        if t.is_final:
            self.pending.append(t.text)
            self.interim = ""
        else:
            self.interim = t.text
        return True

    def _load(self) -> None:
        from decider_gguf import GGUFDecider

        t = time.perf_counter()
        self._decider = GGUFDecider(max_length=1024)
        self._decide("warm up", [])
        print(f"[decider] ready in {time.perf_counter() - t:.1f}s")

    def _decide(self, text: str, recent: list[Task]) -> Decision:
        t = time.perf_counter()
        actions = ACTIONS if recent else {k: v for k, v in ACTIONS.items() if k not in ("update", "cancel")}
        questions = {
            "action": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": actions},
            # Catches "Move the retry logic" before "...into the api client" arrives.
            "finished": {"type": "noul", "instructions": "Did the developer finish their sentence?"},
        }
        state = {}
        names = dict(zip(TASK_NAMES, reversed(recent)))  # name -> task, newest first
        if recent:
            state["tasks"] = {name: names[name].text[: self.max_task_chars] for name in reversed(names)}
        if len(recent) > 1:
            questions["target"] = {
                "type": "choice",
                "instructions": "Which task does the speech refer to? If it does not name or describe one, the newest.",
                "criteria": dict.fromkeys(names),
            }
        state["speech"] = text

        answers = self._decider.decide({"state": state, "questions": questions})["answers"]
        choice, finished = answers["action"]["choice"], answers["finished"]["noul"]
        if choice in TASK_ACTIONS and finished < self.min_finished:
            choice = "wait"
        target = None
        if choice in ("update", "cancel"):
            target = recent[-1].id
            if "target" in answers:  # vague speech is a coin flip: only leave the newest task when sure
                probabilities = answers["target"]["probabilities"]
                if probabilities["newest"] < 0.2:
                    target = names[max((n for n in names if n != "newest"), key=probabilities.get)].id
        probabilities = answers["action"]["probabilities"]
        return Decision(choice, probabilities, finished, target, text, time.perf_counter() - t)


def _tail(text: str, limit: int) -> str:
    """The last `limit` characters, starting at a whole word."""
    if len(text) <= limit:
        return text
    cut = text[-limit:]
    return cut[cut.find(" ") + 1 :] if " " in cut else cut
