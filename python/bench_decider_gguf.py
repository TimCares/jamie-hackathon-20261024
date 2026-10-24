"""GGUF decider: answer check against the reference model, then speed tests.

Run:  uv run --extra gguf bench_decider_gguf.py              # quality check + speed
      uv run --extra gguf bench_decider_gguf.py --no-compare # speed only (skips loading the 9 GB reference)
"""

import argparse
import gc
import time

from bench_decider import QUESTIONS, TICKET

TEAM = QUESTIONS["team"]
GOLDEN = [
    {"state": TICKET, "questions": QUESTIONS},
    {"state": "I was charged twice for my October invoice, please refund one.", "questions": {"team": TEAM}},
    {"state": "Do you offer volume discounts if we buy 500 seats?", "questions": {"team": TEAM}},
    {"state": "The webhook returns 500 errors since this morning and our checkout is down!!!", "questions": QUESTIONS},
    {"state": "Thanks, all good now, appreciate the quick help :)", "questions": {
        "frustration": QUESTIONS["frustration"], "urgent": QUESTIONS["urgent"]}},
    {"state": "def add(a, b): return a - b", "questions": {
        "bug": {"type": "noul", "instructions": "Does this function have a bug, given its name?"}}},
    {"state": "Paris is the capital of Germany.", "questions": {
        "true": {"type": "noul", "instructions": "Is the statement factually correct?"}}},
    {"state": "The movie was a total waste of two hours.", "questions": {
        "sentiment": {"type": "choice", "criteria": {"positive": None, "neutral": None, "negative": None}}}},
    {"state": "Meeting moved to Thursday 3pm, room B.", "questions": {
        "kind": {"type": "choice", "instructions": "What kind of message is this?",
                 "criteria": {"scheduling": None, "complaint": None, "spam": None, "question": None}}}},
    {"state": "WIN A FREE IPHONE!!! click http://totally-legit.example now", "questions": {
        "spam": {"type": "noul", "instructions": "Is this spam?"}}},
    {"state": {"user": "alice", "role": "viewer", "action": "delete_project"}, "questions": {
        "allow": {"type": "noul", "instructions": "Should a viewer be allowed to perform this action?"}}},
    {"state": "Answer: The derivative of x^2 is 2x.", "questions": {
        "quality": {"type": "score", "instructions": "How correct is the answer?",
                    "criteria": ["wrong", "partially correct", "correct"]}}},
]


def probs(answer: dict) -> list[float]:
    if answer["type"] == "noul":
        return [1 - answer["noul"], answer["noul"]]
    return list(answer["probabilities"].values())


def label(answer: dict) -> str:
    if answer["type"] == "noul":
        return "yes" if answer["noul"] >= 0.5 else "no"
    if answer["type"] == "score":
        return str(max(answer["probabilities"], key=answer["probabilities"].get))
    return answer["choice"]


def compare() -> None:
    import torch

    import decider
    import decider_gguf

    print("reference (fp32, upstream path) ...")
    ref_model = decider.load(bf16=False)
    reference = [ref_model.decide(r) for r in GOLDEN]
    del ref_model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    gguf = decider_gguf.get_decider()
    agree, total, worst = 0, 0, 0.0
    for request, ref in zip(GOLDEN, reference):
        got = gguf.decide(request)
        for key, ref_answer in ref["answers"].items():
            answer = got["answers"][key]
            delta = max(abs(a - b) for a, b in zip(probs(ref_answer), probs(answer)))
            same = label(ref_answer) == label(answer)
            agree += same
            total += 1
            worst = max(worst, delta)
            flag = "ok " if same else "DIFF"
            print(f"  {flag} {key:<12} ref={label(ref_answer):<10} gguf={label(answer):<10} max|Δp|={delta:.3f}"
                  f"   {str(request['state'])[:50]!r}")
    print(f"\nagreement {agree}/{total}   worst max|Δp| {worst:.3f}\n")


def speed(n: int) -> None:
    import decider_gguf
    from bench_decider import report

    start = time.perf_counter()
    d = decider_gguf.get_decider()
    print(f"load: {time.perf_counter() - start:.1f} s  (0 if the quality check already loaded it)")
    request = {"state": TICKET, "questions": QUESTIONS}
    for _ in range(3):
        d.decide(request)

    def timed(fn, iterations):
        times = []
        for _ in range(iterations):
            t = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t) * 1000)
        return times

    tokens = d.decide(request)["usage"]["input_tokens"]
    report("single request (3 questions)", timed(lambda: d.decide(request), n), tokens)
    one = {"state": TICKET, "questions": {"urgent": QUESTIONS["urgent"]}}
    tokens = d.decide(one)["usage"]["input_tokens"]
    report("single request (1 question)", timed(lambda: d.decide(one), n), tokens)
    for repeat in (10, 50, 200):
        long = {"state": TICKET * repeat, "questions": QUESTIONS}
        tokens = d.decide(long)["usage"]["input_tokens"]
        report(f"long state ({tokens} tokens)", timed(lambda: d.decide(long), max(3, n // 4)), tokens)
    for size in (8, 32):
        batch = [{"state": f"Ticket #{i}: {TICKET}", "questions": QUESTIONS} for i in range(size)]
        tokens = sum(r["usage"]["input_tokens"] for r in d.decide(batch))
        report(f"batch of {size}", timed(lambda: d.decide(batch), max(3, n // 4)), tokens, size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-compare", action="store_true")
    parser.add_argument("-n", type=int, default=20)
    args = parser.parse_args()
    if not args.no_compare:
        compare()
    speed(args.n)
