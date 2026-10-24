"""Speed tests for the local decider.

Run:  uv run bench_decider.py                 # best device, default dtype
      uv run bench_decider.py --fp32          # MPS/CPU in the upstream fp32 path
      uv run bench_decider.py --device cpu -n 5
"""

import argparse
import statistics
import time

import torch

import decider

QUESTIONS = {
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
}
TICKET = "My integration keeps failing with a 401 since we rotated our API key. Please help. "


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def timed(fn, device: str, n: int) -> list[float]:
    times = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        sync(device)
        times.append((time.perf_counter() - start) * 1000)
    return times


def report(label: str, times: list[float], tokens: int, items: int = 1) -> None:
    p50 = statistics.median(times)
    p95 = sorted(times)[max(0, round(0.95 * len(times)) - 1)]
    print(
        f"{label:<34} p50 {p50:8.1f} ms   p95 {p95:8.1f} ms   "
        f"{items * 1000 / p50:7.1f} req/s   {tokens * 1000 / p50:9.0f} tok/s"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--fp32", action="store_true", help="disable bf16 on MPS/CPU")
    parser.add_argument("-n", type=int, default=20, help="iterations per test")
    args = parser.parse_args()

    device = args.device or decider.best_device()
    print(f"device={device}  torch={torch.__version__}")

    start = time.perf_counter()
    d = decider.load(device, bf16=False if args.fp32 else None)
    print(f"load + warmup: {time.perf_counter() - start:.1f} s   dtype={next(d.backbone.parameters()).dtype}\n")

    request = {"state": TICKET, "questions": QUESTIONS}
    for _ in range(3):  # warm kernels / allocator
        d.decide(request)

    # 1) latency: one request, three questions, short state
    tokens = d.decide(request)["usage"]["input_tokens"]
    report("single request (3 questions)", timed(lambda: d.decide(request), device, args.n), tokens)

    # 2) one question vs three: joint questions share one forward pass
    one = {"state": TICKET, "questions": {"urgent": QUESTIONS["urgent"]}}
    tokens = d.decide(one)["usage"]["input_tokens"]
    report("single request (1 question)", timed(lambda: d.decide(one), device, args.n), tokens)

    # 3) state length scaling
    for repeat in (10, 50, 200):
        long = {"state": TICKET * repeat, "questions": QUESTIONS}
        tokens = d.decide(long)["usage"]["input_tokens"]
        report(f"long state ({tokens} tokens)", timed(lambda: d.decide(long), device, max(3, args.n // 4)), tokens)

    # 4) batch throughput
    for size in (8, 32):
        batch = [{"state": f"Ticket #{i}: {TICKET}", "questions": QUESTIONS} for i in range(size)]
        tokens = sum(r["usage"]["input_tokens"] for r in d.decide(batch))
        report(f"batch of {size}", timed(lambda: d.decide(batch), device, max(3, args.n // 4)), tokens, size)


if __name__ == "__main__":
    main()
