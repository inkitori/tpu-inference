#!/usr/bin/env python3
"""Greedy-completion correctness gate for Hy3 serving changes.

Usage: correctness_check.py <label>
Saves completions to ~/hy3_bench_artifacts/correctness_<label>.json and, if a
'baseline' file exists and label != baseline, diffs against it.
"""
import json
import sys
import urllib.request
from pathlib import Path

BASE = "http://localhost:8000/v1/completions"
MODEL = "/dev/shm/models/Hy3-4bit-mtp-mlx"
OUTDIR = Path(__file__).parent

PROMPTS = [
    "The capital of France is",
    "Explain the difference between a process and a thread in operating systems.",
    "Write a Python function that returns the nth Fibonacci number using iteration.",
    "Translate to French: 'The weather is beautiful today and I plan to go hiking.'",
    "List the first 10 prime numbers, separated by commas.",
    "What causes the seasons on Earth? Answer in two sentences.",
    "Complete this SQL query to find the top 5 customers by total order value:\nSELECT customer_id,",
    "Summarize the plot of Romeo and Juliet in one paragraph.",
    "def quicksort(arr):\n    ",
    "A train travels 120 km in 1.5 hours. What is its average speed in km/h? Show your work.",
]


def complete(prompt: str) -> str:
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": 200,
        "temperature": 0.0,
        "seed": 0,
    }).encode()
    req = urllib.request.Request(BASE, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["choices"][0]["text"]


def main():
    label = sys.argv[1]
    out = {}
    for i, p in enumerate(PROMPTS):
        out[p] = complete(p)
        print(f"[{i+1}/{len(PROMPTS)}] ok ({len(out[p])} chars)")
    path = OUTDIR / f"correctness_{label}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path}")

    base_path = OUTDIR / "correctness_baseline.json"
    if label != "baseline" and base_path.exists():
        base = json.loads(base_path.read_text())
        n_exact = sum(1 for p in PROMPTS if base.get(p) == out[p])
        print(f"exact match vs baseline: {n_exact}/{len(PROMPTS)}")
        for p in PROMPTS:
            if base.get(p) != out[p]:
                b, n = base.get(p, ""), out[p]
                k = next((j for j in range(min(len(b), len(n))) if b[j] != n[j]),
                         min(len(b), len(n)))
                print(f"--- DIFF prompt: {p[:60]!r} (first divergence at char {k})")
                print(f"  base: ...{b[max(0,k-40):k+80]!r}")
                print(f"  new : ...{n[max(0,k-40):k+80]!r}")


if __name__ == "__main__":
    main()
