#!/usr/bin/env python3
"""Fire concurrent greedy requests at the DFlash server and report outputs.
Usage: fire_concurrent.py <mode> <n_concurrent>
  mode: uniform | ragged
Reads server at 127.0.0.1:8000. Greedy (temp 0).
Prints each request's prompt-tag and completion text.
"""
import sys, json, time, threading, urllib.request

URL = "http://127.0.0.1:8000/v1/completions"
MODEL = "openai/gpt-oss-20b"

UNIFORM_PROMPTS = [
    "The capital of France is",
    "The first president of the United States was",
    "The chemical symbol for gold is",
    "The speed of light in a vacuum is approximately",
    "The author of the Harry Potter series is",
    "The largest planet in our solar system is",
    "Water is composed of hydrogen and",
    "The square root of 144 is",
]

# Ragged: deliberately different prompt LENGTHS (short -> long) to make the
# batch contain requests at different decode positions in the same step.
RAGGED_PROMPTS = [
    "Hi.",
    "List the first five prime numbers:",
    "Count from one to ten:",
    ("Once upon a time in a small village nestled between two great mountains, "
     "there lived an old clockmaker who was known throughout the land for his "
     "extraordinary craftsmanship. Every morning he would open his shop and "
     "continue the story of his life by telling visitors about"),
    "Name three primary colors:",
    ("Explain in one sentence why the sky appears blue during the day, taking "
     "into account the scattering of sunlight by the atmosphere, which is a "
     "phenomenon known as"),
    "2 + 2 =",
    ("The history of computing spans many centuries, from the abacus to modern "
     "supercomputers. One of the most important early figures in this history was"),
]


def fire(idx, prompt, max_tokens, results, stagger):
    if stagger:
        time.sleep(idx * 0.12)  # staggered arrival -> mixed decode positions
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 0,
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.loads(r.read())
        txt = data["choices"][0]["text"]
        results[idx] = (True, prompt[:40], txt, time.time() - t0)
    except Exception as e:
        results[idx] = (False, prompt[:40], repr(e), time.time() - t0)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "uniform"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    if mode == "ragged":
        base = RAGGED_PROMPTS
        stagger = True
        max_tokens = 48
    else:
        base = UNIFORM_PROMPTS
        stagger = False
        max_tokens = 32
    prompts = [base[i % len(base)] for i in range(n)]
    results = [None] * n
    threads = []
    for i, p in enumerate(prompts):
        t = threading.Thread(target=fire, args=(i, p, max_tokens, results, stagger))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    ok = sum(1 for r in results if r and r[0])
    print(f"=== {mode} n={n}: {ok}/{n} succeeded ===")
    for i, r in enumerate(results):
        if r is None:
            print(f"[{i}] NO RESULT")
            continue
        success, tag, txt, dt = r
        flag = "OK " if success else "ERR"
        oneline = txt.replace("\n", "\\n")[:120]
        print(f"[{i}] {flag} ({dt:.1f}s) {tag!r} -> {oneline!r}")


if __name__ == "__main__":
    main()
