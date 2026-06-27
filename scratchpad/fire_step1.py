#!/usr/bin/env python3
"""Fire N concurrent max_tokens=1 greedy requests, capture the single next token
(via logprobs top-1) for each prompt. This is the MATCHED-SHAPE oracle: both the
spec-on and target-only servers see the identical single-step request shape, so
any argmax difference is a pure bf16 target-forward effect (or a verifier bug if
the spec path emits a non-argmax token).
Usage: fire_step1.py <n_concurrent> <out.json>
"""
import sys, json, threading, urllib.request

URL = "http://127.0.0.1:8000/v1/completions"
MODEL = "openai/gpt-oss-20b"

PROMPTS = [
    "The capital of France is",
    "List the first five prime numbers:",
    "The chemical symbol for gold is",
    "The largest planet in our solar system is",
    "The square root of 144 is",
    "Name three primary colors:",
    "Water is composed of hydrogen and",
    "The first president of the United States was",
]


def fire(idx, prompt, results):
    body = json.dumps({
        "model": MODEL, "prompt": prompt, "max_tokens": 1,
        "temperature": 0.0, "logprobs": 20, "seed": 0,
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        ch = data["choices"][0]
        txt = ch["text"]
        top = ch["logprobs"]["top_logprobs"][0]  # dict token->logprob
        # top-2 gap
        items = sorted(top.items(), key=lambda kv: kv[1], reverse=True)
        gap = (items[0][1] - items[1][1]) if len(items) >= 2 else None
        results[idx] = {"ok": True, "prompt": prompt, "next": txt,
                        "argmax_lp": items[0][1], "gap_nats": gap,
                        "top2": items[:2]}
    except Exception as e:
        results[idx] = {"ok": False, "prompt": prompt, "err": repr(e)}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/step1.json"
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(n)]
    results = [None] * n
    threads = [threading.Thread(target=fire, args=(i, p, results))
               for i, p in enumerate(prompts)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    with open(out, "w") as f:
        json.dump(results, f)
    ok = sum(1 for r in results if r and r["ok"])
    print(f"wrote {out}: {ok}/{n} OK")
    for i, r in enumerate(results):
        if r and r["ok"]:
            print(f"[{i}] next={r['next']!r} gap={r['gap_nats']:.3f}nats {r['prompt'][:30]!r}")
        else:
            print(f"[{i}] ERR {r}")


if __name__ == "__main__":
    main()
