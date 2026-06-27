#!/usr/bin/env python3
"""Fire N concurrent greedy long-generation requests at the spec-on server.

Smoke goal: confirm the server survives c=N concurrent requests each generating
~4096 tokens at max-model-len 4224 WITHOUT OOM, and returns coherent output.
NO logprobs (spec-on logprobs path crashes the engine).
"""
import concurrent.futures as cf
import json
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/completions"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 32
MAXTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 4000

PROMPTS = [
    "The capital of France is",
    "The first president of the United States was",
    "The chemical symbol for gold is",
    "The speed of light in a vacuum is approximately",
    "The author of the Harry Potter series is",
    "The largest planet in our solar system is",
    "Water is made of hydrogen and",
    "Two plus two equals",
]


def one(i):
    prompt = PROMPTS[i % len(PROMPTS)]
    body = json.dumps({
        "model": "openai/gpt-oss-20b",
        "prompt": prompt,
        "max_tokens": MAXTOK,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1200) as r:
            data = json.loads(r.read())
        txt = data["choices"][0]["text"]
        ntok = data.get("usage", {}).get("completion_tokens", -1)
        dt = time.time() - t0
        return (i, True, ntok, dt, prompt, txt[:80].replace("\n", " "))
    except Exception as e:
        return (i, False, -1, time.time() - t0, prompt, f"ERROR: {e!r}")


def main():
    print(f"Firing {N} concurrent requests, max_tokens={MAXTOK}, greedy ...",
          flush=True)
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=N) as ex:
        results = list(ex.map(one, range(N)))
    ok = sum(1 for r in results if r[1])
    print(f"\n=== {ok}/{N} succeeded in {time.time()-t0:.1f}s ===", flush=True)
    for i, success, ntok, dt, prompt, head in sorted(results):
        tag = "OK " if success else "FAIL"
        print(f"[{tag}] req{i:02d} ntok={ntok} {dt:6.1f}s | {prompt!r} -> {head!r}",
              flush=True)
    sys.exit(0 if ok == N else 1)


if __name__ == "__main__":
    main()
