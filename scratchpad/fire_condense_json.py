#!/usr/bin/env python3
"""Same condense-triggering workload as fire_condense.py, but dumps the FULL
completion text per request to a JSON file for a rigorous spec-on vs target-only
diff. Usage: fire_condense_json.py <n_total> <out.json>
Greedy (temp 0), seed 0, min_tokens==max_tokens. No logprobs (crashes spec-on).
"""
import sys, json, time, threading, urllib.request

URL = "http://127.0.0.1:8000/v1/completions"
MODEL = "openai/gpt-oss-20b"

PROMPTS = [
    "The capital of France is",
    "List the first five prime numbers:",
    "The chemical symbol for gold is",
    "The largest planet in our solar system is",
    "Name three primary colors:",
    "Water is composed of hydrogen and",
    "2 + 2 =",
    "The first president of the United States was",
]


def fire(idx, prompt, max_tokens, results):
    body = json.dumps({
        "model": MODEL, "prompt": prompt,
        "max_tokens": max_tokens, "min_tokens": max_tokens,
        "temperature": 0.0, "seed": 0, "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1200) as r:
            data = json.loads(r.read())
        txt = data["choices"][0]["text"]
        ct = data.get("usage", {}).get("completion_tokens", -1)
        results[idx] = {"ok": True, "prompt": prompt, "len": max_tokens,
                        "ct": ct, "text": txt, "dt": time.time() - t0}
    except Exception as e:
        results[idx] = {"ok": False, "prompt": prompt, "len": max_tokens,
                        "err": repr(e), "dt": time.time() - t0}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    out = sys.argv[2] if len(sys.argv) > 2 else "fire_out.json"
    length_cycle = [32, 64, 96, 128, 256, 512, 1024, 1536]
    results = [None] * n
    threads = []
    for i in range(n):
        prompt = PROMPTS[i % len(PROMPTS)]
        max_tokens = length_cycle[i % len(length_cycle)]
        t = threading.Thread(target=fire, args=(i, prompt, max_tokens, results))
        threads.append(t)
        time.sleep(0.02)
        t.start()
    for t in threads:
        t.join()
    ok = sum(1 for r in results if r and r["ok"])
    print(f"=== n={n}: {ok}/{n} succeeded, wrote {out} ===")
    ft = sorted(r["dt"] for r in results if r and r["ok"])
    if ft:
        print(f"finish times: min={ft[0]:.1f}s med={ft[len(ft)//2]:.1f}s "
              f"max={ft[-1]:.1f}s (spread={ft[-1]-ft[0]:.1f}s)")
    with open(out, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
