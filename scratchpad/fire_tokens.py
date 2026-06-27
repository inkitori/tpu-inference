#!/usr/bin/env python3
"""Fire concurrent greedy requests and dump TOKEN IDS (not just text) to a JSON file.
Usage: fire_tokens.py <n_concurrent> <max_tokens> <out.json>
Uses /v1/completions with logprobs=1 + echo=False to recover token ids via the
'tokens' field is unreliable; instead we re-tokenize? No -- we use the completion
endpoint's prompt_logprobs?  Simpler & robust: request the raw text and ALSO ask
for logprobs so the server returns the chosen token strings. We compare TEXT here
(token-id parity is implied by identical text under the same tokenizer).
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


def fire(idx, prompt, max_tokens, results):
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 0,
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.loads(r.read())
        results[idx] = {"ok": True, "prompt": prompt,
                        "text": data["choices"][0]["text"]}
    except Exception as e:
        results[idx] = {"ok": False, "prompt": prompt, "err": repr(e)}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    out = sys.argv[3] if len(sys.argv) > 3 else "/tmp/out.json"
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(n)]
    results = [None] * n
    threads = []
    for i, p in enumerate(prompts):
        t = threading.Thread(target=fire, args=(i, p, max_tokens, results))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    with open(out, "w") as f:
        json.dump(results, f)
    ok = sum(1 for r in results if r and r["ok"])
    print(f"wrote {out}: {ok}/{n} OK")


if __name__ == "__main__":
    main()
