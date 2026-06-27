#!/usr/bin/env python3
"""Fire a CONDENSE-TRIGGERING workload at the DFlash server.

To force vLLM's InputBatch.condense (the slot-move path our fix targets) we
need: (a) MORE total requests than slots so a queue exists, and (b) a SPREAD of
output lengths so short requests FINISH while long ones are still decoding ->
the freed low slots get backfilled by still-running / queued long requests ->
slot moves -> condense.

Usage: fire_condense.py <n_total> [max_seqs]
  n_total : total requests to fire (use ~2x max_seqs, e.g. 64 for 32 slots).
Greedy (temp 0), seed 0. Reads server at 127.0.0.1:8000.
Prints per-request status + a histogram of output lengths so we can confirm the
staggered-finish (and therefore condense) actually happened.
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
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,  # force exactly max_tokens so lengths are
                                   # deterministic and the spread is controlled
        "temperature": 0.0,
        "seed": 0,
        "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1200) as r:
            data = json.loads(r.read())
        txt = data["choices"][0]["text"]
        usage = data.get("usage", {})
        ct = usage.get("completion_tokens", -1)
        results[idx] = (True, prompt[:30], txt, ct, max_tokens, time.time() - t0)
    except Exception as e:
        results[idx] = (False, prompt[:30], repr(e), -1, max_tokens,
                        time.time() - t0)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    # A wide spread of output lengths: a cluster of SHORT requests (finish
    # quickly, freeing slots early) mixed with LONG requests (still decoding when
    # the short ones finish) -> guaranteed staggered finishes -> condense.
    # Pattern repeats so every prompt appears at several lengths.
    length_cycle = [32, 64, 96, 128, 256, 512, 1024, 1536]
    results = [None] * n
    threads = []
    for i in range(n):
        prompt = PROMPTS[i % len(PROMPTS)]
        max_tokens = length_cycle[i % len(length_cycle)]
        t = threading.Thread(target=fire,
                             args=(i, prompt, max_tokens, results))
        threads.append(t)
        # Fire all near-simultaneously so the first batch fills all 32 slots and
        # the rest queue; tiny stagger to avoid a thundering-herd connect.
        time.sleep(0.02)
        t.start()
    for t in threads:
        t.join()

    ok = sum(1 for r in results if r and r[0])
    print(f"=== condense workload n={n}: {ok}/{n} succeeded ===")
    # finish-time spread proves staggered completion (condense pressure)
    finish_times = sorted(r[5] for r in results if r and r[0])
    if finish_times:
        print(f"finish times: min={finish_times[0]:.1f}s "
              f"med={finish_times[len(finish_times)//2]:.1f}s "
              f"max={finish_times[-1]:.1f}s "
              f"(spread={finish_times[-1]-finish_times[0]:.1f}s)")
    # confirm requested lengths were honored (min_tokens) -> deterministic spread
    bad_len = [(i, r[3], r[4]) for i, r in enumerate(results)
               if r and r[0] and r[3] != r[4]]
    if bad_len:
        print(f"WARNING: {len(bad_len)} reqs did not hit requested length "
              f"(first few: {bad_len[:5]})")
    for i, r in enumerate(results):
        if r is None:
            print(f"[{i:2d}] NO RESULT")
            continue
        success, tag, txt, ct, req_len, dt = r
        if not success:
            print(f"[{i:2d}] ERR ({dt:.1f}s) {tag!r} -> {txt!r}")
            continue
        oneline = txt.replace("\n", "\\n")[:60]
        print(f"[{i:2d}] OK  ({dt:6.1f}s) len={ct:5d}/{req_len:<5d} "
              f"{tag!r} -> {oneline!r}")


if __name__ == "__main__":
    main()
