#!/usr/bin/env python3
"""Compare two output JSONs (spec-on vs target-only) per prompt.
Usage: compare_outputs.py specon.json target.json
Reports, per request: whether text is IDENTICAL, and if not, the char index of
first divergence + the surrounding context (so a human can judge near-tie vs bug).
"""
import sys, json

a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))

assert len(a) == len(b), f"length mismatch {len(a)} vs {len(b)}"

n_identical = 0
for i, (ra, rb) in enumerate(zip(a, b)):
    if not (ra.get("ok") and rb.get("ok")):
        print(f"[{i}] SKIP (a.ok={ra.get('ok')} b.ok={rb.get('ok')})")
        continue
    ta, tb = ra["text"], rb["text"]
    prompt = ra["prompt"][:40]
    if ta == tb:
        n_identical += 1
        print(f"[{i}] IDENTICAL ({len(ta)} chars) prompt={prompt!r}")
    else:
        # find first diverging char
        j = 0
        m = min(len(ta), len(tb))
        while j < m and ta[j] == tb[j]:
            j += 1
        # how many chars agreed before divergence
        agreed = ta[:j]
        print(f"[{i}] DIVERGE at char {j} (agreed {j} chars) prompt={prompt!r}")
        print(f"     agreed prefix : ...{agreed[-30:]!r}")
        print(f"     spec-on  cont : {ta[j:j+40]!r}")
        print(f"     target   cont : {tb[j:j+40]!r}")

print(f"\n=== {n_identical}/{len(a)} requests TOKEN-IDENTICAL (text-identical) ===")
