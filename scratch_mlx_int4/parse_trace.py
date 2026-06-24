#!/usr/bin/env python3
"""Parse a JAX/XLA profiler trace.json.gz into a per-op DEVICE-time breakdown.
Usage: python parse_trace.py <prof_dir> [n_decode_steps]
Reports top ops + category totals for ONE TPU device plane (SPMD => one chip ~ all).
"""
import sys, os, gzip, json, glob, re
from collections import defaultdict

prof_dir = sys.argv[1] if len(sys.argv) > 1 else "scratch_mlx_int4/prof"
n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 95  # 96 gen tokens => ~95 decode + 1 prefill

cands = glob.glob(os.path.join(prof_dir, "**", "*.trace.json.gz"), recursive=True)
if not cands:
    cands = glob.glob(os.path.join(prof_dir, "**", "*.trace.json"), recursive=True)
if not cands:
    print("NO trace.json(.gz) found under", prof_dir)
    for f in glob.glob(os.path.join(prof_dir, "**", "*"), recursive=True):
        print("  ", f)
    sys.exit(1)
trace = max(cands, key=os.path.getsize)
print("trace:", trace)
opener = gzip.open if trace.endswith(".gz") else open
with opener(trace, "rt") as f:
    data = json.load(f)
ev = data["traceEvents"]

pid_name = {}
for e in ev:
    if e.get("ph") == "M" and e.get("name") == "process_name":
        pid_name[e["pid"]] = e.get("args", {}).get("name", "")

print("\n== process planes ==")
for pid, nm in sorted(pid_name.items(), key=lambda x: str(x[1])):
    print(f"  pid={pid}  name={nm!r}")

def is_device(nm):
    nm = nm or ""
    return ("/device:TPU" in nm) or ("TPU" in nm and "XLA" in nm) or bool(re.search(r"device:TPU:\d", nm))

dev_pids = [p for p, nm in pid_name.items() if is_device(nm)]
def tpu_id(nm):
    m = re.search(r"TPU:(\d+)", nm or "")
    return int(m.group(1)) if m else 999
chip0 = sorted(dev_pids, key=lambda p: tpu_id(pid_name[p]))[0] if dev_pids else None
print(f"\n== device pids: {[(p, pid_name[p]) for p in dev_pids]} ; using chip0={chip0} ({pid_name.get(chip0)}) ==")

agg = defaultdict(float); cnt = defaultdict(int); total = 0.0
for e in ev:
    if e.get("ph") != "X" or "dur" not in e:
        continue
    if chip0 is not None and e["pid"] != chip0:
        continue
    nm = e.get("name", "?")
    agg[nm] += e["dur"]; cnt[nm] += 1; total += e["dur"]

def categorize(nm):
    n = nm.lower()
    if any(k in n for k in ["gmm", "megablox", "quantized_matmul", "_matmul", "dot"]):
        return "matmul/gmm"
    if any(k in n for k in ["attention", "ragged", "flash", "paged", "rpa"]):
        return "attention"
    if any(k in n for k in ["all-reduce","reduce-scatter","all-gather","collective","ppermute","all-to-all","psum","reduce_scatter","all_reduce","all_gather"]):
        return "collective"
    if any(k in n for k in ["sort","gather","scatter","top_k","topk","cumsum","argsort","one_hot","iota"]):
        return "moe-routing"
    if any(k in n for k in ["norm","rsqrt","rope","sin","cos","exp","logistic","sigmoid","silu"]):
        return "norm/act"
    if any(k in n for k in ["copy","transpose","bitcast","dynamic-slice","dynamic_slice","dynamic-update","convert","reshape","concatenate","pad"]):
        return "memory/layout"
    if any(k in n for k in ["mul","add","sub","div","fusion","select","compare","clamp","reduce"]):
        return "fusion/elementwise"
    return "other"

print(f"\n== total device-busy on chip0: {total/1000:.2f} ms; n_steps={n_steps} => {total/n_steps:.1f} us/step ==")
print("\n== TOP 40 ops by total device time ==")
print(f"{'op':<52}{'cat':<20}{'tot_ms':>9}{'cnt':>8}{'us/step':>9}")
for nm, d in sorted(agg.items(), key=lambda x: -x[1])[:40]:
    print(f"{nm[:51]:<52}{categorize(nm):<20}{d/1000:>9.2f}{cnt[nm]:>8}{d/n_steps:>9.1f}")

cat = defaultdict(float)
for nm, d in agg.items():
    cat[categorize(nm)] += d
print("\n== CATEGORY TOTALS ==")
print(f"{'category':<24}{'tot_ms':>10}{'us/step':>10}{'%':>8}")
for c, d in sorted(cat.items(), key=lambda x: -x[1]):
    print(f"{c:<24}{d/1000:>10.2f}{d/n_steps:>10.1f}{100*d/total:>8.1f}")
