#!/usr/bin/env python3
"""A/B benchmark + correctness check for the MoE padding-row routing mask.

Run this on a TPU. It launches the SAME model twice in separate subprocesses --
once with the optimization OFF (MOE_MASK_PADDING_ROUTING=0) and once ON (=1) --
and reports, for each:

  * decode throughput across a SWEEP of batch sizes (tok/s), and
  * greedy generations + per-token logprobs for a fixed prompt set.

Then it diffs the two runs: per-batch-size throughput delta (the win at small
batches) AND a regression flag for any batch size the mask makes slower (large
batches have little/no padding, so they must stay ~1.0x), plus max logprob delta
/ token-id equality (proof the masking is numerically exact and outputs stay
coherent and identical).

This file is a standalone test harness -- it is NOT part of the library change,
so just `rm` it (or keep it out of the commit) before opening the PR.

Usage (on TPU):
    python bench_moe_padding_mask.py --model Qwen/Qwen3-30B-A3B --tp 8
    # add --no-ep to test the TP (non expert-parallel) path
    # override prompts/length with --max-tokens / --num-prompts
"""
import argparse
import json
import os
import subprocess
import sys
import time

PROMPTS = [
    "The capital of France is",
    "In one sentence, explain why the sky is blue:",
    "List three prime numbers greater than ten:",
    "def fibonacci(n):",
    "The most important property of a good benchmark is",
    "Translate 'good morning' into Spanish:",
    "A haiku about autumn leaves:",
    "Q: What is 17 multiplied by 4? A:",
]


# --------------------------------------------------------------------------- #
# Worker: runs inside a subprocess with the env var already set.
# --------------------------------------------------------------------------- #
def run_worker(args) -> None:
    from vllm import LLM, SamplingParams

    prompts = PROMPTS[:args.num_prompts]
    batch_sizes = [int(b) for b in args.batch_sizes.split(",") if b]
    max_num_seqs = args.max_num_seqs or max(batch_sizes)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        enable_expert_parallel=not args.no_ep,
        max_model_len=args.max_model_len,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=args.gpu_mem_util,
        trust_remote_code=True,
    )

    # ---- Correctness: greedy decode with logprobs (deterministic) ---------- #
    greedy = SamplingParams(temperature=0.0,
                            max_tokens=args.max_tokens,
                            logprobs=args.logprobs)
    # Warm up compilation so it doesn't pollute the result (not timed).
    llm.generate(prompts, greedy, use_tqdm=False)

    outs = llm.generate(prompts, greedy, use_tqdm=False)
    correctness = []
    for o in outs:
        comp = o.outputs[0]
        chosen_ids, chosen_lp, topk = [], [], []
        for tok_id, step_lp in zip(comp.token_ids, comp.logprobs or []):
            chosen_ids.append(int(tok_id))
            chosen_lp.append(float(step_lp[tok_id].logprob))
            topk.append({int(k): round(float(v.logprob), 5)
                        for k, v in step_lp.items()})
        correctness.append({
            "prompt": o.prompt,
            "text": comp.text,
            "token_ids": chosen_ids,
            "chosen_logprobs": chosen_lp,
            "topk_logprobs": topk,
        })

    # ---- Throughput: batch=1 and batch=8 decode tok/s ---------------------- #
    def measure(batch_size: int) -> dict:
        sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
        batch = (prompts * ((batch_size // len(prompts)) + 1))[:batch_size]
        llm.generate(batch, sp, use_tqdm=False)  # warmup for this shape
        t0 = time.perf_counter()
        res = llm.generate(batch, sp, use_tqdm=False)
        elapsed = time.perf_counter() - t0
        out_toks = sum(len(r.outputs[0].token_ids) for r in res)
        return {
            "batch_size": batch_size,
            "elapsed_s": round(elapsed, 4),
            "output_tokens": out_toks,
            "tok_per_s": round(out_toks / elapsed, 2),
            "tok_per_s_per_user": round(out_toks / elapsed / batch_size, 2),
        }

    throughput = [measure(b) for b in batch_sizes]

    with open(args.out, "w") as f:
        json.dump({"mode": args.mode,
                  "correctness": correctness,
                  "throughput": throughput}, f)
    print(f"[worker mode={args.mode}] wrote {args.out}")


# --------------------------------------------------------------------------- #
# Driver: spawns one worker per mode, then compares.
# --------------------------------------------------------------------------- #
def run_driver(args) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    results = {}
    for mode in ("off", "on"):
        out = os.path.join(args.out_dir, f"result_{mode}.json")
        env = dict(os.environ)
        env["MOE_MASK_PADDING_ROUTING"] = "1" if mode == "on" else "0"
        cmd = [sys.executable, __file__, "--worker", "--mode", mode,
               "--out", out, "--model", args.model, "--tp", str(args.tp),
               "--max-tokens", str(args.max_tokens),
               "--num-prompts", str(args.num_prompts),
               "--logprobs", str(args.logprobs),
               "--max-model-len", str(args.max_model_len),
               "--gpu-mem-util", str(args.gpu_mem_util),
               "--batch-sizes", args.batch_sizes,
               "--max-num-seqs", str(args.max_num_seqs)]
        if args.no_ep:
            cmd.append("--no-ep")
        print(f"\n===== launching worker: mode={mode} "
              f"(MOE_MASK_PADDING_ROUTING={env['MOE_MASK_PADDING_ROUTING']}) =====")
        subprocess.run(cmd, env=env, check=True)
        with open(out) as f:
            results[mode] = json.load(f)

    _report(results["off"], results["on"])


def _report(off: dict, on: dict, regress_threshold: float = 0.97) -> None:
    print("\n" + "=" * 72)
    print("THROUGHPUT SWEEP  (off = mask disabled, on = mask enabled)")
    print("=" * 72)
    print(f"{'batch':>6} | {'off tok/s':>10} | {'on tok/s':>10} | "
          f"{'speedup':>8} | flag")
    regressions = []
    for a, b in zip(off["throughput"], on["throughput"]):
        sp = b["tok_per_s"] / a["tok_per_s"] if a["tok_per_s"] else float("nan")
        # A regression = the optimization made this batch size meaningfully
        # SLOWER. Larger batches have little/no padding, so we expect ~1.0x
        # there; anything below the threshold is a real regression to inspect.
        flag = ""
        if sp < regress_threshold:
            flag = "<< REGRESSION"
            regressions.append((a["batch_size"], sp))
        elif sp >= 1.05:
            flag = "win"
        print(f"{a['batch_size']:>6} | {a['tok_per_s']:>10} | "
              f"{b['tok_per_s']:>10} | {sp:>7.2f}x | {flag}")
    print("-" * 72)
    if regressions:
        print(f"REGRESSION DETECTED at batch sizes "
              f"{[bs for bs, _ in regressions]} "
              f"(speedup < {regress_threshold:.2f}x) -- the mask is slowing "
              f"these down; investigate before merging.")
    else:
        print(f"No regressions: every batch size is >= {regress_threshold:.2f}x "
              f"(optimization never slower than baseline).")

    print("\n" + "=" * 64)
    print("OUTPUT IDENTITY  (masking must be numerically exact)")
    print("=" * 64)
    max_lp_diff = 0.0
    all_ids_equal = True
    for i, (a, b) in enumerate(zip(off["correctness"], on["correctness"])):
        ids_equal = a["token_ids"] == b["token_ids"]
        all_ids_equal &= ids_equal
        n = min(len(a["chosen_logprobs"]), len(b["chosen_logprobs"]))
        diff = max((abs(a["chosen_logprobs"][j] - b["chosen_logprobs"][j])
                    for j in range(n)), default=0.0)
        max_lp_diff = max(max_lp_diff, diff)
        flag = "OK " if ids_equal else "MISMATCH"
        print(f"[{flag}] prompt {i}: token_ids equal={ids_equal}, "
              f"max|Δlogprob|={diff:.2e}")
        if not ids_equal:
            print(f"        off: {a['text'][:80]!r}")
            print(f"        on : {b['text'][:80]!r}")

    print("-" * 64)
    print(f"all token-id sequences identical : {all_ids_equal}")
    print(f"global max |Δ chosen-token logprob|: {max_lp_diff:.3e}")
    verdict = ("PASS: outputs identical + exact"
               if all_ids_equal and max_lp_diff < 1e-3
               else "CHECK: divergence detected -- inspect above")
    print(f"verdict: {verdict}")
    print("\nSample generations (mode=on):")
    for c in on["correctness"][:3]:
        print(f"  {c['prompt']!r} -> {c['text'][:100]!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    p.add_argument("--tp", type=int, default=8)
    p.add_argument("--no-ep", action="store_true",
                  help="disable expert parallelism (test GMM_TP path)")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--batch-sizes", default="1,8,16,32,64,128,256",
                  help="comma-separated decode batch sizes to sweep")
    p.add_argument("--max-num-seqs", type=int, default=0,
                  help="vLLM max_num_seqs; 0 = max(batch_sizes)")
    p.add_argument("--num-prompts", type=int, default=len(PROMPTS))
    p.add_argument("--logprobs", type=int, default=5)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--gpu-mem-util", type=float, default=0.88)
    p.add_argument("--out-dir", default="/tmp/moe_mask_bench")
    # worker-only flags
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--mode", choices=["off", "on"], help=argparse.SUPPRESS)
    p.add_argument("--out", help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.worker:
        run_worker(args)
    else:
        run_driver(args)


if __name__ == "__main__":
    main()
