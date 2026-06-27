#!/usr/bin/env python3
"""Worker-pool load generator for /v1/completions — maintains a fixed concurrency
across a larger TOTAL request pool so backfill/condense fires (bench-v3 scope).

Fixed input len 1 (prompt "Hi"), output len OUT (min_tokens=OUT, ignore_eos,
temperature=0). concurrency workers each pull from a queue of TOTAL requests.
Reports system OUTPUT tokens/sec (total_out / wall), mean/p50/p99 per-request
latency, mean TPOT, ttft, completion-token sanity. NO logprobs (spec-on crashes).

Usage:
  python bench_client_pool.py --port 8000 --concurrency 32 --total 96 --out 4096 --tag run
"""
import argparse, asyncio, json, time
import aiohttp

PROMPT = "Hi"

async def one_request(session, url, model, out, idx, results):
    payload = {
        "model": model, "prompt": PROMPT,
        "max_tokens": out, "min_tokens": out,
        "ignore_eos": True, "temperature": 0.0,
        "stream": True, "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    ttft = None
    n_chunks = 0
    completion_tokens = None
    prompt_tokens = None
    last_tok_time = t0
    inter_token_times = []
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                results.append({"idx": idx, "error": f"HTTP {resp.status}: {body[:300]}"})
                return
            async for raw in resp.content:
                line = raw.decode("utf-8", "replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    continue
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if choices and (choices[0].get("text")):
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - t0
                    else:
                        inter_token_times.append(now - last_tok_time)
                    last_tok_time = now
                    n_chunks += 1
                usage = obj.get("usage")
                if usage:
                    completion_tokens = usage.get("completion_tokens")
                    prompt_tokens = usage.get("prompt_tokens")
    except Exception as e:
        results.append({"idx": idx, "error": f"EXC: {type(e).__name__}: {e}"})
        return
    t1 = time.perf_counter()
    ct = completion_tokens if completion_tokens is not None else n_chunks
    tpot = (sum(inter_token_times) / len(inter_token_times)) if inter_token_times else None
    results.append({
        "idx": idx, "latency": t1 - t0, "ttft": ttft,
        "completion_tokens": ct, "prompt_tokens": prompt_tokens, "tpot": tpot,
    })

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--total", type=int, default=96)
    ap.add_argument("--out", type=int, default=4096)
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}/v1/completions"
    results = []
    timeout = aiohttp.ClientTimeout(total=2 * 60 * 60)  # 2h cap
    connector = aiohttp.TCPConnector(limit=0)
    q = asyncio.Queue()
    for i in range(args.total):
        q.put_nowait(i)

    async def worker(session):
        while True:
            try:
                idx = q.get_nowait()
            except asyncio.QueueEmpty:
                return
            await one_request(session, url, args.model, args.out, idx, results)
            q.task_done()

    wall0 = time.perf_counter()
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        workers = [asyncio.create_task(worker(session)) for _ in range(args.concurrency)]
        await asyncio.gather(*workers)
    wall = time.perf_counter() - wall0

    ok = [r for r in results if "error" not in r]
    err = [r for r in results if "error" in r]
    total_out = sum(r["completion_tokens"] for r in ok)
    lats = sorted(r["latency"] for r in ok)
    def pct(p):
        if not lats: return None
        k = int(round((p/100.0)*(len(lats)-1)))
        return lats[k]
    tpots = [r["tpot"] for r in ok if r["tpot"] is not None]
    ttfts = [r["ttft"] for r in ok if r["ttft"] is not None]

    print("=" * 60)
    print(f"TAG={args.tag}  concurrency={args.concurrency}  total={args.total}  out={args.out}")
    print(f"requests: ok={len(ok)} err={len(err)}")
    if err:
        for e in err[:8]:
            print(f"  ERR[{e['idx']}]: {e['error']}")
    print(f"wall_time_s: {wall:.2f}")
    print(f"total_output_tokens: {total_out}")
    print(f"SYSTEM_OUTPUT_TOK_PER_SEC: {total_out/wall:.2f}")
    if lats:
        print(f"per_req_latency_s mean={sum(lats)/len(lats):.2f} p50={pct(50):.2f} p99={pct(99):.2f} min={lats[0]:.2f} max={lats[-1]:.2f}")
    if ttfts:
        print(f"ttft_s mean={sum(ttfts)/len(ttfts):.3f} max={max(ttfts):.3f}")
    if tpots:
        m = sum(tpots)/len(tpots)
        print(f"TPOT_s mean={m:.5f}  (=> per-stream tok/s {1.0/m:.2f})")
    ctset = sorted(set(r["completion_tokens"] for r in ok))
    print(f"completion_tokens distinct values among ok: {ctset[:10]}")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
