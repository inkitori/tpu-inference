#!/usr/bin/env python3
"""Drive steady-state decode to exercise spec-decode accept metrics.
Fixed concurrency over a request pool. Real-ish prompts, greedy, NO logprobs.
Usage: python fire_accept_load.py --concurrency 24 --total 48 --out 300
"""
import argparse, asyncio, json, time
import aiohttp

PROMPTS = [
    "Write a detailed paragraph about the history of computing, from the abacus to modern supercomputers.",
    "Explain how a transformer neural network processes a sequence of tokens, step by step.",
    "Describe the water cycle in detail, covering evaporation, condensation, precipitation, and collection.",
    "Write an essay about the causes and consequences of the Industrial Revolution in Europe.",
    "Explain the theory of general relativity and how it differs from Newtonian gravity.",
    "Describe how the human immune system defends the body against viral and bacterial infections.",
    "Write a story about an explorer who discovers an ancient city hidden deep in the rainforest.",
    "Explain how modern CPUs use pipelining, caching, and branch prediction to run faster.",
]

async def one(session, url, model, out, idx, results):
    payload = {
        "model": model, "prompt": PROMPTS[idx % len(PROMPTS)],
        "max_tokens": out, "min_tokens": out,
        "ignore_eos": True, "temperature": 0.0,
        "stream": True, "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter(); ct = None
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                results.append({"idx": idx, "error": f"HTTP {resp.status}: {body[:200]}"}); return
            async for raw in resp.content:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"): continue
                data = line[5:].strip()
                if data == "[DONE]": continue
                try: obj = json.loads(data)
                except json.JSONDecodeError: continue
                u = obj.get("usage")
                if u: ct = u.get("completion_tokens")
    except Exception as e:
        results.append({"idx": idx, "error": f"EXC {type(e).__name__}: {e}"}); return
    results.append({"idx": idx, "latency": time.perf_counter()-t0, "ct": ct})

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--total", type=int, default=48)
    ap.add_argument("--out", type=int, default=300)
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    args = ap.parse_args()
    url = f"http://127.0.0.1:{args.port}/v1/completions"
    results = []
    q = asyncio.Queue()
    for i in range(args.total): q.put_nowait(i)
    async def worker(s):
        while True:
            try: idx = q.get_nowait()
            except asyncio.QueueEmpty: return
            await one(s, url, args.model, args.out, idx, results); q.task_done()
    to = aiohttp.ClientTimeout(total=3600)
    conn = aiohttp.TCPConnector(limit=0)
    w0 = time.perf_counter()
    async with aiohttp.ClientSession(timeout=to, connector=conn) as s:
        await asyncio.gather(*[worker(s) for _ in range(args.concurrency)])
    wall = time.perf_counter()-w0
    ok = [r for r in results if "error" not in r]
    err = [r for r in results if "error" in r]
    tot = sum(r["ct"] or 0 for r in ok)
    print(f"ok={len(ok)} err={len(err)} wall={wall:.1f}s out_tok={tot} sys_tok/s={tot/wall:.1f}")
    for e in err[:6]: print("  ERR", e["idx"], e["error"])

if __name__ == "__main__":
    asyncio.run(main())
