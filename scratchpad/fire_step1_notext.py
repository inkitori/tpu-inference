#!/usr/bin/env python3
# max_tokens=1, NO logprobs (avoids spec-on logprobs serialization bug).
# Captures just the single next-token text per prompt for matched-shape compare.
import sys, json, threading, urllib.request
URL="http://127.0.0.1:8000/v1/completions"; MODEL="openai/gpt-oss-20b"
PROMPTS=["The capital of France is","List the first five prime numbers:",
 "The chemical symbol for gold is","The largest planet in our solar system is",
 "The square root of 144 is","Name three primary colors:",
 "Water is composed of hydrogen and","The first president of the United States was"]
def fire(idx,prompt,res):
    body=json.dumps({"model":MODEL,"prompt":prompt,"max_tokens":1,"temperature":0.0,"seed":0}).encode()
    req=urllib.request.Request(URL,data=body,headers={"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req,timeout=120) as r: d=json.loads(r.read())
        res[idx]={"ok":True,"prompt":prompt,"next":d["choices"][0]["text"]}
    except Exception as e: res[idx]={"ok":False,"prompt":prompt,"err":repr(e)}
def main():
    n=int(sys.argv[1]); out=sys.argv[2]
    prompts=[PROMPTS[i%len(PROMPTS)] for i in range(n)]; res=[None]*n
    th=[threading.Thread(target=fire,args=(i,p,res)) for i,p in enumerate(prompts)]
    [t.start() for t in th]; [t.join() for t in th]
    json.dump(res,open(out,"w"))
    ok=sum(1 for r in res if r and r["ok"]); print(f"wrote {out}: {ok}/{n} OK")
    for i,r in enumerate(res):
        print(f"[{i}] {'OK ' if r['ok'] else 'ERR'} next={r.get('next','')!r} {r['prompt'][:30]!r}")
main()
