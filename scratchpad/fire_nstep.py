import sys, json, threading, urllib.request
URL="http://127.0.0.1:8000/v1/completions"; MODEL="openai/gpt-oss-20b"
PROMPTS=["The capital of France is","List the first five prime numbers:",
 "The chemical symbol for gold is","The largest planet in our solar system is",
 "The square root of 144 is","Name three primary colors:",
 "Water is composed of hydrogen and","The first president of the United States was"]
def fire(idx,prompt,mt,res):
    body=json.dumps({"model":MODEL,"prompt":prompt,"max_tokens":mt,"temperature":0.0,"seed":0}).encode()
    req=urllib.request.Request(URL,data=body,headers={"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req,timeout=180) as r: d=json.loads(r.read())
        res[idx]={"ok":True,"prompt":prompt,"text":d["choices"][0]["text"]}
    except Exception as e: res[idx]={"ok":False,"prompt":prompt,"err":repr(e)}
n=int(sys.argv[1]); mt=int(sys.argv[2]); out=sys.argv[3]
prompts=[PROMPTS[i%len(PROMPTS)] for i in range(n)]; res=[None]*n
th=[threading.Thread(target=fire,args=(i,p,mt,res)) for i,p in enumerate(prompts)]
[t.start() for t in th]; [t.join() for t in th]
json.dump(res,open(out,"w")); ok=sum(1 for r in res if r and r["ok"]); print(f"wrote {out}: {ok}/{n} OK")
