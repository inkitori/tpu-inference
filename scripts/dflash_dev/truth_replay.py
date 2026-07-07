"""Replay the dumped serving trajectory through the HF draft, but with
GROUND-TRUTH aux features (HF gpt-oss CPU forward) instead of the dumped TPU
features. Same committed tokens, same draft, same positions — the only
variable is feature fidelity. Compares acceptance vs the TPU-feature replay.
"""
import glob
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

DUMP_DIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/dflash_dump"
B, S = 8, 7
AUX_HS_IDS = [2, 7, 12, 17, 22]

DRAFT_SNAP = glob.glob(
    "/tmp/gcs/bucket/hub/models--z-lab--gpt-oss-20b-DFlash/snapshots/*")[0]
TARGET_SNAP = glob.glob(
    "/tmp/gcs/bucket/hub/models--openai--gpt-oss-20b/snapshots/*")[0]

steps = [
    dict(np.load(p))
    for p in sorted(glob.glob(os.path.join(DUMP_DIR, "step_*.npz")))
]

# committed stream
tok_at = {}
for st in steps:
    a0 = int(st["out_qsl"][1] - st["out_qsl"][0]) - B
    for j in range(a0):
        tok_at[int(st["out_positions"][j])] = int(st["out_ids"][j])
    tok_at[int(st["out_positions"][a0])] = int(st["out_ids"][a0])
seq = [tok_at[p] for p in range(max(tok_at) + 1)]

from transformers import AutoModel, AutoModelForCausalLM, Mxfp4Config
from transformers.cache_utils import DynamicCache

print("loading target (dequantized bf16)...", flush=True)
target = AutoModelForCausalLM.from_pretrained(
    TARGET_SNAP,
    dtype=torch.bfloat16,
    quantization_config=Mxfp4Config(dequantize=True))
target.eval()

ids = torch.tensor([seq], dtype=torch.long)
with torch.no_grad():
    out = target(ids, output_hidden_states=True, use_cache=False)
truth = {
    li: out.hidden_states[hid][0]  # (T, 2880) bf16
    for li, hid in enumerate(AUX_HS_IDS)
}
embed_t = target.model.embed_tokens.weight.detach()
lm_head_t = target.lm_head.weight.detach()
del out, target

hf = AutoModel.from_pretrained(DRAFT_SNAP,
                               trust_remote_code=True,
                               dtype=torch.bfloat16,
                               attn_implementation="eager")
hf.eval()

cache = DynamicCache()
tot = {"tpu": 0, "truth": 0}
nsteps = 0
cache2 = DynamicCache()

for t, st in enumerate(steps):
    a0 = int(st["out_qsl"][1] - st["out_qsl"][0]) - B
    ctx_pos = st["out_positions"][:a0].astype(np.int64)
    noise_pos = st["out_positions"][a0:a0 + B].astype(np.int64)
    noise_ids = st["out_ids"][a0:a0 + B].astype(np.int64)
    all_pos = torch.from_numpy(np.concatenate([ctx_pos,
                                               noise_pos])).unsqueeze(0)
    noise_emb = embed_t[torch.from_numpy(noise_ids)].unsqueeze(0)
    p0 = int(noise_pos[0])

    drafts = {}
    for kind, kcache in (("tpu", cache), ("truth", cache2)):
        if kind == "tpu":
            raw = np.concatenate([st[f"aux{i}"][:a0] for i in range(5)],
                                 axis=-1)
            th = torch.from_numpy(raw).to(torch.bfloat16).unsqueeze(0)
        else:
            th = torch.cat([truth[li][ctx_pos] for li in range(5)],
                           dim=-1).unsqueeze(0)
        with torch.no_grad():
            o = hf(target_hidden=th,
                   noise_embedding=noise_emb,
                   position_ids=all_pos,
                   past_key_values=kcache,
                   use_cache=True,
                   is_causal=False)
            logits = o[:, -B + 1:, :] @ lm_head_t.T
            drafts[kind] = logits.argmax(dim=-1)[0].numpy()
        kcache.crop(int(st["out_seq_lens"][0]) - B)

    def match_len(d):
        n = 0
        for b in range(1, S + 1):
            tr = tok_at.get(p0 + b)
            if tr is None or int(d[b - 1]) != tr:
                break
            n += 1
        return n

    m_tpu, m_truth = match_len(drafts["tpu"]), match_len(drafts["truth"])
    tot["tpu"] += m_tpu
    tot["truth"] += m_truth
    nsteps += 1
    print(f"step {t:3d}: tpu-feat match={m_tpu} truth-feat match={m_truth}")

print(f"\nover {nsteps} steps: accepted/step with TPU features = "
      f"{tot['tpu']/nsteps:.2f}, with ground-truth features = "
      f"{tot['truth']/nsteps:.2f}")
