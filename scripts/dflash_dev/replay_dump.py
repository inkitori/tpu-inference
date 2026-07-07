"""Offline replay of SPEC_DFLASH_DUMP serving dumps through the HF reference.

Bisects the acceptance bug:
  - HF-replay drafts (from the DUMPED serving aux features) accept well
    -> aux capture is fine; our propose path/assembly diverges (compare
       replay vs served drafts and the dumped assembled stream).
  - HF-replay drafts also accept ~20% -> the captured aux features (or the
    shared embed/lm_head) are wrong.

Run after a serving session with:
  SPEC_DFLASH_DUMP=/tmp/dflash_dump bash scripts/serve_dflash.sh --no-async-scheduling
  (single request, short prompt, greedy)
then:
  HF_MODULES_CACHE=/tmp/hfmods python scripts/dflash_dev/replay_dump.py /tmp/dflash_dump
"""
import glob
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

DUMP_DIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/dflash_dump"
B = 8  # draft block size
S = 7  # num speculative tokens

DRAFT_SNAP = glob.glob(
    "/tmp/gcs/bucket/hub/models--z-lab--gpt-oss-20b-DFlash/snapshots/*")[0]
TARGET_SNAP = glob.glob(
    "/tmp/gcs/bucket/hub/models--openai--gpt-oss-20b/snapshots/*")[0]

steps = []
for path in sorted(glob.glob(os.path.join(DUMP_DIR, "step_*.npz"))):
    steps.append(dict(np.load(path)))
print(f"loaded {len(steps)} dumped steps from {DUMP_DIR}")
assert steps, "no dumps found"

# ---------------------------------------------------------------- weights
from safetensors import safe_open

target_files = glob.glob(os.path.join(TARGET_SNAP, "*.safetensors"))
embed_t = lm_head_t = None
for f in target_files:
    with safe_open(f, framework="pt") as sf:
        for k in sf.keys():
            if k == "model.embed_tokens.weight":
                embed_t = sf.get_tensor(k)
            elif k == "lm_head.weight":
                lm_head_t = sf.get_tensor(k)
assert embed_t is not None and lm_head_t is not None, "target embed/lm_head not found"
print(f"target embed {tuple(embed_t.shape)} lm_head {tuple(lm_head_t.shape)}")

# --- check the served (shared) embed/lm_head against the checkpoint ---
s0 = steps[0]
for name, ckpt in (("embed", embed_t), ("lm_head", lm_head_t)):
    served_slice = s0[f"{name}_slice"]  # (512, D) fp32
    ckpt_slice = ckpt[:512].float().numpy()
    diff = np.abs(served_slice - ckpt_slice).max()
    served_norms = s0[f"{name}_row_norms"][:ckpt.shape[0]]
    ckpt_norms = torch.linalg.norm(ckpt.float(), dim=1).numpy()
    ndiff = np.abs(served_norms - ckpt_norms).max()
    print(f"shared {name}: slice max|diff|={diff:.6f} "
          f"row-norm max|diff|={ndiff:.6f}")

# ---------------------------------------------------------------- HF draft
from transformers import AutoModel

hf = AutoModel.from_pretrained(DRAFT_SNAP,
                               trust_remote_code=True,
                               dtype=torch.bfloat16,
                               attn_implementation="eager")
hf.eval()
embed_t = embed_t.to(torch.bfloat16)
lm_head_t = lm_head_t.to(torch.bfloat16)

from transformers.cache_utils import DynamicCache

# ------------------------------------------------- committed-stream oracle
committed = {}  # position -> token id (greedy target continuation)
for st in steps:
    a0 = int(st["out_qsl"][1] - st["out_qsl"][0]) - B
    for j in range(a0):  # accepted ctx rows
        committed[int(st["out_positions"][j])] = int(st["out_ids"][j])
    # noise[0] = the token sampled this step (bonus/correction) — committed.
    committed[int(st["out_positions"][a0])] = int(st["out_ids"][a0])

# ---------------------------------------------------------------- replay
cache = DynamicCache()
tot_served = tot_replay = tot_steps = 0
agree_all = 0

for t, st in enumerate(steps):
    a0 = int(st["out_qsl"][1] - st["out_qsl"][0]) - B
    ctx_pos = st["out_positions"][:a0]
    noise_pos = st["out_positions"][a0:a0 + B]
    noise_ids = st["out_ids"][a0:a0 + B]
    cache_len = cache.get_seq_length()

    # positions must be contiguous [cache_len, start+B)
    all_pos = np.concatenate([ctx_pos, noise_pos])
    expect = np.arange(cache_len, int(noise_pos[-1]) + 1)
    if not np.array_equal(all_pos, expect):
        print(f"step {t}: POSITION MISMATCH cache_len={cache_len} "
              f"ctx={ctx_pos[:5]}..{ctx_pos[-3:] if a0 else '[]'} "
              f"noise={noise_pos}")

    raw_ctx = np.concatenate([st[f"aux{i}"][:a0] for i in range(5)], axis=-1)
    target_hidden = torch.from_numpy(raw_ctx).to(torch.bfloat16).unsqueeze(0)

    # cross-check our combined ctx rows vs HF fc+hidden_norm on the raw aux
    with torch.no_grad():
        hf_combined = hf.hidden_norm(hf.fc(target_hidden))[0].float().numpy()
    ours_combined = st["out_combined"][:a0]
    if a0 > 0:
        cd = np.abs(hf_combined - ours_combined)
        denom = np.linalg.norm(hf_combined) * np.linalg.norm(ours_combined)
        ccos = float(np.sum(hf_combined * ours_combined) / denom)
    else:
        cd, ccos = np.zeros(1), 1.0

    noise_emb = embed_t[torch.from_numpy(noise_ids.astype(np.int64))]
    position_ids = torch.from_numpy(all_pos.astype(np.int64)).unsqueeze(0)
    with torch.no_grad():
        out = hf(
            target_hidden=target_hidden,
            noise_embedding=noise_emb.unsqueeze(0),
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            is_causal=False,
        )
        logits = out[:, -B + 1:, :] @ lm_head_t.T
        replay_draft = logits.argmax(dim=-1)[0].numpy()
    cache.crop(int(st["out_seq_lens"][0]) - B)

    served_draft = st["draft_token_ids"].reshape(-1)[:S]

    p0 = int(noise_pos[0])

    def match_len(draft):
        n = 0
        for b in range(1, S + 1):
            truth = committed.get(p0 + b)
            if truth is None or int(draft[b - 1]) != truth:
                break
            n += 1
        return n

    m_served, m_replay = match_len(served_draft), match_len(replay_draft)
    agree = int(np.sum(served_draft == replay_draft))
    tot_served += m_served
    tot_replay += m_replay
    agree_all += agree
    tot_steps += 1
    print(f"step {t:3d}: a={a0:3d} pos0={p0:4d} served_match={m_served} "
          f"replay_match={m_replay} served==replay {agree}/7 "
          f"combined cos={ccos:.5f} max|d|={cd.max():.4f}")

print(f"\nTOTALS over {tot_steps} steps: "
      f"served accepted/step={tot_served/tot_steps:.2f}  "
      f"HF-replay accepted/step={tot_replay/tot_steps:.2f}  "
      f"served-vs-replay agreement={agree_all/(7*tot_steps):.1%}")
