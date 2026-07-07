"""Compare serving-dumped aux hidden states against HF gpt-oss ground truth.

Rebuilds the committed token stream from SPEC_DFLASH_DUMP dumps, teacher-forces
it through HF gpt-oss-20b on CPU (mxfp4 dequantized to bf16) with
output_hidden_states=True, and compares hidden_states[2,7,12,17,22] (outputs
of layers 1,6,11,16,21) against the dumped aux features at every position the
serving path captured.
"""
import glob
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

DUMP_DIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/dflash_dump"
B = 8
AUX_HS_IDS = [2, 7, 12, 17, 22]  # hidden_states indices = outputs of 1,6,11,16,21

TARGET_SNAP = glob.glob(
    "/tmp/gcs/bucket/hub/models--openai--gpt-oss-20b/snapshots/*")[0]

steps = [
    dict(np.load(p))
    for p in sorted(glob.glob(os.path.join(DUMP_DIR, "step_*.npz")))
]
print(f"loaded {len(steps)} steps")

# committed token at each position, and dumped aux rows per position
tok_at = {}
aux_at = {}  # pos -> list of 5 arrays (2880,)
for st in steps:
    a0 = int(st["out_qsl"][1] - st["out_qsl"][0]) - B
    in_q0 = int(st["in_qsl"][1] - st["in_qsl"][0])
    for j in range(a0):
        pos = int(st["out_positions"][j])
        tok_at[pos] = int(st["out_ids"][j])
        # aux row j of the input stream (src = in_qsl[0] + j = j for req 0)
        aux_at[pos] = [st[f"aux{i}"][j] for i in range(5)]
    tok_at[int(st["out_positions"][a0])] = int(st["out_ids"][a0])

max_pos = max(aux_at)
seq = [tok_at[p] for p in range(max_pos + 1)]
print(f"committed stream length {len(seq)}; aux captured at "
      f"{len(aux_at)} positions up to {max_pos}")

from transformers import AutoModelForCausalLM, Mxfp4Config

print("loading HF gpt-oss-20b (CPU, dequantized)...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    TARGET_SNAP,
    dtype=torch.bfloat16,
    quantization_config=Mxfp4Config(dequantize=True),
)
model.eval()

ids = torch.tensor([seq], dtype=torch.long)
with torch.no_grad():
    out = model(ids, output_hidden_states=True, use_cache=False)
hs = out.hidden_states  # tuple: [emb, out_l0, ..., out_l23]
print("hf forward done; hidden_states:", len(hs), hs[0].shape, flush=True)

for li, hid in enumerate(AUX_HS_IDS):
    ref = hs[hid][0].float().numpy()  # (T, 2880)
    cos_list, dif_list = [], []
    for pos, aux in sorted(aux_at.items()):
        a = aux[li]
        r = ref[pos]
        cos_list.append(
            float(np.dot(a, r) / (np.linalg.norm(a) * np.linalg.norm(r))))
        dif_list.append(float(np.abs(a - r).mean() / np.abs(r).mean()))
    cos_arr = np.array(cos_list)
    print(f"aux{li} (hf hidden_states[{hid}]): cos mean={cos_arr.mean():.5f} "
          f"min={cos_arr.min():.5f} rel_meanabs={np.mean(dif_list):.4f}")
    worst = np.argsort(cos_arr)[:5]
    poss = sorted(aux_at.keys())
    print(f"   worst positions: {[(poss[w], round(cos_arr[w],4)) for w in worst]}")

# also greedy-match check: does the HF target argmax reproduce the committed
# stream? (validates that the serving target == HF target functionally)
logits = out.logits[0].float()
pred = logits.argmax(dim=-1).numpy()
match = sum(
    1 for p in range(len(seq) - 1) if pred[p] == seq[p + 1]) / (len(seq) - 1)
print(f"HF-target greedy match with served continuation: {match:.1%}")
