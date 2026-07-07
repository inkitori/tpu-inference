"""Run z-lab's spec_generate loop (pure HF reference, CPU) on the same prompt
the serving dump used, and report per-step acceptance — the checkpoint's true
capability on this exact text, independent of our TPU stack.

The decode loop below is a line-for-line port of
DFlashDraftModel.spec_generate from the checkpoint's dflash.py, with
acceptance_lengths recorded.
"""
import glob
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

DUMP_DIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/dflash_dump"
MAX_NEW = int(sys.argv[2]) if len(sys.argv) > 2 else 120

DRAFT_SNAP = glob.glob(
    "/tmp/gcs/bucket/hub/models--z-lab--gpt-oss-20b-DFlash/snapshots/*")[0]
TARGET_SNAP = glob.glob(
    "/tmp/gcs/bucket/hub/models--openai--gpt-oss-20b/snapshots/*")[0]

st = dict(np.load(os.path.join(DUMP_DIR, "step_0000.npz")))
a0 = int(st["out_qsl"][1] - st["out_qsl"][0]) - 8
prompt_ids = st["out_ids"][:a0].astype(np.int64)
print(f"prompt: {len(prompt_ids)} tokens {prompt_ids[:8]}...")

from transformers import AutoModel, AutoModelForCausalLM, Mxfp4Config
from transformers.cache_utils import DynamicCache

print("loading target (CPU, dequantized bf16)...", flush=True)
target = AutoModelForCausalLM.from_pretrained(
    TARGET_SNAP,
    dtype=torch.bfloat16,
    quantization_config=Mxfp4Config(dequantize=True))
target.eval()
print("loading draft...", flush=True)
draft = AutoModel.from_pretrained(DRAFT_SNAP,
                                  trust_remote_code=True,
                                  dtype=torch.bfloat16,
                                  attn_implementation="eager")
draft.eval()

hfmod = sys.modules[type(draft).__module__]
extract_context_feature = hfmod.extract_context_feature  # from utils.py
sample = hfmod.sample

input_ids = torch.tensor([prompt_ids])
block_size = draft.block_size
mask_token_id = draft.mask_token_id
target_layer_ids = draft.target_layer_ids
num_input_tokens = input_ids.shape[1]
max_length = num_input_tokens + MAX_NEW

with torch.inference_mode():
    output_ids = torch.full((1, max_length + block_size),
                            mask_token_id,
                            dtype=torch.long)
    position_ids = torch.arange(output_ids.shape[1]).unsqueeze(0)
    past_target = DynamicCache()
    past_draft = DynamicCache()

    out = target(input_ids,
                 position_ids=position_ids[:, :num_input_tokens],
                 past_key_values=past_target,
                 use_cache=True,
                 logits_to_keep=1,
                 output_hidden_states=True)
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens:num_input_tokens + 1] = sample(
        out.logits, 0.0)
    target_hidden = extract_context_feature(out.hidden_states,
                                            target_layer_ids)

    acceptance_lengths = []
    start = num_input_tokens
    while start < max_length:
        block_output_ids = output_ids[:, start:start + block_size].clone()
        block_position_ids = position_ids[:, start:start + block_size]
        noise_embedding = target.model.embed_tokens(block_output_ids)
        draft_logits = target.lm_head(
            draft(target_hidden=target_hidden,
                  noise_embedding=noise_embedding,
                  position_ids=position_ids[:,
                                            past_draft.get_seq_length():start +
                                            block_size],
                  past_key_values=past_draft,
                  use_cache=True,
                  is_causal=False)[:, -block_size + 1:, :])
        past_draft.crop(start)
        block_output_ids[:, 1:] = sample(draft_logits)

        out = target(block_output_ids,
                     position_ids=block_position_ids,
                     past_key_values=past_target,
                     use_cache=True,
                     output_hidden_states=True)
        posterior = sample(out.logits, 0.0)
        acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]
                             ).cumprod(dim=1).sum(dim=1)[0].item()
        output_ids[:, start:start + acceptance_length +
                   1] = block_output_ids[:, :acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:,
                                                                 acceptance_length]
        start += acceptance_length + 1
        past_target.crop(start)
        target_hidden = extract_context_feature(
            out.hidden_states, target_layer_ids)[:, :acceptance_length + 1, :]
        acceptance_lengths.append(acceptance_length + 1)
        print(f"step {len(acceptance_lengths):3d}: accepted+bonus="
              f"{acceptance_length + 1}", flush=True)

gen = output_ids[0, num_input_tokens:max_length]
gen = gen[gen != mask_token_id]
print("generated ids:", gen.tolist())
al = np.array(acceptance_lengths)
print(f"\nREFERENCE acceptance over {len(al)} steps: "
      f"mean accept len (incl bonus) = {al.mean():.2f}; "
      f"accepted draft tokens/step = {al.mean() - 1:.2f}")
