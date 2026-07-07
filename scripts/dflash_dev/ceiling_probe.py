"""Acceptance CEILING on ShareGPT bench traffic: run z-lab's spec_generate
loop (pure HF reference, CPU — HF draft vs HF target, both exact) on the same
prompts `vllm bench serve --dataset-name sharegpt` uses, with EOS ignored
(generation forced to MAX_NEW like the bench's --ignore-eos).

This bounds what serving acceptance could reach if the serving verifier
matched HF argmax exactly. Compare against the served bench accept len.

Usage:
  HF_MODULES_CACHE=/tmp/hfmods OMP_NUM_THREADS=64 \
    python scripts/dflash_dev/ceiling_probe.py <sharegpt.json> [n_prompts] [max_new]
"""
import glob
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

SHAREGPT = sys.argv[1]
N_PROMPTS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
MAX_NEW = int(sys.argv[3]) if len(sys.argv) > 3 else 150

DRAFT_SNAP = glob.glob(
    "/tmp/gcs/bucket/hub/models--z-lab--gpt-oss-20b-DFlash/snapshots/*")[0]
TARGET_SNAP = glob.glob(
    "/tmp/gcs/bucket/hub/models--openai--gpt-oss-20b/snapshots/*")[0]

from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          Mxfp4Config)
from transformers.cache_utils import DynamicCache

tok = AutoTokenizer.from_pretrained(TARGET_SNAP)

# Mirror vllm bench's ShareGPT sampling closely enough: first human turn as
# the raw (non-chat-templated) prompt, filtered to 4 < prompt_len and
# 4 < output_len, prompt_len + output_len <= 2048 (server max_model_len).
with open(SHAREGPT) as f:
    data = json.load(f)
prompts = []
for entry in data:
    conv = entry.get("conversations") or []
    if len(conv) < 2:
        continue
    prompt_text = conv[0]["value"]
    completion_text = conv[1]["value"]
    p_ids = tok(prompt_text).input_ids
    c_ids = tok(completion_text).input_ids
    if len(p_ids) < 4 or len(c_ids) < 4:
        continue
    if len(p_ids) > 1024 or len(p_ids) + len(c_ids) > 2048:
        continue
    prompts.append((prompt_text, len(p_ids)))
    if len(prompts) >= N_PROMPTS:
        break
print(f"selected {len(prompts)} prompts, lens: {[l for _, l in prompts]}",
      flush=True)

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
extract_context_feature = hfmod.extract_context_feature
sample = hfmod.sample

block_size = draft.block_size
mask_token_id = draft.mask_token_id
target_layer_ids = draft.target_layer_ids

all_accept = []
for pi, (prompt_text, plen) in enumerate(prompts):
    input_ids = torch.tensor([tok(prompt_text).input_ids])
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
            block_output_ids = output_ids[:, start:start +
                                          block_size].clone()
            block_position_ids = position_ids[:, start:start + block_size]
            noise_embedding = target.model.embed_tokens(block_output_ids)
            draft_logits = target.lm_head(
                draft(target_hidden=target_hidden,
                      noise_embedding=noise_embedding,
                      position_ids=position_ids[:,
                                                past_draft.get_seq_length():
                                                start + block_size],
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
            output_ids[:, start + acceptance_length +
                       1] = posterior[:, acceptance_length]
            start += acceptance_length + 1
            past_target.crop(start)
            target_hidden = extract_context_feature(
                out.hidden_states,
                target_layer_ids)[:, :acceptance_length + 1, :]
            acceptance_lengths.append(acceptance_length + 1)

    al = np.array(acceptance_lengths)
    all_accept.extend(acceptance_lengths)
    print(
        f"prompt {pi} (len {plen}): steps={len(al)} "
        f"accept len (incl bonus)={al.mean():.2f}",
        flush=True)

al = np.array(all_accept)
print(f"\nCEILING over {len(prompts)} ShareGPT prompts, {len(al)} steps: "
      f"mean accept len (incl bonus) = {al.mean():.3f} "
      f"(served bench measured ~2.22)")
