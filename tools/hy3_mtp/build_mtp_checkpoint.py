"""Build a local Hy3-preview-4bit checkpoint variant that adds the MTP layer.

The MLX 4-bit conversion dropped model.layers.80.* (the num_nextn_predict_layers=1
MTP block). This script recovers those weights from the base checkpoint's shards
111/112 (already downloaded), quantizes them into the exact MLX int4 affine
format mimicking layer 79's tensor patterns, and assembles a checkpoint dir:

  ~/hy3_mtp/Hy3-preview-4bit-mtp/
    model-0000{1..34}-of-00034.safetensors  -> symlinks to the MLX snapshot
    model-mtp.safetensors                   <- new layer-80 tensors
    model.safetensors.index.json            <- merged index
    config.json                             <- + layers.80 router-gate 8-bit override
    tokenizer/etc                           -> symlinks

Layer-80 tensor plan (mimic MLX layer 79 exactly):
  switch_mlp.{gate,up,down}_proj.{weight,scales,biases}  stacked [192,out,*] 4-bit
  shared_mlp.{...}                                       4-bit triplets
  self_attn.{q,k,v,o}_proj.{...}                         4-bit triplets
  mlp.router.gate.{...}                                  8-bit triplet (+cfg override)
  mlp.router.expert_bias                                 f32 (renamed from mlp.expert_bias)
  norms (input/post_attention/final_layernorm, q/k_norm)  bf16 passthrough
MTP extras:
  eh_proj.weight  bf16 (plain nn.Linear in HYV3MTP - must NOT be quantized)
  enorm.weight, hnorm.weight  bf16
"""
import json
import os
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import glob

# Base-model shards holding layer 80 (download first, ~7.9GB):
#   mkdir -p ~/hy3_mtp/shards && cd ~/hy3_mtp/shards
#   for s in 00111 00112; do curl -L -O \
#     "https://huggingface.co/tencent/Hy3-preview/resolve/main/model-$s-of-00112.safetensors"; done
SHARD_DIR = os.path.expanduser("~/hy3_mtp/shards")
_MOUNT = os.environ.get("TPU_GCS_MOUNT", "/tmp/gcs/bucket")
SNAP = sorted(glob.glob(
    f"{_MOUNT}/hub/models--mlx-community--Hy3-preview-4bit/snapshots/*"))[0]
OUT_DIR = os.path.expanduser("~/hy3_mtp/Hy3-preview-4bit-mtp")
GS = 64
L = "model.layers.80."

# ---------------------------------------------------------------- quant utils

def mlx_quantize(w: np.ndarray, bits: int, gs: int = GS):
    """Affine per-group quant along the input dim (axis -1), MLX layout.

    w: [out, in] float32. Returns (packed uint32 [out, in*bits/32],
    scales bf16-able f32 [out, in/gs], biases f32 [out, in/gs]).
    Reconstruction: w ~= scales*q + biases with q unsigned in [0, 2^bits-1].
    """
    out, inn = w.shape
    assert inn % gs == 0
    qmax = (1 << bits) - 1
    g = w.reshape(out, inn // gs, gs)
    wmin = g.min(axis=-1)
    wmax = g.max(axis=-1)
    scale = (wmax - wmin) / qmax
    # Constant group: scale 0 -> encode exactly via bias, scale 1.
    zero = scale <= 1e-12
    scale = np.where(zero, 1.0, scale)
    bias = wmin
    # bf16-round scale/bias FIRST, then quantize against the rounded values so
    # the on-disk affine params reproduce our q optimally.
    scale = torch.from_numpy(scale.astype(np.float32)).to(torch.bfloat16).to(torch.float32).numpy()
    bias = torch.from_numpy(bias.astype(np.float32)).to(torch.bfloat16).to(torch.float32).numpy()
    scale = np.where(scale == 0.0, 1.0, scale)  # bf16 rounding could re-zero
    q = np.rint((g - bias[..., None]) / scale[..., None])
    q = np.clip(q, 0, qmax).astype(np.uint32).reshape(out, inn)
    # Pack little-endian within each uint32 word: element k of a word occupies
    # bits [bits*k, bits*(k+1)) -- must match tpu_inference mlx_unpack.
    per_word = 32 // bits
    qw = q.reshape(out, inn // per_word, per_word)
    shifts = (np.arange(per_word, dtype=np.uint32) * bits)
    packed = (qw << shifts).sum(axis=-1, dtype=np.uint64).astype(np.uint32)
    return packed, scale, bias


def mlx_dequant_np(packed, scale, bias, bits, gs=GS):
    per_word = 32 // bits
    mask = (1 << bits) - 1
    shifts = (np.arange(per_word, dtype=np.uint32) * bits)
    q = ((packed[..., None].astype(np.uint64) >> shifts) & mask).astype(np.float32)
    q = q.reshape(*packed.shape[:-1], -1)
    s = np.repeat(scale, gs, axis=-1)
    b = np.repeat(bias, gs, axis=-1)
    return q * s + b


def to_bf16(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x)).to(torch.bfloat16)


# ------------------------------------------------------------- sanity: packing
def verify_pack_order():
    """Round-trip a real MLX tensor: unpack(pack(unpack(x))) must equal unpack(x)
    and pack(unpack(x)) must equal x bit-for-bit -> proves nibble order."""
    idx = json.load(open(f"{SNAP}/model.safetensors.index.json"))["weight_map"]
    name = "model.layers.79.self_attn.k_proj.weight"
    with safe_open(f"{SNAP}/{idx[name]}", framework="pt") as f:
        real = f.get_tensor(name).numpy()  # uint32 [1024, 512]
    per_word, mask = 8, 0xF
    shifts = (np.arange(per_word, dtype=np.uint32) * 4)
    q = ((real[..., None].astype(np.uint64) >> shifts) & mask).astype(np.uint32)
    repacked = (q << shifts).sum(axis=-1, dtype=np.uint64).astype(np.uint32)
    assert np.array_equal(repacked, real), "nibble order mismatch vs real MLX data"
    print("pack order verified against real MLX tensor")


# ----------------------------------------------------------------- main build
def main():
    verify_pack_order()
    os.makedirs(OUT_DIR, exist_ok=True)

    src = {}
    for s in ("00111", "00112"):
        f = safe_open(f"{SHARD_DIR}/model-{s}-of-00112.safetensors", framework="pt")
        for k in f.keys():
            if k.startswith(L):
                src[k] = f  # lazy handle
    def get(name):
        return src[L + name].get_tensor(L + name)

    out_tensors: dict[str, torch.Tensor] = {}
    report = []

    def quant_into(dst_name, w_bf16: torch.Tensor, bits):
        w = w_bf16.to(torch.float32).numpy()
        packed, scale, bias = mlx_quantize(w, bits)
        deq = mlx_dequant_np(packed, scale, bias, bits)
        err = np.abs(deq - w).max()
        rel = err / (np.abs(w).max() + 1e-9)
        report.append((dst_name, bits, float(rel)))
        out_tensors[dst_name + ".weight"] = torch.from_numpy(packed.astype(np.uint32))
        out_tensors[dst_name + ".scales"] = to_bf16(scale)
        out_tensors[dst_name + ".biases"] = to_bf16(bias)

    # ---- stacked routed experts -> switch_mlp
    for proj in ("gate_proj", "up_proj", "down_proj"):
        packs, scs, bss = [], [], []
        for e in range(192):
            w = get(f"mlp.experts.{e}.{proj}.weight").to(torch.float32).numpy()
            p, s, b = mlx_quantize(w, 4)
            packs.append(p); scs.append(s); bss.append(b)
        out_tensors[f"{L}mlp.switch_mlp.{proj}.weight"] = torch.from_numpy(np.stack(packs))
        out_tensors[f"{L}mlp.switch_mlp.{proj}.scales"] = to_bf16(np.stack(scs))
        out_tensors[f"{L}mlp.switch_mlp.{proj}.biases"] = to_bf16(np.stack(bss))
        # spot-check reconstruction on expert 0
        deq = mlx_dequant_np(packs[0], scs[0], bss[0], 4)
        w0 = get(f"mlp.experts.0.{proj}.weight").to(torch.float32).numpy()
        report.append((f"switch_mlp.{proj}[e0]", 4,
                       float(np.abs(deq - w0).max() / (np.abs(w0).max() + 1e-9))))
        print(f"stacked {proj} done")

    # ---- shared expert + attention projections (4-bit)
    for name in ("mlp.shared_mlp.gate_proj", "mlp.shared_mlp.up_proj",
                 "mlp.shared_mlp.down_proj", "self_attn.q_proj",
                 "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"):
        quant_into(L + name, get(name + ".weight"), 4)

    # ---- router gate (8-bit, mirrors layers 1..79 override)
    quant_into(L + "mlp.router.gate", get("mlp.router.gate.weight"), 8)

    # ---- renames + bf16 passthrough
    out_tensors[L + "mlp.router.expert_bias"] = get("mlp.expert_bias")  # f32
    for name in ("input_layernorm.weight", "post_attention_layernorm.weight",
                 "final_layernorm.weight", "self_attn.q_norm.weight",
                 "self_attn.k_norm.weight", "enorm.weight", "hnorm.weight",
                 "eh_proj.weight"):
        out_tensors[L + name] = get(name)

    mtp_file = "model-mtp.safetensors"
    save_file(out_tensors, f"{OUT_DIR}/{mtp_file}")
    print(f"wrote {mtp_file}: {len(out_tensors)} tensors, "
          f"{os.path.getsize(f'{OUT_DIR}/{mtp_file}')/1e9:.2f} GB")

    # ---- symlinks to the MLX snapshot
    for fn in os.listdir(SNAP):
        if fn.startswith("model-") and fn.endswith(".safetensors"):
            dst = f"{OUT_DIR}/{fn}"
            if not os.path.exists(dst):
                os.symlink(f"{SNAP}/{fn}", dst)
    for fn in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
               "generation_config.json"):
        dst = f"{OUT_DIR}/{fn}"
        if not os.path.exists(dst):
            os.symlink(f"{SNAP}/{fn}", dst)

    # ---- merged index
    idx = json.load(open(f"{SNAP}/model.safetensors.index.json"))
    for k, v in out_tensors.items():
        idx["weight_map"][k] = mtp_file
    idx["metadata"]["total_size"] += sum(t.numel() * t.element_size()
                                         for t in out_tensors.values())
    json.dump(idx, open(f"{OUT_DIR}/model.safetensors.index.json", "w"))

    # ---- config: add the layer-80 router-gate 8-bit override to BOTH quant blocks
    cfg = json.load(open(f"{SNAP}/config.json"))
    for blk in ("quantization", "quantization_config"):
        if blk in cfg:
            cfg[blk]["model.layers.80.mlp.router.gate"] = {"group_size": GS, "bits": 8}
    json.dump(cfg, open(f"{OUT_DIR}/config.json", "w"), indent=2)

    print("\nreconstruction max relative errors (vs bf16 source):")
    for name, bits, rel in report:
        print(f"  {name:45s} {bits}b  {rel:.4f}")
    print(f"\ncheckpoint ready: {OUT_DIR}")


if __name__ == "__main__":
    main()
