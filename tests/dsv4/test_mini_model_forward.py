"""Task 12 -- INTEGRATION CRUX: synthetic mini-model build + load + one forward.

Builds the DeepSeek-V4-Flash *mini* model on the torchax/vLLM path
(``MODEL_IMPL_TYPE=vllm``, ``NEW_MODEL_DESIGN=1``) with synthetic FP4-expert /
FP8-linear dummy weights, then runs ONE 64-token forward on the real v6e-8
production mesh (TP=8 split 2-way attn-DP x 4-way model, + EP), asserting the
logits have the right shape and are FINITE.

This is the first time the WHOLE attention stack runs together end-to-end, so it
catches: untraceable ops, NaN/Inf, shape/sharding bugs, the FP4-GMM compile, the
wo_a PWAL dequant path, and the mla_swa kernel wiring. Per-block NUMERICAL parity
is already pinned by Tasks 5/7/8/9/10/11 -- this test does NOT build a pure-torch
reference; finite logits from the real ``model_fn`` on the real mesh is the bar.

No full-model load: synthetic small-config weights in the real quant formats only.
"""
import jax
import torch

# NEW_MODEL_DESIGN=1 and MODEL_IMPL_TYPE=vllm MUST already be set in the env
# (tests/dsv4/conftest.py sets them before collection; also pass the shell prefix).
# Do NOT rely on os.environ.setdefault here -- it runs after tpu_inference is
# imported and is therefore too late to affect ShardingAxisName/mesh selection.
from tests.dsv4.build_mini_model import build_mini_model, run_mini_forward
from tests.dsv4.mesh_fixtures import (assert_threefry_partitionable,  # noqa
                                      dsv4_mesh)


def test_mini_model_eager_forward_runs(dsv4_mesh):
    assert_threefry_partitionable()
    with jax.set_mesh(dsv4_mesh):
        model, vllm_config = build_mini_model(dsv4_mesh)
        n = 64  # <=128 (one window)
        input_ids = torch.arange(n, dtype=torch.int32) % 1280
        positions = torch.arange(n, dtype=torch.int32)
        logits = run_mini_forward(model, vllm_config, input_ids, positions)
    assert logits.shape[0] == n
    assert torch.isfinite(logits.float()).all(), "non-finite logits (NaN/Inf)"
