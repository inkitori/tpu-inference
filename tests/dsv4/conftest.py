"""Env gate for all DSV4 tests. MUST run before any tpu_inference/jax import.

NEW_MODEL_DESIGN=1 selects the 6-axis ShardingAxisNameBase + production mesh;
without it the mesh fixture and any MLA/DP-attention model build raise ValueError.
These are read at import time, so set them here (setdefault, so a shell-provided
value wins) AND prefer the shell prefix `NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm`.
"""
import os

os.environ.setdefault("NEW_MODEL_DESIGN", "1")
os.environ.setdefault("MODEL_IMPL_TYPE", "vllm")


def pytest_configure(config):
    # `slow` is NOT registered by the repo's tests/conftest.py (only
    # disable_jax_cache, bvt). Register it here so Task 15's @pytest.mark.slow
    # and `-m slow` selection work without an "unknown marker" warning.
    config.addinivalue_line(
        "markers", "slow: loads the real ~187 GiB model; coherence milestone only.")
