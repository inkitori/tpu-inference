"""Validate the ragged-metadata builders match the mla_swa kernel-test pattern."""
import jax.numpy as jnp
import numpy as np

from tpu_inference.layers.vllm.custom_ops.deepseek_v4_attention import \
    build_swa_ragged_metadata


class _MD:
    """Minimal stand-in for AttentionMetadata (only the 4 fields read)."""
    def __init__(self, seq_lens, block_tables, query_start_loc,
                 request_distribution):
        self.seq_lens = seq_lens
        self.block_tables = block_tables
        self.query_start_loc = query_start_loc
        self.request_distribution = request_distribution


def test_build_metadata_passes_through_canonical_fields():
    # Mirror mla_swa_test step-2: 2 decode + 1 prefill(len 3); total seqs = 3.
    seq_lens = jnp.array([5, 5, 7], dtype=jnp.int32)
    block_tables = jnp.arange(3 * 4, dtype=jnp.int32)           # flat page table
    cu_q = jnp.array([0, 1, 2, 5], dtype=jnp.int32)             # decode,decode,prefill(3)
    distribution = jnp.array([2, 2, 3], dtype=jnp.int32)        # [decode, prefill_bnd, total]
    md = _MD(seq_lens, block_tables, cu_q, distribution)

    kv_lens, page_indices, cu_q_lens, dist = build_swa_ragged_metadata(md)
    np.testing.assert_array_equal(np.asarray(kv_lens), np.asarray(seq_lens))
    np.testing.assert_array_equal(np.asarray(page_indices),
                                  np.asarray(block_tables))
    np.testing.assert_array_equal(np.asarray(cu_q_lens), np.asarray(cu_q))
    np.testing.assert_array_equal(np.asarray(dist), np.asarray(distribution))
    assert dist.shape == (3,)
    assert cu_q_lens.shape[0] == seq_lens.shape[0] + 1
