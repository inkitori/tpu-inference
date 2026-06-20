# Phase 0 completion report

**Status:** DONE

## What was added

### Deliverable 1: Incremental HF decode oracle (`build_hf_decode_oracle`)

Added to `tests/models/jax/glm_moe_dsa_harness.py`:

- `MEDIUM_PAGE_SIZE = 128`, `MEDIUM_SEQ_DENSE = 2040`, `MEDIUM_SEQ_SPARSE = 3000` constants (§B11)
- `_MEDIUM_CONFIG_KWARGS` dict with exact §B11 values (see below)
- `medium_glm_moe_dsa_config(**overrides)` fixture function
- `build_hf_decode_oracle(input_ids, decode_ids, *, cfg, seed, randomize_buffers)` function

### Deliverable 2: Medium config fixture (`medium_glm_moe_dsa_config`)

Added to same file using the exact §B11 values below.

---

## Exact §B11 values used (transcribed from core.md §B11 verbatim)

```python
_MEDIUM_CONFIG_KWARGS = dict(
    hidden_size=6144,
    num_hidden_layers=6,          # 3 dense + 3 sparse (first_k_dense_replace=3 default)
    num_attention_heads=64,
    num_key_value_heads=64,
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    qk_nope_head_dim=192,
    v_head_dim=256,
    index_head_dim=128,
    index_n_heads=32,
    index_topk=2048,
    index_topk_freq=4,            # kwargs-only: drives indexer_types in __post_init__
    index_skip_topk_offset=3,     # (not persisted as an attribute)
    n_routed_experts=16,
    num_experts_per_tok=8,
    moe_intermediate_size=512,
    vocab_size=2048,
    rope_type="default",          # plain RoPE, no YaRN
)
```

`page_size=128` is a harness constant (`MEDIUM_PAGE_SIZE`) not a config kwarg — same pattern as `TINY_PAGE_SIZE=16`.

---

## Decode oracle API discovered from `modeling_glm_moe_dsa.py`

- `GlmMoeDsaForCausalLM.forward` does **NOT** accept `cache_position` — the decode cursor flows solely via `position_ids` + `past_key_values.get_seq_length()`
- `DynamicCache(config=model.config)` is how the cache is constructed (line 743 of modeling file); it auto-wires per-layer types from `config.layer_types` — DSA layers get `DynamicIndexedLayer` with the extra `indexer_keys` buffer
- Decode loop pattern:
  1. Prefill: `position_ids = arange(prompt_len).unsqueeze(0)`, pass `past_key_values=cache, use_cache=True`
  2. Each decode step: `pos = [[past_key_values.get_seq_length()]]`, pass single token + pos + growing cache
- `update_indexer` is called internally inside `GlmMoeDsaIndexer.forward` (only for "full" layers); no explicit caller needed in the oracle wrapper

---

## Test results

Full test suite run: `28 passed, 4 warnings in 43.42s`

Pytest summary line: `28 passed, 4 warnings in 43.42s`

### Decode oracle achieved maxabs deltas (fp32)

| Step | cum_len | maxabs |
|------|---------|--------|
| 0    | 4       | 2.35e-06 |
| 1    | 5       | 2.12e-06 |
| 2    | 6       | 1.97e-06 |
| 3    | 7       | 2.26e-06 |

All steps well below 1e-4 tolerance (~50× margin). The oracle is numerically identical to fresh full forwards, confirming the decode cache threading is correct.

---

## Concerns

None. Both gates pass cleanly with large tolerance margins.
