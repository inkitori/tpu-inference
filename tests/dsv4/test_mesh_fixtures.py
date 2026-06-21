import jax
import jax.numpy as jnp
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from tests.dsv4.mesh_fixtures import (assert_sharded_like,
                                      assert_threefry_partitionable,
                                      dsv4_mesh)


def test_mesh_is_8_chip_attn_dp(dsv4_mesh):
    # 6-axis production mesh, total 8 devices.
    assert dsv4_mesh.devices.size == 8
    # Assert ordered axis names — a permutation regression must not silently pass.
    assert dsv4_mesh.axis_names == ("data", "attn_dp", "attn_dp_expert", "expert", "model", "dcp")
    # Assert axis sizes in order: data=1, attn_dp=2, attn_dp_expert=1, expert=1, model=4, dcp=1.
    assert tuple(dsv4_mesh.shape.values()) == (1, 2, 1, 1, 4, 1)


def test_assert_sharded_like_passes_for_committed_array(dsv4_mesh):
    assert_threefry_partitionable()
    # 8 rows so it shards across the 4-way "model" axis cleanly.
    x = jnp.arange(8 * 4, dtype=jnp.float32).reshape(8, 4)
    sh = NamedSharding(dsv4_mesh, P("model", None))
    x = jax.device_put(x, sh)
    assert_sharded_like(x, dsv4_mesh, P("model", None))  # must not raise


def test_assert_sharded_like_raises_on_mismatch(dsv4_mesh):
    # Commit a fully-replicated array, then assert a *different* spec — oracle must raise.
    # Use 8 rows (divisible by model-axis size 4) so the shape is compatible with P("model", None);
    # the only reason AssertionError fires is the sharding mismatch, not a dimension error.
    x = jnp.arange(8 * 4, dtype=jnp.float32).reshape(8, 4)
    replicated_sh = NamedSharding(dsv4_mesh, P(None, None))
    x = jax.device_put(x, replicated_sh)
    with pytest.raises(AssertionError):
        assert_sharded_like(x, dsv4_mesh, P("model", None))
