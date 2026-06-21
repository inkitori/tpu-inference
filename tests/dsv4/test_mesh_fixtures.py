import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from tests.dsv4.mesh_fixtures import (assert_sharded_like,
                                      assert_threefry_partitionable,
                                      dsv4_mesh)


def test_mesh_is_8_chip_attn_dp(dsv4_mesh):
    # 6-axis production mesh, total 8 devices.
    assert dsv4_mesh.devices.size == 8
    assert set(dsv4_mesh.axis_names) == {
        "data", "attn_dp", "attn_dp_expert", "expert", "model", "dcp"
    }


def test_assert_sharded_like_passes_for_committed_array(dsv4_mesh):
    assert_threefry_partitionable()
    # 8 rows so it shards across the 4-way "model" axis cleanly.
    x = jnp.arange(8 * 4, dtype=jnp.float32).reshape(8, 4)
    sh = NamedSharding(dsv4_mesh, P("model", None))
    x = jax.device_put(x, sh)
    assert_sharded_like(x, dsv4_mesh, P("model", None))  # must not raise
