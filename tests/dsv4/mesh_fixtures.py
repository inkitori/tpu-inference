"""Production-mesh fixture (TP=8 + EP + DP-attention) and sharding oracle for DSV4 tests."""
import jax
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tests.layers.common.utils import get_spmd_mesh


def assert_threefry_partitionable() -> None:
    """8-way-sharded threefry RNG is bit-identical to single-device only when on."""
    assert jax.config.jax_threefry_partitionable, (
        "jax_threefry_partitionable must be True for shard-equivalence")


@pytest.fixture(scope="module")
def dsv4_mesh():
    """Production sharding: TP=8 split as 2-way attn-DP x 4-way model, + EP axes."""
    if len(jax.devices()) < 8:
        pytest.skip("DSV4 production mesh requires 8 TPU devices")
    mesh = get_spmd_mesh(num_devices=8, enable_attn_dp=True)
    yield mesh


def assert_sharded_like(arr: jax.Array, mesh: Mesh, spec: P) -> None:
    """Assert on the COMMITTED sharding of a real array (eval_shape gives None)."""
    assert isinstance(arr.sharding, NamedSharding), (
        f"expected NamedSharding, got {type(arr.sharding)}")
    expected = NamedSharding(mesh, spec)
    assert arr.sharding == expected, (
        f"sharding mismatch: got {arr.sharding!r}, want {expected!r}")
