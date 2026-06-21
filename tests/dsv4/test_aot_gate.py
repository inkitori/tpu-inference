import jax
import jax.numpy as jnp

from tests.dsv4.aot_gate import aot_compile, make_aval


def test_aot_compile_returns_compiled_executable():
    def f(x, y):
        return x @ y

    a = make_aval((16, 16), jnp.bfloat16)
    b = make_aval((16, 16), jnp.bfloat16)
    compiled = aot_compile(f, a, b)
    assert isinstance(compiled, jax.stages.Compiled)


def test_aot_compile_surfaces_compile_error():
    # A shape-incompatible matmul must fail at lower/compile time, not silently.
    def bad(x, y):
        return x @ y  # (16,16) @ (8,8) -> contraction mismatch

    a = make_aval((16, 16), jnp.float32)
    b = make_aval((8, 8), jnp.float32)
    import pytest
    with pytest.raises(Exception):
        aot_compile(bad, a, b)
