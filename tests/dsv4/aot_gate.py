"""AOT compile gate: jit(fn).lower(*avals).compile() forces Mosaic passes.

Use as a pre-flight before any full run: it surfaces untraceable/Mosaic compile
errors (e.g. the FP4 GMM MosaicError) in seconds with no weights/data.
"""
import jax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


def make_aval(shape, dtype, mesh=None, spec=None) -> jax.ShapeDtypeStruct:
    if mesh is not None and spec is not None:
        return jax.ShapeDtypeStruct(shape=shape, dtype=dtype,
                                    sharding=NamedSharding(mesh, spec))
    return jax.ShapeDtypeStruct(shape=shape, dtype=dtype)


def aot_compile(fn, *args, mesh=None, static_argnums=(),
                static_argnames=()) -> jax.stages.Compiled:
    """Lower AND compile fn against the given avals/arrays. .compile() is what
    runs the Mosaic backend; .lower() alone only serializes (spec 6.3 A)."""
    jitted = jax.jit(fn, static_argnums=static_argnums,
                     static_argnames=static_argnames)
    if mesh is not None:
        with jax.set_mesh(mesh):
            return jitted.lower(*args).compile()
    return jitted.lower(*args).compile()
