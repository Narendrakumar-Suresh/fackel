import jax
import jax.numpy as jnp


def tmap(fn, *trees):
    """Elementwise op across one or more matching pytrees."""
    return jax.tree_util.tree_map(fn, *trees)


def add(a, b):
    return tmap(lambda x, y: x + y, a, b)


def sub(a, b):
    return tmap(lambda x, y: x - y, a, b)


def scale(t, s):
    return tmap(lambda x: x * s, t)


def zeros_like(t):
    return tmap(jnp.zeros_like, t)


def norm(t):
    leaves = jax.tree_util.tree_leaves(t)
    return jnp.sqrt(sum(jnp.sum(x**2) for x in leaves))
