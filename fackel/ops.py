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


def orthogonalize(g, steps=5, eps=1e-7):
    """Muon's Newton-Schulz orthogonalization."""

    if g.ndim != 2:
        return g

    transpose = g.shape[0] > g.shape[1]

    x = jnp.where(transpose, g.T, g)

    x = x / (jnp.linalg.norm(x) + eps)

    a = 3.4445
    b = -4.7750
    c = 2.0315

    def body(_, x):
        A = x @ x.T
        B = b * A + c * (A @ A)
        return a * x + B @ x

    x = jax.lax.fori_loop(0, steps, body, x)

    return jnp.where(transpose, x.T, x)
