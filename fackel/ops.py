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
    """Muon's 5th-order Newton-Schulz orthogonalization."""
    if g.ndim != 2:
        return g

    transpose = g.shape[0] < g.shape[1]
    x = g.T if transpose else g

    x = x / (jnp.linalg.norm(x, ord="fro") + eps)

    a, b, c = 3.4445, -4.7750, 2.0315
    eye = jnp.eye(x.shape[1])

    for _ in range(steps):
        xTx = x.T @ x
        x = x @ (a * eye + b * xTx + c * (xTx @ xTx))

    x = x.T if transpose else x

    scale = jnp.sqrt(max(g.shape[0], g.shape[1]))
    return x * scale


def clip_grad_norm(grads, max_norm=1.0):
    """
    Clips the global norm of a PyTree of gradients.
    """
    g_norm = norm(grads)
    clip_coef = max_norm / (g_norm + 1e-6)
    clip_coef = jnp.minimum(1.0, clip_coef)
    clipped_grads = tmap(lambda g: g * clip_coef, grads)

    return clipped_grads
