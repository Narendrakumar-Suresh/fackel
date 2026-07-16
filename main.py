import jax
import jax.numpy as jnp

from fackel.nn import linear
from fackel.activations import relu

# ---- create layers: each call gives its own init/apply pair ----
init1, apply1 = linear(4, 16)
init2, apply2 = linear(16, 3)

# ---- init params ----
key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)

p1 = init1(k1)
p2 = init2(k2)

params = {"l1": p1, "l2": p2}


# ---- forward function, written by hand, no dispatch ----
def forward(params, x):
    x = apply1(params["l1"], x)
    x =relu(x)
    x = apply2(params["l2"], x)
    return x


if __name__ == "__main__":
    x = jnp.ones((1, 4))
    y = forward(params, x)
    print("output:", y)
    print("output shape:", y.shape)