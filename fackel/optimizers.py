import jax.numpy as jnp
from fackel.ops import tmap, zeros_like, orthogonalize


def sgd(lr):
    """Stateless: no opt_state needed, just update(grads, params)."""

    def update(grads, params):
        return tmap(lambda p, g: p - lr * g, params, grads)

    return update


def momentum(lr, beta=0.9):
    """Needs state: velocity carried across steps."""

    def init(params):
        return zeros_like(params)  # velocity

    def update(grads, opt_state, params):
        velocity = tmap(lambda v, g: beta * v + g, opt_state, grads)
        new_params = tmap(lambda p, v: p - lr * v, params, velocity)
        return new_params, velocity

    return init, update


def adam(lr, b1=0.9, b2=0.999, eps=1e-8):
    """Needs state: first/second moment estimates + step count."""

    def init(params):
        return {
            "m": zeros_like(params),
            "v": zeros_like(params),
            "t": 0,
        }

    def update(grads, opt_state, params):
        t = opt_state["t"] + 1

        m = tmap(lambda m, g: b1 * m + (1 - b1) * g, opt_state["m"], grads)
        v = tmap(lambda v, g: b2 * v + (1 - b2) * g**2, opt_state["v"], grads)

        m_hat = tmap(lambda m: m / (1 - b1**t), m)
        v_hat = tmap(lambda v: v / (1 - b2**t), v)

        new_params = tmap(
            lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps),
            params,
            m_hat,
            v_hat,
        )
        new_opt_state = {"m": m, "v": v, "t": t}
        return new_params, new_opt_state

    return init, update


def adamw(lr, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.01):
    """Adam with DECOUPLED weight decay - decay applied directly to params"""

    def init(params):
        return {
            "m": zeros_like(params),
            "v": zeros_like(params),
            "t": 0,
        }

    def update(grads, opt_state, params):
        t = opt_state["t"] + 1

        m = tmap(lambda m, g: b1 * m + (1 - b1) * g, opt_state["m"], grads)
        v = tmap(lambda v, g: b2 * v + (1 - b2) * g**2, opt_state["v"], grads)

        m_hat = tmap(lambda m: m / (1 - b1**t), m)
        v_hat = tmap(lambda v: v / (1 - b2**t), v)

        new_params = tmap(
            lambda p, mh, vh: (
                p - lr * mh / (jnp.sqrt(vh) + eps) - lr * weight_decay * p
            ),
            params,
            m_hat,
            v_hat,
        )
        new_opt_state = {"m": m, "v": v, "t": t}
        return new_params, new_opt_state

    return init, update


def muon(lr, beta=0.95):

    def init(params):
        return zeros_like(params)

    def update(grads, state, params):

        momentum = tmap(
            lambda m, g: beta * m + g,
            state,
            grads,
        )

        updates = tmap(orthogonalize, momentum)

        params = tmap(
            lambda p, u: p - lr * u,
            params,
            updates,
        )

        return params, momentum

    return init, update
