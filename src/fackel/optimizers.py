import jax
import jax.numpy as jnp
from fackel.ops import tmap, zeros_like, orthogonalize


def sgd(lr):
    def init(params):
        return {"t": 0}

    def update(grads, opt_state, params):
        t = opt_state["t"] + 1
        current_lr = lr(t) if callable(lr) else lr

        new_params = tmap(lambda p, g: p - current_lr * g, params, grads)
        return new_params, {"t": t}

    return init, update


def momentum(lr, beta=0.9):
    def init(params):
        return {"velocity": zeros_like(params), "t": 0}

    def update(grads, opt_state, params):
        t = opt_state["t"] + 1
        current_lr = lr(t) if callable(lr) else lr

        velocity = tmap(lambda v, g: beta * v + g, opt_state["velocity"], grads)
        new_params = tmap(lambda p, v: p - current_lr * v, params, velocity)

        return new_params, {"velocity": velocity, "t": t}

    return init, update


def adam(lr, b1=0.9, b2=0.999, eps=1e-8):
    def init(params):
        return {
            "m": zeros_like(params),
            "v": zeros_like(params),
            "t": 0,
        }

    def update(grads, opt_state, params):
        t = opt_state["t"] + 1
        current_lr = lr(t) if callable(lr) else lr

        m = tmap(lambda m, g: b1 * m + (1 - b1) * g, opt_state["m"], grads)
        v = tmap(lambda v, g: b2 * v + (1 - b2) * g**2, opt_state["v"], grads)

        m_hat = tmap(lambda m: m / (1 - b1**t), m)
        v_hat = tmap(lambda v: v / (1 - b2**t), v)

        new_params = tmap(
            lambda p, mh, vh: p - current_lr * mh / (jnp.sqrt(vh) + eps),
            params,
            m_hat,
            v_hat,
        )
        new_opt_state = {"m": m, "v": v, "t": t}
        return new_params, new_opt_state

    return init, update


def adamw(lr, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.01):
    def init(params):
        return {
            "m": zeros_like(params),
            "v": zeros_like(params),
            "t": 0,
        }

    def update(grads, opt_state, params):
        t = opt_state["t"] + 1
        current_lr = lr(t) if callable(lr) else lr

        m = tmap(lambda m, g: b1 * m + (1 - b1) * g, opt_state["m"], grads)
        v = tmap(lambda v, g: b2 * v + (1 - b2) * g**2, opt_state["v"], grads)

        m_hat = tmap(lambda m: m / (1 - b1**t), m)
        v_hat = tmap(lambda v: v / (1 - b2**t), v)

        new_params = tmap(
            lambda p, mh, vh: (
                p
                - current_lr * mh / (jnp.sqrt(vh) + eps)
                - current_lr * weight_decay * p
            ),
            params,
            m_hat,
            v_hat,
        )
        new_opt_state = {"m": m, "v": v, "t": t}
        return new_params, new_opt_state

    return init, update


def muon(
    lr,
    muon_beta=0.95,
    adamw_fallback_lr_ratio=1.0,
    weight_decay=0.01,
    b1=0.9,
    b2=0.999,
    ns_steps=5,
):
    def init(params):
        return {
            "step": 0,
            "muon_mom": zeros_like(params),
            "adam_m": zeros_like(params),
            "adam_v": zeros_like(params),
        }

    def update(grads, opt_state, params):
        step = opt_state["step"] + 1
        current_lr = lr(step) if callable(lr) else lr

        def leaf_update(p, g, mom, m, v):
            if p.ndim == 2:
                new_mom = muon_beta * mom + g
                update_direction = orthogonalize(new_mom, steps=ns_steps)
                new_p = p - current_lr * weight_decay * p
                new_p = new_p - current_lr * update_direction
                return new_p, new_mom, m, v
            else:
                new_m = b1 * m + (1 - b1) * g
                new_v = b2 * v + (1 - b2) * (g**2)
                m_hat = new_m / (1 - b1**step)
                v_hat = new_v / (1 - b2**step)
                adam_lr = current_lr * adamw_fallback_lr_ratio
                new_p = p - adam_lr * weight_decay * p
                new_p = new_p - adam_lr * m_hat / (jnp.sqrt(v_hat) + 1e-8)
                return new_p, mom, new_m, new_v

        results = tmap(
            leaf_update,
            params,
            grads,
            opt_state["muon_mom"],
            opt_state["adam_m"],
            opt_state["adam_v"],
        )

        is_tuple_leaf = lambda x: isinstance(x, tuple)

        new_params = jax.tree_util.tree_map(
            lambda x: x[0], results, is_leaf=is_tuple_leaf
        )

        new_opt_state = {
            "step": step,
            "muon_mom": jax.tree_util.tree_map(
                lambda x: x[1], results, is_leaf=is_tuple_leaf
            ),
            "adam_m": jax.tree_util.tree_map(
                lambda x: x[2], results, is_leaf=is_tuple_leaf
            ),
            "adam_v": jax.tree_util.tree_map(
                lambda x: x[3], results, is_leaf=is_tuple_leaf
            ),
        }
        return new_params, new_opt_state

    return init, update
