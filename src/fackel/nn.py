import jax
import jax.numpy as jnp
from jax.nn.initializers import he_normal, orthogonal, glorot_uniform


def linear(in_dim, out_dim):
    def init(key):
        w = he_normal()(key, (in_dim, out_dim))
        b = jnp.zeros(out_dim)
        return {"w": w, "b": b}, {}

    def apply(params, state, x, **kwargs):
        out = x @ params["w"] + params["b"]
        return out, state

    return init, apply


def _conv_init_params(key, kernel_shape, out_ch):

    w = he_normal()(key, kernel_shape)
    b = jnp.zeros(out_ch)
    return {"w": w, "b": b}


def conv1d(in_ch, out_ch, kernel_size, stride=1, padding="SAME"):

    kernel_shape = (kernel_size, in_ch, out_ch)

    def init(key):
        return _conv_init_params(key, kernel_shape, out_ch), {}

    def apply(params, state, x, **kwargs):
        y = jax.lax.conv_general_dilated(
            x,
            params["w"],
            window_strides=(stride,),
            padding=padding,
            dimension_numbers=("NWC", "WIO", "NWC"),
        )
        return y + params["b"], state

    return init, apply


def conv2d(in_ch, out_ch, kernel_size, stride=1, padding="SAME"):
    if isinstance(kernel_size, int):
        kh, kw = kernel_size, kernel_size
    else:
        kh, kw = kernel_size

    if isinstance(stride, int):
        sh, sw = stride, stride
    else:
        sh, sw = stride

    kernel_shape = (kh, kw, in_ch, out_ch)

    def init(key):
        return _conv_init_params(key, kernel_shape, out_ch), {}

    def apply(params, state, x, **kwargs):
        y = jax.lax.conv_general_dilated(
            x,
            params["w"],
            window_strides=(sh, sw),
            padding=padding,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        return y + params["b"], state

    return init, apply


def conv3d(in_ch, out_ch, kernel_size, stride=1, padding="SAME"):
    if isinstance(kernel_size, int):
        kd, kh, kw = kernel_size, kernel_size, kernel_size
    else:
        kd, kh, kw = kernel_size
    if isinstance(stride, int):
        sd, sh, sw = stride, stride, stride
    else:
        sd, sh, sw = stride

    kernel_shape = (kd, kh, kw, in_ch, out_ch)

    def init(key):
        return _conv_init_params(key, kernel_shape, out_ch), {}

    def apply(params, state, x, **kwargs):
        y = jax.lax.conv_general_dilated(
            x,
            params["w"],
            window_strides=(sd, sh, sw),
            padding=padding,
            dimension_numbers=("NDHWC", "DHWIO", "NDHWC"),
        )
        return y + params["b"], state

    return init, apply


def sequential(*layers):
    def init(key):
        params = []
        states = []
        for layer in layers:
            if isinstance(layer, tuple):
                key, subkey = jax.random.split(key)
                p, s = layer[0](subkey)
                params.append(p)
                states.append(s)
            else:
                params.append(None)
                states.append(None)
        return params, states

    def apply(params, states, x, rng=None, is_training=True, **kwargs):
        new_states = []
        for layer, p, s in zip(layers, params, states):
            current_rng = None
            if rng is not None:
                rng, current_rng = jax.random.split(rng)

            if isinstance(layer, tuple):
                x, new_s = layer[1](
                    p, s, x, rng=current_rng, is_training=is_training, **kwargs
                )
                new_states.append(new_s)
            else:
                x = layer(x)
                new_states.append(None)
        return x, new_states

    return init, apply


def flatten(x):
    return x.reshape(x.shape[0], -1)


def dropout(rate):
    def init(key):
        return None, {}

    def apply(params, state, x, rng=None, is_training=True, **kwargs):
        if not is_training or rate == 0.0:
            return x, state

        if rng is None:
            raise ValueError(
                "An `rng` key must be provided to Dropout during training."
            )

        keep_prob = 1.0 - rate
        mask = jax.random.bernoulli(rng, p=keep_prob, shape=x.shape)
        out = jnp.where(mask, x / keep_prob, 0.0)
        return out, state

    return init, apply


def batch_norm(num_features, momentum=0.9, epsilon=1e-5):

    def init(key):
        params = {"scale": jnp.ones(num_features), "shift": jnp.zeros(num_features)}
        state = {
            "running_mean": jnp.zeros(num_features),
            "running_var": jnp.ones(num_features),
        }
        return params, state

    def apply(params, state, x, is_training=True, **kwargs):
        reduce_axes = tuple(range(x.ndim - 1))

        if is_training:
            batch_mean = jnp.mean(x, axis=reduce_axes)
            batch_var = jnp.var(x, axis=reduce_axes)

            new_mean = momentum * state["running_mean"] + (1.0 - momentum) * batch_mean
            new_var = momentum * state["running_var"] + (1.0 - momentum) * batch_var
            new_state = {"running_mean": new_mean, "running_var": new_var}

            mean, var = batch_mean, batch_var
        else:
            mean, var = state["running_mean"], state["running_var"]
            new_state = state

        x_norm = (x - mean) / jnp.sqrt(var + epsilon)
        out = x_norm * params["scale"] + params["shift"]

        return out, new_state

    return init, apply

# --- Core Primitive Additions ---

def rmsnorm(dim, eps=1e-6):
    def init(key):
        return {"scale": jnp.ones(dim)}, {}

    def apply(params, state, x, **kwargs):
        # Calculate mean across the last dimension (hidden feature size)
        rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + eps)
        
        # Enforce robust broadcasting by matching scale shape to input rank
        scale = params["scale"]
        if x.ndim > 1:
            # Reshape scale from (Dim,) to (1, ..., 1, Dim) to match input rank
            scale_shape = [1] * (x.ndim - 1) + [dim]
            scale = scale.reshape(scale_shape)
            
        return (x / rms) * scale, state

    return init, apply

def rnn(in_dim, out_dim):
    def init(key):
        k1, k2 = jax.random.split(key)
        w_i = glorot_uniform()(k1, (in_dim, out_dim))
        w_h = orthogonal()(k2, (out_dim, out_dim))
        b = jnp.zeros(out_dim)
        return {"w_i": w_i, "w_h": w_h, "b": b}, {}

    def apply(params, state, x, **kwargs):
        # x shape: (Batch, Seq, Dim). Swap axes to (Seq, Batch, Dim) for scan
        x_seq = jnp.swapaxes(x, 0, 1)
        batch_size = x.shape[0]

        def step(h, xt):
            h_next = jnp.tanh(xt @ params["w_i"] + h @ params["w_h"] + params["b"])
            return h_next, h_next

        init_carry = jnp.zeros((batch_size, out_dim))
        _, out_seq = jax.lax.scan(step, init_carry, x_seq)

        # Swap back to (Batch, Seq, Dim)
        return jnp.swapaxes(out_seq, 0, 1), state

    return init, apply


def gru(in_dim, out_dim):
    def init(key):
        k1, k2 = jax.random.split(key)
        w_i = glorot_uniform()(k1, (in_dim, 3 * out_dim))
        w_h = orthogonal()(k2, (out_dim, 3 * out_dim))
        b_i = jnp.zeros(3 * out_dim)
        b_h = jnp.zeros(3 * out_dim)
        return {"w_i": w_i, "w_h": w_h, "b_i": b_i, "b_h": b_h}, {}

    def apply(params, state, x, **kwargs):
        x_seq = jnp.swapaxes(x, 0, 1)
        batch_size = x.shape[0]

        def step(h, xt):
            gate_inputs_i = xt @ params["w_i"] + params["b_i"]
            gate_inputs_h = h @ params["w_h"] + params["b_h"]

            ir, iz, in_ = jnp.split(gate_inputs_i, 3, axis=-1)
            hr, hz, hn = jnp.split(gate_inputs_h, 3, axis=-1)

            r = jax.nn.sigmoid(ir + hr)
            z = jax.nn.sigmoid(iz + hz)

            # The GRU reset gate applies only to the hidden state computation of the new candidate
            n = jnp.tanh(in_ + r * hn)
            h_next = (1.0 - z) * n + z * h

            return h_next, h_next

        init_carry = jnp.zeros((batch_size, out_dim))
        _, out_seq = jax.lax.scan(step, init_carry, x_seq)

        return jnp.swapaxes(out_seq, 0, 1), state

    return init, apply


def lstm(in_dim, out_dim):
    def init(key):
        k1, k2 = jax.random.split(key)
        # Concatenate weights for i, f, g, o gates
        w_i = glorot_uniform()(k1, (in_dim, 4 * out_dim))
        w_h = orthogonal()(k2, (out_dim, 4 * out_dim))

        b = jnp.zeros(4 * out_dim)
        # Initialize forget gate bias to 1.0 (standard practice to prevent early forgetting)
        b = b.at[out_dim : 2 * out_dim].set(1.0)

        return {"w_i": w_i, "w_h": w_h, "b": b}, {}

    def apply(params, state, x, **kwargs):
        x_seq = jnp.swapaxes(x, 0, 1)
        batch_size = x.shape[0]

        def step(carry, xt):
            h, c = carry
            gates = xt @ params["w_i"] + h @ params["w_h"] + params["b"]
            i, f, g, o = jnp.split(gates, 4, axis=-1)

            i = jax.nn.sigmoid(i)
            f = jax.nn.sigmoid(f)
            o = jax.nn.sigmoid(o)
            g = jnp.tanh(g)

            c_next = f * c + i * g
            h_next = o * jnp.tanh(c_next)

            return (h_next, c_next), h_next

        init_carry = (
            jnp.zeros((batch_size, out_dim)),
            jnp.zeros((batch_size, out_dim)),
        )
        _, out_seq = jax.lax.scan(step, init_carry, x_seq)

        return jnp.swapaxes(out_seq, 0, 1), state

    return init, apply


def rglru(in_dim, out_dim):
    def init(key):
        k1, k2, k3, k4 = jax.random.split(key, 4)

        w_r = glorot_uniform()(k1, (in_dim, out_dim))
        w_i = glorot_uniform()(k2, (in_dim, out_dim))
        w_x = glorot_uniform()(k3, (in_dim, out_dim))

        b_r = jnp.zeros(out_dim)
        b_i = jnp.zeros(out_dim)
        b_x = jnp.zeros(out_dim)

        theta = jax.random.uniform(k4, (out_dim,), minval=-1.0, maxval=1.0)

        return {
            "w_r": w_r,
            "w_i": w_i,
            "w_x": w_x,
            "b_r": b_r,
            "b_i": b_i,
            "b_x": b_x,
            "theta": theta,
        }, {}

    def apply(params, state, x, **kwargs):
        r_pre = x @ params["w_r"] + params["b_r"]
        i_pre = x @ params["w_i"] + params["b_i"]
        x_proj = x @ params["w_x"] + params["b_x"]

        r = jax.nn.sigmoid(r_pre)
        i = jax.nn.sigmoid(i_pre)

        c = 8.0 * jax.nn.softplus(params["theta"])
        a = jnp.exp(-c * r)

        m = jnp.sqrt(1.0 - a**2) * i * x_proj

        def combine(e1, e2):
            a1, m1 = e1
            a2, m2 = e2
            return a1 * a2, a2 * m1 + m2

        _, out_seq = jax.lax.associative_scan(combine, (a, m), axis=1)

        return out_seq, state

    return init, apply


def embedding(num_embeddings, embedding_dim):
    """
    Standard lookup table that stores embeddings of a fixed dictionary and size.
    """

    def init(key):
        # Initializing with a truncated normal distribution or scaled normal
        # (mean=0, std=0.02) is standard practice for NLP embeddings.
        w = jax.random.normal(key, (num_embeddings, embedding_dim)) * 0.02
        return {"w": w}, {}

    def apply(params, state, x, **kwargs):
        # x is an array of integer indices.
        # JAX handles multi-dimensional indexing automatically (e.g., batch x seq_len)
        out = params["w"][x]
        return out, state

    return init, apply
