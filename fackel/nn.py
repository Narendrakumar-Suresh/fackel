import jax, jax.numpy as jnp


def linear(in_dim, out_dim):
    def init(key):
        wkey, bkey = jax.random.split(key)
        w = jax.random.normal(wkey, (in_dim, out_dim)) * jnp.sqrt(2.0 / in_dim)
        b = jnp.zeros(out_dim)
        return {"w": w, "b": b}

    def apply(params, x):
        return x @ params["w"] + params["b"]

    return init, apply

def _conv_init_params(key, kernel_shape, out_ch, fan_in):
    wkey, bkey = jax.random.split(key)
    w = jax.random.normal(wkey, kernel_shape) * jnp.sqrt(2.0 / fan_in)
    b = jnp.zeros(out_ch)
    return {"w": w, "b": b}


def conv1d(in_ch, out_ch, kernel_size, stride=1, padding="SAME"):
    """1D conv. Input x shape: (batch, length, channels) -> NWC."""
    kernel_shape = (kernel_size, in_ch, out_ch)  # WIO
    fan_in = kernel_size * in_ch

    def init(key):
        return _conv_init_params(key, kernel_shape, out_ch, fan_in)

    def apply(params, x):
        y = jax.lax.conv_general_dilated(
            x,
            params["w"],
            window_strides=(stride,),
            padding=padding,
            dimension_numbers=("NWC", "WIO", "NWC"),
        )
        return y + params["b"]

    return init, apply


def conv2d(in_ch, out_ch, kernel_size, stride=1, padding="SAME"):
    """2D conv. Input x shape: (batch, height, width, channels) -> NHWC."""
    if isinstance(kernel_size, int):
        kh, kw = kernel_size, kernel_size
    else:
        kh, kw = kernel_size
        
    if isinstance(stride, int):
        sh, sw = stride, stride
    else:
        sh, sw = stride

    kernel_shape = (kh, kw, in_ch, out_ch)  # HWIO
    fan_in = kh * kw * in_ch

    def init(key):
        return _conv_init_params(key, kernel_shape, out_ch, fan_in)

    def apply(params, x):
        y = jax.lax.conv_general_dilated(
            x,
            params["w"],
            window_strides=(sh, sw),
            padding=padding,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        return y + params["b"]

    return init, apply


def conv3d(in_ch, out_ch, kernel_size, stride=1, padding="SAME"):
    """3D conv. Input x shape: (batch, depth, height, width, channels) -> NDHWC."""
    if isinstance(kernel_size, int):
        kd, kh, kw = kernel_size, kernel_size, kernel_size
    else:
        kd, kh, kw = kernel_size
    if isinstance(stride, int):
        sd, sh, sw = stride, stride, stride
    else:
        sd, sh, sw = stride

    kernel_shape = (kd, kh, kw, in_ch, out_ch)  # DHWIO
    fan_in = kd * kh * kw * in_ch

    def init(key):
        return _conv_init_params(key, kernel_shape, out_ch, fan_in)

    def apply(params, x):
        y = jax.lax.conv_general_dilated(
            x,
            params["w"],
            window_strides=(sd, sh, sw),
            padding=padding,
            dimension_numbers=("NDHWC", "DHWIO", "NDHWC"),
        )
        return y + params["b"]

    return init, apply