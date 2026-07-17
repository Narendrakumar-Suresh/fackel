import os
import sys
import urllib.request
import jax
import jax.numpy as jnp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fackel.nn import linear, conv1d, rglru, embedding
from fackel.loss import cross_entropy_loss
from fackel.optimizers import adamw
from fackel.io import save_checkpoint, load_checkpoint


# --- Core Primitive Additions ---
def rmsnorm(dim, eps=1e-6):
    def init(key):
        return {"scale": jnp.ones(dim)}, {}

    def apply(params, state, x, **kwargs):
        rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + eps)
        return (x / rms) * params["scale"], state

    return init, apply


# --- Diagram Architecture Definitions ---
def gated_mlp(d_model):
    left_init, left_apply = linear(d_model, d_model)
    right_init, right_apply = linear(d_model, d_model)
    out_init, out_apply = linear(d_model, d_model)

    def init(key):
        k1, k2, k3 = jax.random.split(key, 3)
        p_l, s_l = left_init(k1)
        p_r, s_r = right_init(k2)
        p_o, s_o = out_init(k3)
        return {"left": p_l, "right": p_r, "out": p_o}, {
            "left": s_l,
            "right": s_r,
            "out": s_o,
        }

    def apply(params, state, x, **kwargs):
        out_l, s_l = left_apply(params["left"], state["left"], x, **kwargs)
        out_r, s_r = right_apply(params["right"], state["right"], x, **kwargs)
        fused = jax.nn.gelu(out_l) * out_r
        out, s_o = out_apply(params["out"], state["out"], fused, **kwargs)
        return out, {"left": s_l, "right": s_r, "out": s_o}

    return init, apply


def recurrent_block(d_model):
    left_init, left_apply = linear(d_model, d_model)
    right_init, right_apply = linear(d_model, d_model)
    conv_init, conv_apply = conv1d(
        in_ch=d_model, out_ch=d_model, kernel_size=4, padding="SAME"
    )
    rglru_init, rglru_apply = rglru(in_dim=d_model, out_dim=d_model)
    out_init, out_apply = linear(d_model, d_model)

    def init(key):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        p_l, s_l = left_init(k1)
        p_r, s_r = right_init(k2)
        p_c, s_c = conv_init(k3)
        p_rg, s_rg = rglru_init(k4)
        p_o, s_o = out_init(k5)
        return {"left": p_l, "right": p_r, "conv": p_c, "rglru": p_rg, "out": p_o}, {
            "left": s_l,
            "right": s_r,
            "conv": s_c,
            "rglru": s_rg,
            "out": s_o,
        }

    def apply(params, state, x, **kwargs):
        out_l, s_l = left_apply(params["left"], state["left"], x, **kwargs)
        out_r, s_r = right_apply(params["right"], state["right"], x, **kwargs)

        out_c, s_c = conv_apply(params["conv"], state["conv"], out_r, **kwargs)
        out_rg, s_rg = rglru_apply(params["rglru"], state["rglru"], out_c, **kwargs)

        fused = jax.nn.gelu(out_l) * out_rg
        out, s_o = out_apply(params["out"], state["out"], fused, **kwargs)
        return out, {"left": s_l, "right": s_r, "conv": s_c, "rglru": s_rg, "out": s_o}

    return init, apply


def residual_block(d_model):
    norm1_init, norm1_apply = rmsnorm(d_model)
    mix_init, mix_apply = recurrent_block(d_model)
    norm2_init, norm2_apply = rmsnorm(d_model)
    mlp_init, mlp_apply = gated_mlp(d_model)

    def init(key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        p_n1, s_n1 = norm1_init(k1)
        p_mix, s_mix = mix_init(k2)
        p_n2, s_n2 = norm2_init(k3)
        p_mlp, s_mlp = mlp_init(k4)
        return {"norm1": p_n1, "mix": p_mix, "norm2": p_n2, "mlp": p_mlp}, {
            "norm1": s_n1,
            "mix": s_mix,
            "norm2": s_n2,
            "mlp": s_mlp,
        }

    def apply(params, state, x, **kwargs):
        res1 = x
        n1, s_n1 = norm1_apply(params["norm1"], state["norm1"], x, **kwargs)
        mix_out, s_mix = mix_apply(params["mix"], state["mix"], n1, **kwargs)
        x = res1 + mix_out

        res2 = x
        n2, s_n2 = norm2_apply(params["norm2"], state["norm2"], x, **kwargs)
        mlp_out, s_mlp = mlp_apply(params["mlp"], state["mlp"], n2, **kwargs)
        x = res2 + mlp_out
        return x, {"norm1": s_n1, "mix": s_mix, "norm2": s_n2, "mlp": s_mlp}

    return init, apply


def hawk_model(vocab_size, d_model, num_layers=2):
    embed_init, embed_apply = embedding(vocab_size, d_model)
    layers = [residual_block(d_model) for _ in range(num_layers)]
    head_init, head_apply = linear(d_model, vocab_size)

    def init(key):
        keys = jax.random.split(key, 2 + num_layers)
        p_emb, s_emb = embed_init(keys[0])
        p_layers, s_layers = zip(*[lyr[0](keys[1 + i]) for i, lyr in enumerate(layers)])
        p_head, s_head = head_init(keys[-1])
        return {"emb": p_emb, "layers": list(p_layers), "head": p_head}, {
            "emb": s_emb,
            "layers": list(s_layers),
            "head": s_head,
        }

    def apply(params, state, x, **kwargs):
        x, s_emb = embed_apply(params["emb"], state["emb"], x, **kwargs)
        new_s_layers = []
        for i, lyr in enumerate(layers):
            x, s_l = lyr[1](params["layers"][i], state["layers"][i], x, **kwargs)
            new_s_layers.append(s_l)
        logits, s_head = head_apply(params["head"], state["head"], x, **kwargs)
        return logits, {"emb": s_emb, "layers": new_s_layers, "head": s_head}

    return init, apply


# --- Data & Operations Pipeline ---
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_FILE = "tinyshakespeare.txt"

if not os.path.exists(DATA_FILE):
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)

with open(DATA_FILE, "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}
data = jnp.array([char_to_int[ch] for ch in text], dtype=jnp.int32)


def get_batches(data, batch_size, seq_len):
    n = len(data) - seq_len - 1
    while True:
        idx = jax.random.randint(jax.random.PRNGKey(42), (batch_size,), 0, n)
        x_batches = jnp.stack([data[i : i + seq_len] for i in idx])
        y_batches = jnp.stack([data[i + 1 : i + seq_len + 1] for i in idx])
        yield x_batches, y_batches


d_model = 128
init_fn, apply_fn = hawk_model(vocab_size, d_model, num_layers=2)
total_steps = 1000
opt_init, opt_update = adamw(lr=1e-3, weight_decay=0.01)


@jax.jit
def train_step(params, opt_state, model_state, x, y):
    def loss_fn(p):
        logits, new_model_state = apply_fn(p, model_state, x, is_training=True)
        loss = cross_entropy_loss(
            logits.reshape(-1, vocab_size), y.reshape(-1), num_classes=vocab_size
        )
        return loss, new_model_state

    (loss, new_model_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    params, opt_state = opt_update(grads, opt_state, params)
    return params, opt_state, new_model_state, loss


def generate_text(params, model_state, start_str, length=200, temperature=0.8):
    input_ids = [char_to_int[ch] for ch in start_str]
    generated = start_str
    for _ in range(length):
        x = jnp.array([input_ids], dtype=jnp.int32)
        logits, _ = apply_fn(params, model_state, x, is_training=False)
        next_char_logits = logits[0, -1, :] / temperature
        probs = jax.nn.softmax(next_char_logits)
        next_id = int(
            jax.random.choice(jax.random.PRNGKey(len(generated)), len(probs), p=probs)
        )
        generated += int_to_char[next_id]
        input_ids.append(next_id)
        if len(input_ids) > 64:
            input_ids.pop(0)
    return generated


if __name__ == "__main__":
    batch_size = 64
    seq_len = 64
    batch_stream = get_batches(data, batch_size, seq_len)

    key = jax.random.PRNGKey(1337)
    key, init_key = jax.random.split(key)
    params, model_state = init_fn(init_key)
    opt_state = opt_init(params)

    for step in range(total_steps):
        xb, yb = next(batch_stream)
        params, opt_state, model_state, loss = train_step(
            params, opt_state, model_state, xb, yb
        )
        if step % 100 == 0:
            print(f"Step {step:4d} | Loss: {loss:.4f}")

    ckpt_path = "hawk_model.safetensors"
    save_checkpoint(params, ckpt_path)
    params = load_checkpoint(ckpt_path, params)

    print("\n--- Generating Sample Text ---")
    print(generate_text(params, model_state, start_str="ROMEO: ", length=250))
