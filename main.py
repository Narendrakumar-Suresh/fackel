import jax
import jax.numpy as jnp

from fackel.nn import conv2d, linear, sequential, dropout, flatten
from fackel.activations import relu
from fackel.loss import cross_entropy_loss
from fackel.optimizers import adamw
from fackel.ops import clip_grad_norm
from fackel.schedulers import cosine_annealing_with_warmup


def load_mnist(test_size=10000):
    from sklearn.datasets import fetch_openml

    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    x = mnist.data.astype("float32") / 255.0
    y = mnist.target.astype("int64")

    x = x.reshape(-1, 28, 28, 1)

    x_train, x_test = x[:-test_size], x[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    return (jnp.array(x_train), jnp.array(y_train)), (
        jnp.array(x_test),
        jnp.array(y_test),
    )


def infinite_batches(x, y, batch_size, key):
    n = x.shape[0]
    while True:
        key, shuffle_key = jax.random.split(key)
        perm = jax.random.permutation(shuffle_key, n)
        for i in range(0, n - batch_size + 1, batch_size):
            idx = perm[i : i + batch_size]
            yield x[idx], y[idx]


# 1. Model Definition
init_fn, apply_fn = sequential(
    conv2d(in_ch=1, out_ch=8, kernel_size=3, stride=1, padding="SAME"),
    relu,
    conv2d(in_ch=8, out_ch=16, kernel_size=3, stride=2, padding="SAME"),
    relu,
    flatten,
    linear(in_dim=14 * 14 * 16, out_dim=64),
    relu,
    dropout(0.5),
    linear(in_dim=64, out_dim=10),
)

# 2. Scheduler and Optimizer Initialization
total_steps = 1500
lr_schedule = cosine_annealing_with_warmup(
    max_lr=1e-3, warmup_steps=150, total_steps=total_steps, end_lr=1e-5
)

opt_init, opt_update = adamw(lr=lr_schedule)


@jax.jit
def accuracy(params, model_state, x, y):
    logits, _ = apply_fn(params, model_state, x, is_training=False)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)


@jax.jit
def train_step(params, opt_state, model_state, key, x, y):
    key, step_key = jax.random.split(key)

    def loss_fn(p):
        logits, new_model_state = apply_fn(
            p, model_state, x, rng=step_key, is_training=True
        )
        loss = cross_entropy_loss(logits, y, num_classes=10)
        return loss, new_model_state

    (loss, new_model_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    grads = clip_grad_norm(grads, max_norm=1.0)
    params, opt_state = opt_update(grads, opt_state, params)

    return params, opt_state, new_model_state, key, loss


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_mnist()

    key = jax.random.PRNGKey(0)
    key, init_key, data_key = jax.random.split(key, 3)

    params, model_state = init_fn(init_key)
    opt_state = opt_init(params)

    batch_size = 128
    batch_stream = infinite_batches(x_train, y_train, batch_size, key=data_key)

    print(
        "Starting training with explicit routing, gradient clipping, and LR scheduling..."
    )
    for step in range(total_steps):
        xb, yb = next(batch_stream)

        params, opt_state, model_state, key, loss = train_step(
            params, opt_state, model_state, key, xb, yb
        )

        if step % 100 == 0:
            print(f"step {step:4d}  loss {loss:.4f}")

        if step > 0 and step % 500 == 0:
            acc = accuracy(params, model_state, x_test[:1000], y_test[:1000])
            print(f"--- step {step} eval acc: {acc:.4f}")

    final_acc = accuracy(params, model_state, x_test, y_test)
    print(f"final test accuracy: {final_acc:.4f}")
