"""
Small CNN built entirely from fackel (conv2d + linear + relu + adam),
trained on MNIST.

Needs: pip install scikit-learn  (just for the dataset loader, not for training)
First run downloads MNIST (~55MB) via sklearn and caches it locally under
~/scikit_learn_data, so it only fetches once.
"""

import jax
import jax.numpy as jnp
import numpy as np

from fackel.nn import conv2d, linear
from fackel.activations import relu
from fackel.optimizers import adam


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
def load_mnist(test_size=10000):
    """Loads MNIST via sklearn's fetch_openml. Swap this out for whatever
    loader you already have -- the rest of the file only cares that x is
    (N, 28, 28, 1) float32 in [0,1] and y is (N,) int labels."""
    from sklearn.datasets import fetch_openml

    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    x = mnist.data.astype("float32") / 255.0          # (70000, 784)
    y = mnist.target.astype("int64")                   # (70000,)

    x = x.reshape(-1, 28, 28, 1)                        # (70000, 28, 28, 1)

    x_train, x_test = x[:-test_size], x[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    return (jnp.array(x_train), jnp.array(y_train)), (jnp.array(x_test), jnp.array(y_test))


# ---------------------------------------------------------------------------
# 2. Model: conv -> relu -> conv -> relu -> flatten -> linear -> linear
# ---------------------------------------------------------------------------
init_c1, apply_c1 = conv2d(in_ch=1, out_ch=8, kernel_size=3, stride=1, padding="SAME")
init_c2, apply_c2 = conv2d(in_ch=8, out_ch=16, kernel_size=3, stride=2, padding="SAME")
# after stride-2 conv on 28x28 input -> 14x14 spatial, 16 channels -> flatten = 14*14*16
init_l1, apply_l1 = linear(in_dim=14 * 14 * 16, out_dim=64)
init_l2, apply_l2 = linear(in_dim=64, out_dim=10)


def model_init(key):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return {
        "c1": init_c1(k1),
        "c2": init_c2(k2),
        "l1": init_l1(k3),
        "l2": init_l2(k4),
    }


def forward(params, x):
    x = apply_c1(params["c1"], x)
    x = relu(x)
    x = apply_c2(params["c2"], x)
    x = relu(x)
    x = x.reshape(x.shape[0], -1)          # flatten (N, H, W, C) -> (N, H*W*C)
    x = apply_l1(params["l1"], x)
    x = relu(x)
    x = apply_l2(params["l2"], x)          # logits (N, 10)
    return x


# ---------------------------------------------------------------------------
# 3. Loss + optimizer + train step
# ---------------------------------------------------------------------------
def loss_fn(params, x, y):
    logits = forward(params, x)
    onehot = jax.nn.one_hot(y, num_classes=10)
    return jnp.mean(-jnp.sum(onehot * jax.nn.log_softmax(logits), axis=-1))


def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)


opt_init, opt_update = adam(lr=1e-3)


@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    params, opt_state = opt_update(grads, opt_state, params)
    return params, opt_state, loss


# ---------------------------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------------------------
def data_iterator(x, y, batch_size, key):
    n = x.shape[0]
    perm = jax.random.permutation(key, n)
    for i in range(0, n - batch_size + 1, batch_size):
        idx = perm[i:i + batch_size]
        yield x[idx], y[idx]


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_mnist()

    key = jax.random.PRNGKey(0)
    key, init_key = jax.random.split(key)
    params = model_init(init_key)
    opt_state = opt_init(params)

    batch_size = 128
    epochs = 3

    for epoch in range(epochs):
        key, shuffle_key = jax.random.split(key)
        for step, (xb, yb) in enumerate(data_iterator(x_train, y_train, batch_size, shuffle_key)):
            params, opt_state, loss = train_step(params, opt_state, xb, yb)
            if step % 100 == 0:
                print(f"epoch {epoch}  step {step:4d}  loss {loss:.4f}")

        test_acc = accuracy(params, x_test[:1000], y_test[:1000])
        print(f"epoch {epoch} done -- test acc (first 1000): {test_acc:.4f}")

    final_acc = accuracy(params, x_test, y_test)
    print(f"final test accuracy: {final_acc:.4f}")