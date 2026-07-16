import jax
import jax.numpy as jnp

def mse_loss(pred,target):
    return jnp.mean((pred-target)**2)

def cross_entropy_loss(logits, labels, num_classes):
    onehot=jax.nn.one_hot(labels, num_classes)
    return jnp.mean(-jnp.sum(onehot*jax.nn.log_softmax(logits),axis=1))

def binary_cross_entropy_loss(logits, labels):
    return jnp.mean(
        jnp.maximum(logits, 0) - logits * labels + jnp.log1p(jnp.exp(-jnp.abs(logits)))
    )

    