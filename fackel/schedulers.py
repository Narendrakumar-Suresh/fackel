import jax.numpy as jnp


def linear_decay_with_warmup(max_lr, warmup_steps, total_steps, end_lr=0.0):
    def schedule(step):
        warmup_lr = max_lr * (step / jnp.maximum(1, warmup_steps))
        decay_steps = jnp.maximum(1, total_steps - warmup_steps)
        step_after_warmup = jnp.maximum(0, step - warmup_steps)

        decay_ratio = step_after_warmup / decay_steps
        decay_lr = max_lr - decay_ratio * (max_lr - end_lr)
        decay_lr = jnp.maximum(end_lr, decay_lr)

        return jnp.where(step < warmup_steps, warmup_lr, decay_lr)

    return schedule


def cosine_annealing_with_warmup(max_lr, warmup_steps, total_steps, end_lr=0.0):
    def schedule(step):
        warmup_lr = max_lr * (step / jnp.maximum(1, warmup_steps))
        decay_steps = jnp.maximum(1, total_steps - warmup_steps)
        step_after_warmup = jnp.maximum(0, step - warmup_steps)
        step_after_warmup = jnp.minimum(step_after_warmup, decay_steps)
        cosine_ratio = 0.5 * (1.0 + jnp.cos(jnp.pi * step_after_warmup / decay_steps))
        decay_lr = end_lr + (max_lr - end_lr) * cosine_ratio
        return jnp.where(step < warmup_steps, warmup_lr, decay_lr)

    return schedule
