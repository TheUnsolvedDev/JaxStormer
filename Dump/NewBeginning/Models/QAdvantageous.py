import flax.linen
import jax
import jax.numpy as jnp
import numpy as np
import flax
import config
import distrax

rng = jax.random.PRNGKey(0)


class Q_adv(flax.linen.Module):
    num_actions: int

    @flax.linen.compact
    def __call__(self, state):
        x = flax.linen.Dense(config.HIDDEN)(state)
        x = flax.linen.silu(x)
        x = flax.linen.Dense(config.HIDDEN)(x)
        x = flax.linen.silu(x)
        value_stream = flax.linen.Dense(1)(x)
        advantage_stream = flax.linen.Dense(self.num_actions)(x)
        q_values = value_stream + \
            (advantage_stream - jnp.mean(advantage_stream, axis=-1, keepdims=True))
        return q_values
