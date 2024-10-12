import flax.linen
import jax
import jax.numpy as jnp
import numpy as np
import flax
import config

rng = jax.random.PRNGKey(0)


class Q(flax.linen.Module):
    num_actions: int

    @flax.linen.compact
    def __call__(self, state):
        x = flax.linen.Dense(config.HIDDEN)(state)
        x = flax.linen.silu(x)
        x = flax.linen.Dense(config.HIDDEN)(x)
        x = flax.linen.silu(x)
        x = flax.linen.Dense(self.num_actions)(x)
        return x
