import flax.linen
import jax
import jax.numpy as jnp
import numpy as np
import flax
import config
import distrax

rng = jax.random.PRNGKey(0)



class Q_SA(flax.linen.Module):
    @flax.linen.compact
    def __call__(self, state, action):
        combined_input = jnp.concat([state, action], axis=-1)
        x = flax.linen.Dense(config.HIDDEN)(combined_input)
        x = flax.linen.silu(x)
        x = flax.linen.Dense(config.HIDDEN)(x)
        x = flax.linen.silu(x)
        x = flax.linen.Dense(1)(x)
        return x