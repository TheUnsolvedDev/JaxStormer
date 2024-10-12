import flax.linen
import jax
import jax.numpy as jnp
import numpy as np
import flax
import config
import distrax

rng = jax.random.PRNGKey(0)


class Pi_cont(flax.linen.Module):
    @flax.linen.compact
    def __call__(self, state, seed=jax.random.PRNGKey(0)):
        x = flax.linen.Dense(config.HIDDEN)(state)
        x = flax.linen.silu(x)
        x = flax.linen.Dense(config.HIDDEN)(x)
        x = flax.linen.silu(x)
        mu = flax.linen.Dense(1)(x)
        sigma = flax.linen.Dense(1)(x)
        disribution = distrax.Independent(distrax.Normal(
            loc=mu, scale=sigma), reinterpreted_batch_ndims=1)
        sample = disribution.sample(seed=seed)
        return sample,(mu,sigma)