import jax
import jax.numpy as jnp

rng = jax.random.PRNGKey(0)
ALPHA = 1e-4
GAMMA = 0.99