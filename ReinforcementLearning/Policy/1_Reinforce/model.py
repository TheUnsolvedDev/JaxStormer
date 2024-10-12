import flax.linen
import jax
import jax.numpy as jnp
import numpy as np
import flax

class PolicyNetwork(flax.linen.Module):
    action_dim: int

    @flax.linen.compact
    def __call__(self, x):
        x = flax.linen.Dense(16)(x)
        x = flax.linen.gelu(x)
        x = flax.linen.Dense(16)(x)
        x = flax.linen.gelu(x)
        x = flax.linen.Dense(self.action_dim)(x)
        probs = flax.linen.softmax(x)
        return probs
    
if __name__ == '__main__':
    model = PolicyNetwork(action_dim=2)
    print(model.tabulate(jax.random.PRNGKey(0), jnp.ones((128, 5))))