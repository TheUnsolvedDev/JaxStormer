import jax
import jax.numpy as jnp
import flax
import numpy as np

class NN(flax.linen.Module):
    output_size: int
    
    @flax.linen.compact
    def __call__(self, x):
        x = flax.linen.Dense(features=64)(x)
        x = jax.nn.relu(x)
        x = flax.linen.Dense(features=64)(x)
        x = jax.nn.relu(x)
        x = flax.linen.Dense(features=self.output_size)(x)
        return x
    
if __name__ == '__main__':
    rng = jax.random.PRNGKey(0)
    output_size = 10
    model = NN(output_size)
    
    x = jax.random.normal(rng, (1, 28*28))
    print(model.tabulate(rngs=rng, x=x, compute_flops=True,compute_vjp_flops=True))