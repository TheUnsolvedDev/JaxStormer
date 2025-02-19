import flax.linen
import jax
import jax.numpy as jnp
import numpy as np
import flax

from config import *

class QNetwork(flax.linen.Module):
    output_size: int
    @flax.linen.compact
    def __call__(self, x):
        x = flax.linen.Conv(features=32, kernel_size=(8, 8), padding='VALID')(x)
        x = flax.linen.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = flax.linen.gelu(x)
        x = flax.linen.Conv(features=64, kernel_size=(4, 4), padding='VALID')(x)
        x = flax.linen.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = flax.linen.gelu(x)
        x = flax.linen.Conv(features=64, kernel_size=(3, 3), padding='VALID')(x)
        x = flax.linen.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = flax.linen.gelu(x)
        x = x.reshape((x.shape[0], -1))  # flatten
        x = flax.linen.Dense(features=32)(x)
        x = flax.linen.gelu(x)
        x = flax.linen.Dense(features=self.output_size)(x)
        return x
    
if __name__ == '__main__':
    model = QNetwork(output_size=4)
    x = jnp.ones((1, 84, 84, 4))
    print(model.tabulate(rng, x))