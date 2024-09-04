import jax
import jax.numpy as jnp

def example_func():
    arr = jnp.arange(10)
    length_mask = jnp.array(5)  # Pretend this is computed dynamically
    # Use dynamic_slice instead of regular slicing
    sliced_arr = jax.lax.dynamic_slice(arr, (0,), (length_mask,))
    return sliced_arr

sliced = jax.jit(example_func)()
print(sliced)