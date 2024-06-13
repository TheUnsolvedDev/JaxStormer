import jax
import functools
import jax.numpy as jnp

class Buffer:
    def __init__(self, state_shape, max_size=500) -> None:
        self.max_size = max_size
        self.buffer_state = {
            'states': jnp.empty((max_size, state_shape), dtype=jnp.float32),
            'actions': jnp.empty((max_size,), dtype=jnp.int32),
            'rewards': jnp.empty((max_size,), dtype=jnp.int32),
            'discounted_rewards': jnp.empty((max_size,), dtype=jnp.int32),
        }
        self.current_idx = 0

    @functools.partial(jax.jit, static_argnums=(0,))
    def add(self, buffer_state, experience):
        state, action, reward = experience
        idx = self.current_idx % self.max_size
        buffer_state['states'] = buffer_state['states'].at[idx].set(state)
        buffer_state['actions'] = buffer_state['actions'].at[idx].set(action)
        buffer_state['rewards'] = buffer_state['rewards'].at[idx].set(reward)
        self.current_idx += 1
        return buffer_state
    
    
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def get_trajectory(self):