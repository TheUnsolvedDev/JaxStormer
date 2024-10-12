import jax
import jax.numpy as jnp
import numpy as np
import functools


class TransitionBuffer:
    def __init__(self, state_shape, max_size=1000):
        self.state_shape = state_shape
        self.max_size = max_size
        self.reset()

    def reset(self):
        self.buffer_state = {
            'states': jnp.empty((self.max_size, self.state_shape[0]), dtype=jnp.float32),
            'actions': jnp.empty((self.max_size,), dtype=jnp.float32),
            'rewards': jnp.empty((self.max_size,), dtype=jnp.float32),
        }
        self.current_idx = 0

    @functools.partial(jax.jit, static_argnums=(0,))
    def add(self, buffer_state, idx, experience):
        state, action, reward = experience
        buffer_state['states'] = buffer_state['states'].at[idx].set(state)
        buffer_state['actions'] = buffer_state['actions'].at[idx].set(action)
        buffer_state['rewards'] = buffer_state['rewards'].at[idx].set(reward)
        return buffer_state

    @functools.partial(jax.jit, static_argnums=(0,))
    def sample(self):
        experiences = (
            self.buffer_state['states'][:self.current_idx],
            self.buffer_state['actions'][:self.current_idx],
            self.buffer_state['rewards'][:self.current_idx],
        )
        return experiences
