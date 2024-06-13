import flax
import jax
import jax.numpy as jnp
import numpy as np

rng = jax.random.PRNGKey(0)


class UniformReplayBuffer:
    def __init__(self, buffer_size, batch_size, state_shape=(4,)):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.state_shape = state_shape
        self.buffer_state = {
            "states": jnp.empty((buffer_size, state_shape), dtype=jnp.float32),
            "actions": jnp.empty((buffer_size,), dtype=jnp.int32),
            "rewards": jnp.empty((buffer_size,), dtype=jnp.int32),
            "next_states": jnp.empty((buffer_size, state_shape), dtype=jnp.float32),
            "dones": jnp.empty((buffer_size,), dtype=jnp.bool_),
        }

    def add(self, buffer_state, experience, idx):
        state, action, reward, next_state, done = experience
        idx = idx % self.buffer_size

        buffer_state['states'] = buffer_state['states'].at[idx].set(state)
        buffer_state['actions'] = buffer_state['actions'].at[idx].set(action)
        buffer_state['rewards'] = buffer_state['rewards'].at[idx].set(reward)
        buffer_state['next_states'] = buffer_state['next_states'].at[idx].set(
            next_state)
        buffer_state['dones'] = buffer_state['dones'].at[idx].set(done)
        value = (buffer_state, state, action, reward, next_state, done)
        return buffer_state
