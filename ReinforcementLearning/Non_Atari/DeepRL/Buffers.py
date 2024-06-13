import jax
import numpy as np
import jax.numpy as jnp
import functools


class ReplayBuffer:
    def __init__(self, buffer_size=1024, batch_size=64, state_shape=4, num_envs=16):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer_state = {
            "states": jnp.empty((buffer_size*num_envs, state_shape), dtype=jnp.float32),
            "actions": jnp.empty((buffer_size*num_envs,), dtype=jnp.int32),
            "rewards": jnp.empty((buffer_size*num_envs,), dtype=jnp.int32),
            "next_states": jnp.empty((buffer_size*num_envs, state_shape), dtype=jnp.float32),
            "dones": jnp.empty((buffer_size*num_envs,), dtype=jnp.bool_),
        }

        self.current_buffer_size = 0
        self.current_buffer_index = 0
        self.num_envs = num_envs
        self.state_shape = state_shape

    @functools.partial(jax.jit, static_argnums=(0,))
    def add(self, buffer_state, experience):
        state, action, reward, next_state, done = experience
        idx = (self.current_buffer_index + self.num_envs) % self.buffer_size
        val_init = (buffer_state, state, action, reward, next_state, done)

        def fill(i, val):
            state, action, reward, next_state, done = val
            idx = (i + self.num_envs) % self.buffer_size
            return (self.buffer_state, state, action, reward, next_state, done)

        buffer_state = jax.lax.fori_loop(0, self.num_envs, fill, val_init)
        self.current_buffer_index = (
            self.current_buffer_index + self.num_envs) % self.buffer_size
        self.current_buffer_size = min(
            self.current_buffer_size + self.num_envs, self.buffer_size)
        return buffer_state

    @functools.partial(jax.jit, static_argnums=(0,))
    def sample(self, key, buffer_state, current_buffer_size):

        @functools.partial(jax.vmap, in_axes=(0, None))
        def sample_batch(indexes, buffer):
            return jax.tree_map(lambda x: x[indexes], buffer)

        key, subkey = jax.random.split(key)
        indexes = jax.random.randint(subkey, shape=(self.batch_size,),
                                     minval=0,
                                     maxval=current_buffer_size+self.num_envs,
                                     dtype=jnp.int32)
        return sample_batch(indexes, buffer_state)
