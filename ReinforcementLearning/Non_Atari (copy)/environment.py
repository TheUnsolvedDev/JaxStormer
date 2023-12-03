from import_packages import *


class Env:
    def __init__(self, env='Acrobot-v1', num_envs=8) -> None:
        self.env, self.env_params = gymnax.make(env)
        self.vmap_keys = jax.random.split(key, num_envs)
        self.vmap_reset = jax.vmap(self.env.reset, in_axes=(0, None))
        self.vmap_step = jax.vmap(self.env.step, in_axes=(0, 0, 0, None))

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self):
        obs, state = self.vmap_reset(self.vmap_keys, self.env_params)
        return obs, state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        next_obs, next_state, reward, done, _ = self.vmap_step(
            self.vmap_keys, state, action, self.env_params)
        return next_obs, next_state, reward, done, _


if __name__ == "__main__":
    env = Env()
    print(env.reset())
