import gymnax
import jax
import jax.numpy as jnp
import numpy as np

class Environment:
    def __init__(self, env='Acrobot-v1') -> None:
        self.env, self.env_params = gymnax.make(env)
        self.rng, self.key_reset, self.key_act, self.key_step = jax.random.split(jax.random.PRNGKey(0), 4)
        self.num_actions = self.env.num_actions
        self.observation_shape = self.env.observation_space(self.env_params).sample(self.rng).shape

    def reset(self):
        return self.env.reset(self.key_reset, self.env_params)
    
    def step(self, obs, action):
        return self.env.step(self.key_step, obs, action, self.env_params)


if __name__ == '__main__':
    env = Environment()
    print(env.__dict__)