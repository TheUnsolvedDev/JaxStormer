import jax
import jax.numpy as np
import numpy as np

gpu = jax.devices('gpu')[0]
cpu = jax.devices('cpu')[0]

class MonteCarloRollout:
    def __init__(self, env, policy, num_envs, num_steps, gamma):
        self.env = env
        self.policy = policy
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.gamma = gamma
        
class TemporalDifferenceRollout:
    def __init__(self, env, policy, num_envs, num_steps, gamma):
        self.env = env
        self.policy = policy
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.gamma = gamma