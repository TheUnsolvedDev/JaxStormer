import gymnasium as gym
import numpy as np

from config import *

class Environment(object):
    def __init__(self, env_name = ENV_NAME, num_envs = 5):
        self.num_envs = num_envs
        if num_envs == 1:
            self.envs = gym.make(env_name,render_mode='human')
            # self.envs = gym.wrappers.Autoreset(self.envs)
        else:
            self.envs = gym.make_vec(env_name, num_envs=num_envs, vectorization_mode='async')
        
    def reset(self):
        return self.envs.reset()[0]
    
    def step(self, actions):
        next_states, rewards, terminated, truncated, infos = self.envs.step(actions)
        return next_states, rewards, terminated, truncated, infos
        
    
if __name__ == '__main__':
    env = Environment(num_envs=1)
    states_together = env.reset()
    for _ in range(1000):
        actions = np.random.randint(0,2)
        print(actions)
        states_together, rewards, dones, truncateds, info = env.step(actions)
        print(rewards)
        print(rewards*(1-dones), dones, truncateds)
        input()
    