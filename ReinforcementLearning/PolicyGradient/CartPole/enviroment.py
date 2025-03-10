import gymnasium as gym
import numpy as np

class Environment(object):
    def __init__(self, env_name = 'CartPole-v1', num_envs = 5):
        if num_envs == 1:
            self.envs = gym.make(env_name,render_mode='human')
            # self.envs = gym.wrappers.Autoreset(self.envs)
        else:
            self.envs = gym.vector.SyncVectorEnv(
                [lambda: gym.make(env_name) for _ in range(num_envs)],
                autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP
            )
        
    def reset(self):
        return self.envs.reset()[0]
    
    def step(self, actions):
        next_state_together, rewards, dones, truncateds, info = self.envs.step(actions)
        return next_state_together, list(map(int, 1-np.logical_or(dones,truncateds))), dones, truncateds, info
        
    
if __name__ == '__main__':
    env = Environment(num_envs=5)
    states_together = env.reset()
    for _ in range(1000):
        actions = np.random.randint(0,2,5)
        print(actions)
        states_together, rewards, dones, truncateds, info = env.step(actions)
        print(rewards*(1-dones), dones, truncateds)
        input()
        
    