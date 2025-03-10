import jax
import jax.numpy as jnp
import flax
import tqdm

from enviroment import *
from model import *
from config import *

def simulate():
    env_name = 'CartPole-v1'
    env = Environment(env_name, num_envs=10)
    agent = PolicyGradientAgent(input_shape=5,output_shape=2)
    key = jax.random.PRNGKey(0)
    
    states = env.reset()
    for i in tqdm.tqdm(range(1,1e+6 + 1)):
        actions = agent.act(rng, agent.policy_state.params,states)
        next_states, rewards, dones, truncateds, _ = env.step(actions)
        

if __name__ == '__main__':
    simulate()