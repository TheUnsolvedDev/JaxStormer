import jax
import jax.numpy as jnp
import numpy as np
import argparse
import tqdm

from environment import *
from agent import *
from config import *


def main():
    env_name = 'Acrobot-v1'
    env = Environment(env_name)
    agent = Agent(input_shape=env.observation_shape, output_shape=env.num_actions)
    print(agent.policy.tabulate(jax.random.PRNGKey(0), jnp.ones((128, env.observation_shape[0]))))

    state, obs = env.reset()
    
    for i in tqdm.tqdm(range(10**6)):
        state_expanded = jnp.expand_dims(state, axis=0)
        action = agent.act(rng, agent.policy_state.params, state_expanded)
        next_state, next_obs, reward, done, _ = env.step(obs, action)
        experience = (state, action, reward*(1-done))
        agent.transition_buffer.buffer_state = agent.transition_buffer.add(
            agent.transition_buffer.buffer_state,agent.transition_buffer.current_idx, experience)
        agent.transition_buffer.current_idx = (agent.transition_buffer.current_idx+1)%agent.transition_buffer.max_size
        state = next_state
        obs = next_obs

        if done:
            states, actions, rewards = agent.transition_buffer.sample()
            discounted_rewards = agent.get_discounted_rewards(rewards)
            loss, agent.policy_state = agent.update(agent.policy_state, (states, actions, rewards))
            print('loss:', loss, 'reward:', np.sum(rewards))
            agent.transition_buffer.reset()
            state, obs = env.reset()
            


if __name__ == '__main__':
    main()
