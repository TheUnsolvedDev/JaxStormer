from config import *
from model import *
from enviroment import *
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import flax
import tqdm
import os

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


def test(env_name, agent):
    env = Environment(env_name=env_name, num_envs=1)
    total_results = 0

    for _ in range(200):
        states = env.reset()
        actions = agent.act(jax.random.PRNGKey(
            0), agent.policy_state.params, states)
        next_states, rewards, dones, truncateds, _ = env.step(
            np.array(actions))
        states = next_states
        total_results += np.sum(rewards)
        if dones:
            break
    print("Total results", total_results)


def simulate():
    env_name = ENV_NAME
    env = Environment(env_name, num_envs=NUM_ENVS)
    agent = PolicyGradientAgent(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE)
    buffer = TrajectoryBuffer(input_size=INPUT_SHAPE, num_envs=NUM_ENVS, trajectory_size=TRAJECTORY_SIZE)
    key = jax.random.PRNGKey(0)

    states = env.reset()
    losses = []
    for i in tqdm.tqdm(range(1, int(1e+6) + 1)):
        actions = agent.act(key, agent.policy_state.params, states)
        next_states, rewards, dones, truncateds, _ = env.step(
            np.array(actions))
        buffer.init_state = buffer.add(
            buffer.init_state, states, actions, rewards, dones)
        states = next_states
        if i % 200 == 0:
            experiences = buffer.get_buffer(buffer.init_state)
            agent.policy_state, loss = agent.update(
                agent.policy_state, experiences)
            losses.append(loss)
            # print(f'Iteration {i}, Loss: {loss}')
            if i % int(1e+5) == 0:
                test(env_name, agent)
                plt.plot(losses)
                plt.xlabel('loss')
                plt.ylabel('iterations * 200')
                plt.savefig('Losses.png')


if __name__ == '__main__':
    simulate()
