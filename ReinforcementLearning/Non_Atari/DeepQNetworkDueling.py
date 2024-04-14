import jax
import numpy as np
import jax.numpy as jnp
import gymnasium as gym
import flax
from flax.training.train_state import TrainState
import optax
import functools
import matplotlib.pyplot as plt
import tqdm
import gc
import argparse

ALPHA = 0.0001
GAMMA = 0.99
BATCH_SIZE = 64
CAPACITY = 20000
UPDATE_EVERY = 20


def smooth_rewards(rewards, window_size=10):
    smoothed_rewards = np.zeros_like(rewards)
    for i in range(len(rewards)):
        window_start = max(0, i - window_size // 2)
        window_end = min(len(rewards), i + window_size // 2 + 1)
        smoothed_rewards[i] = np.mean(rewards[window_start:window_end])
    return smoothed_rewards


def plot_data(mean, std, name):
    x = range(len(mean))

    plt.plot(x, mean, color='blue', label='Mean')
    plt.plot(x, smooth_rewards(mean), color='orange', label='smoothed')
    plt.fill_between(x, mean - std, mean + std, color='blue',
                     alpha=0.3, label='Mean ± Std')

    plt.xlabel('Steps')
    plt.ylabel('Rewards')
    plt.title(name)
    plt.legend()
    plt.grid(True)
    plt.savefig(name.replace(' ', '_')+'.png')
    # plt.show(block = False)
    plt.close()


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class ValueNetwork(flax.linen.Module):
    action_dim: int
    action_type: str

    @flax.linen.compact
    def __call__(self, x: jnp.ndarray):
        x = flax.linen.Dense(16)(x)
        x = flax.linen.selu(x)
        x = flax.linen.Dense(16)(x)
        x = flax.linen.selu(x)
        value_stream = flax.linen.Dense(1)(x)
        advantage_stream = flax.linen.Dense(self.action_dim)(x)
        if self.action_type == 'max':
            q_values = value_stream + \
                (advantage_stream - jnp.max(advantage_stream, axis=-1, keepdims=True))
        elif self.action_type == 'mean':
            q_values = value_stream + \
                (advantage_stream - jnp.mean(advantage_stream, axis=-1, keepdims=True))
        else:
            print('Choose option wisely', self.action_type)
            exit(0)
        return q_values


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return np.random.choice(len(self.buffer), size=(batch_size,))

    def get_batch(self, indices):
        states, actions, rewards, next_states, dones = zip(
            *[self.buffer[i] for i in indices])
        states = jnp.array(states)
        actions = jnp.array(actions)
        rewards = jnp.array(rewards)
        next_states = jnp.array(next_states)
        dones = jnp.array(dones)
        return states, actions, rewards, next_states, dones


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


class DuelingDQN:
    def __init__(self, env, num_actions, observation_shape, seed=0, type='mean') -> None:
        self.seed = seed
        self.rng = jax.random.PRNGKey(seed)
        self.num_actions = num_actions
        self.observation_shape = observation_shape
        self.env = env

        self.value = ValueNetwork(action_dim=num_actions, action_type=type)
        self.value_state = TrainState.create(
            apply_fn=self.value.apply,
            params=self.value.init(self.rng, jnp.ones(observation_shape)),
            target_params=self.value.init(
                self.rng, jnp.ones(observation_shape)),
            tx=optax.adam(learning_rate=ALPHA)
        )
        self.value.apply = jax.jit(self.value.apply)
        self.value_state = self.value_state.replace(target_params=optax.incremental_update(
            self.value_state.params, self.value_state.target_params, 0.9))
        self.replay_buffer = ReplayBuffer(CAPACITY)
        self.counter = 1
        self.updates = 1

    def sample(self, state, epsilon=0.1):
        if np.random.uniform() < epsilon:
            return np.random.randint(0, self.num_actions)
        q_values = self.value.apply(self.value_state.params, state)
        action = np.array(q_values).argmax(axis=-1)[0]
        return action

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(self, value_state, states, actions, rewards, next_states,  dones):
        value_next_target = self.value.apply(
            value_state.target_params, next_states)
        value_next_target = jnp.max(value_next_target, axis=-1)
        next_q_value = (rewards + (1 - dones) * GAMMA * value_next_target)

        @jax.jit
        def mse_loss(params):
            value_pred = self.value.apply(params, states)
            value_pred = jnp.sum(
                value_pred*jax.nn.one_hot(actions, num_classes=self.num_actions), axis=-1)
            return ((jax.lax.stop_gradient(next_q_value) - value_pred) ** 2).mean()

        loss_value, grads = jax.value_and_grad(
            mse_loss)(value_state.params)
        value_state = value_state.apply_gradients(grads=grads)
        return loss_value, value_state

    def train_single_step(self,reward_shape = 0):
        state = self.env.reset(seed=self.seed)[0]
        key = self.rng
        epsilon = linear_schedule(
            start_e=1, end_e=0.05, duration=500, t=0 if self.counter <= 50 else self.counter-50)
        episode_loss, episode_rewards = 0, 0
        for _ in range(500):
            action = self.sample(np.expand_dims(state, axis=0), epsilon)
            next_state, reward, done, truncated, info = self.env.step(action)
            reward = reward_shape if done or truncated else reward
            self.replay_buffer.push(
                [state, action, reward, next_state, done or truncated])
            state = next_state
            episode_rewards += reward

            if truncated or done:
                break

            if len(self.replay_buffer.buffer) > 128:
                indices = self.replay_buffer.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = self.replay_buffer.get_batch(
                    indices)
                loss_values, self.value_state = self.update(self.value_state,
                                                            states, actions, rewards, next_states,  dones)
                if self.updates % UPDATE_EVERY == 0:
                    self.value_state = self.value_state.replace(target_params=optax.incremental_update(
                        self.value_state.params, self.value_state.target_params, 0.9))
                episode_loss += loss_values
                self.updates += 1
        gc.collect()
        jax.clear_caches()
        self.counter += 1
        return episode_loss, episode_rewards


class Simulation:
    def __init__(self, env_name, algorithm, type, reward_shape=0) -> None:
        self.env_name = env_name
        self.algorithm = algorithm
        self.type = type
        self.reward_shape = reward_shape
        self.env = gym.make(self.env_name)
        self.num_actions = self.env.action_space.n
        self.observation_shape = self.env.observation_space.shape

    def train(self, num_avg=3, episodes=1000):
        self.losses, self.rewards = np.zeros(
            (num_avg, episodes)), np.zeros((num_avg, episodes))

        for seed in range(num_avg):
            self.algo = self.algorithm(
                self.env, self.num_actions, self.observation_shape, seed=seed)
            pbar = tqdm.tqdm(range(1, episodes+1))
            for ep in pbar:
                loss, reward = self.algo.train_single_step(self.reward_shape)
                pbar.set_description(f'Loss: {loss} Rewards: {reward}')
                self.losses[seed][ep-1] = loss
                self.rewards[seed][ep-1] = reward


if __name__ == '__main__':

    cartpole_dqn_max = Simulation(
        'CartPole-v1', algorithm=DuelingDQN, type='mean', reward_shape=-3)
    cartpole_dqn_max.train()
    rewards_cartpole_dqn_max = cartpole_dqn_max.rewards
    mean_rcb = np.mean(rewards_cartpole_dqn_max, axis=0)
    std_rcb = np.std(rewards_cartpole_dqn_max, axis=0)
    plot_data(mean_rcb, std_rcb, name='Cartpole DQN Dueling Mean')

    acrobot_dqn_max = Simulation(
        'Acrobot-v1', algorithm=DuelingDQN, type='mean', reward_shape=-3)
    acrobot_dqn_max.train()
    rewards_acrobot_dqn_max = acrobot_dqn_max.rewards
    mean_rab = np.mean(rewards_acrobot_dqn_max, axis=0)
    std_rab = np.std(rewards_acrobot_dqn_max, axis=0)
    plot_data(mean_rab, std_rab, name='Acrobot DQN Dueling Mean')

    cartpole_dqn_max = Simulation(
        'CartPole-v1', algorithm=DuelingDQN, type='max', reward_shape=3)
    cartpole_dqn_max.train()
    rewards_cartpole_dqn_max = cartpole_dqn_max.rewards
    mean_rcb = np.mean(rewards_cartpole_dqn_max, axis=0)
    std_rcb = np.std(rewards_cartpole_dqn_max, axis=0)
    plot_data(mean_rcb, std_rcb, name='Cartpole DQN Dueling Max')

    acrobot_dqn_max = Simulation(
        'Acrobot-v1', algorithm=DuelingDQN, type='max', reward_shape=3)
    acrobot_dqn_max.train()
    rewards_acrobot_dqn_max = acrobot_dqn_max.rewards
    mean_rab = np.mean(rewards_acrobot_dqn_max, axis=0)
    std_rab = np.std(rewards_acrobot_dqn_max, axis=0)
    plot_data(mean_rab, std_rab, name='Acrobot DQN Dueling Max')
