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

ALPHA = 0.001
GAMMA = 0.99


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



class ActorNetwork(flax.linen.Module):
    action_dim: int

    @flax.linen.compact
    def __call__(self, x):
        x = flax.linen.Dense(16)(x)
        x = flax.linen.gelu(x)
        x = flax.linen.Dense(16)(x)
        x = flax.linen.gelu(x)
        x = flax.linen.Dense(self.action_dim)(x)
        x = flax.linen.softmax(x)
        return x


class CriticNetwork(flax.linen.Module):

    @flax.linen.compact
    def __call__(self, x):
        x = flax.linen.Dense(16)(x)
        x = flax.linen.gelu(x)
        x = flax.linen.Dense(16)(x)
        x = flax.linen.gelu(x)
        x = flax.linen.Dense(1)(x)
        return x


class ActorCritic:
    def __init__(self, env, num_actions, observation_shape, seed=0):
        self.seed = seed
        self.rng = jax.random.PRNGKey(seed)
        self.num_actions = num_actions
        self.observation_shape = observation_shape
        self.env = env

        self.actor = ActorNetwork(num_actions)
        self.actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=self.actor.init(self.rng, jnp.ones(observation_shape)),
            tx=optax.adam(learning_rate=ALPHA/3),
        )
        self.actor.apply = jax.jit(self.actor.apply)

        self.critic = CriticNetwork()
        self.critic_state = TrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic.init(self.rng, jnp.ones(observation_shape)),
            tx=optax.adam(learning_rate=ALPHA),
        )
        self.critic.apply = jax.jit(self.critic.apply)

    @functools.partial(jax.jit, static_argnums=(0,))
    def policy_prob(self, q_state, state):
        probs = self.actor.apply(q_state.params, state)
        return probs

    def sample(self, state):
        probs = self.policy_prob(self.actor_state, state)[0]
        return np.random.choice(self.num_actions, p=np.array(probs))

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(self, actor_state, critic_state, states, actions, next_states, rewards, dones, gamma_t):

        @jax.jit
        def mse_loss(params):
            target = rewards + GAMMA * \
                self.critic.apply(params, next_states)*(1.0-dones)
            pred = self.critic.apply(params, states)
            target = jax.lax.stop_gradient(target)
            delta = target - pred
            return 0.5*jnp.mean(jnp.square(delta)), delta

        @jax.jit
        def log_prob_loss(params):
            probs = self.actor.apply(params, states)
            probs = jnp.clip(probs, a_min=1e-7, a_max=10)
            log_probs = jnp.log(probs)
            actions_new = jax.nn.one_hot(actions, num_classes=self.num_actions)
            prob_reduce = -jnp.sum(log_probs*actions_new, axis=1)
            td = jax.lax.stop_gradient(delta)
            td -= td.mean()
            td /= td.std() + 1e-6
            loss = jnp.mean(prob_reduce*gamma_t*delta)
            entropy = jnp.mean(probs*log_probs)
            return loss + 0.01*entropy

        (loss_critic, delta), grads_critic = jax.value_and_grad(
            mse_loss, has_aux=True)(critic_state.params)

        loss_actor, grads_actor = jax.value_and_grad(
            log_prob_loss)(actor_state.params)

        critic_state = critic_state.apply_gradients(
            grads=grads_critic)
        actor_state = actor_state.apply_gradients(
            grads=grads_actor)
        return loss_actor+loss_critic, actor_state, critic_state

    def train_single_step(self, reward_shape=0):
        state = self.env.reset(seed=self.seed)[0]
        total_reward = 0
        total_loss = 0
        for _ in range(500):
            action = self.sample(np.expand_dims(state, axis=0))
            next_state, reward, done, truncated, info = self.env.step(action)
            # just a basic reward shaping to keep the process going on
            reward = reward_shape if done or truncated else reward
            total_reward += reward

            episode_state = jnp.array([state])
            episode_next_state = jnp.array([next_state])
            episode_reward = jnp.array([reward])
            episode_action = jnp.array([action])
            episode_done = jnp.array([done or truncated])
            gamma_t = GAMMA ** _

            loss, self.actor_state, self.critic_state = self.update(
                self.actor_state, self.critic_state, episode_state, episode_action, episode_next_state, episode_reward, episode_done, gamma_t)
            total_loss += loss

            state = next_state
            if done or truncated:
                break
        jax.clear_caches()
        gc.collect()
        return total_loss, total_reward


class Simulation:
    def __init__(self, env_name, algorithm, reward_shape=0) -> None:
        self.env_name = env_name
        self.algorithm = algorithm
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
    cartpole_actor_critic = Simulation(
        'CartPole-v1', algorithm=ActorCritic, reward_shape=-3)
    cartpole_actor_critic.train()
    rewards_cartpole_actor_critic = cartpole_actor_critic.rewards
    mean_rcb = np.mean(rewards_cartpole_actor_critic, axis=0)
    std_rcb = np.std(rewards_cartpole_actor_critic, axis=0)
    plot_data(mean_rcb, std_rcb, name='Cartpole AC BiModel')

    acrobot_actor_critic = Simulation(
        'Acrobot-v1', algorithm=ActorCritic, reward_shape=3)
    acrobot_actor_critic.train()
    rewards_acrobot_actor_critic = acrobot_actor_critic.rewards
    mean_rab = np.mean(rewards_acrobot_actor_critic, axis=0)
    std_rab = np.std(rewards_acrobot_actor_critic, axis=0)
    plot_data(mean_rab, std_rab, name='Acrobot AC BiModel')
