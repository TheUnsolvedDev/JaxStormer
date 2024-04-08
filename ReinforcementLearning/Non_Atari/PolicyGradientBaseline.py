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


class PolicyNetwork(flax.linen.Module):
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


class BaselineNetwork(flax.linen.Module):

    @flax.linen.compact
    def __call__(self, x):
        x = flax.linen.Dense(16)(x)
        x = flax.linen.gelu(x)
        x = flax.linen.Dense(16)(x)
        x = flax.linen.gelu(x)
        x = flax.linen.Dense(1)(x)
        return x


class MC_Baseline:
    def __init__(self, env, num_actions, observation_shape, seed=0):
        self.seed = seed
        self.rng = jax.random.PRNGKey(seed)
        self.num_actions = num_actions
        self.observation_shape = observation_shape
        self.env = env

        self.policy = PolicyNetwork(num_actions)
        self.policy_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(self.rng, jnp.ones(observation_shape)),
            tx=optax.adam(learning_rate=ALPHA),
        )
        self.policy.apply = jax.jit(self.policy.apply)

        self.baseline = BaselineNetwork()
        self.baseline_state = TrainState.create(
            apply_fn=self.baseline.apply,
            params=self.baseline.init(self.rng, jnp.ones(observation_shape)),
            tx=optax.adam(learning_rate=ALPHA),
        )
        self.baseline.apply = jax.jit(self.baseline.apply)
        print(self.policy.tabulate(self.rng, jnp.ones(
            self.observation_shape)))

    @functools.partial(jax.jit, static_argnums=(0,))
    def policy_prob(self, q_state, state):
        probs = self.policy.apply(q_state.params, state)
        return probs

    def sample(self, state):
        probs = self.policy_prob(self.policy_state, state)[0]
        return np.random.choice(self.num_actions, p=np.array(probs))

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(self, policy_state, baseline_state, states, actions, discounted_rewards, gamma_t):
        def mse_loss(params):
            v_s = self.baseline.apply(params, states)
            delta = jnp.subtract(discounted_rewards, jnp.reshape(v_s, (-1,)))
            loss = jnp.mean(0.5*jnp.square(delta))
            return loss, delta

        def log_prob_loss(params):
            probs = self.policy.apply(params, states)
            log_probs = jnp.log(probs)
            actions_new = jax.nn.one_hot(actions, num_classes=self.num_actions)
            prob_reduce = -jnp.sum(log_probs*actions_new, axis=1)
            loss = jnp.mean(prob_reduce*delta*gamma_t)
            return loss

        (loss_baseline, delta), grads_baseline = jax.value_and_grad(
            mse_loss, has_aux=True)(baseline_state.params)

        loss_policy, grads_policy = jax.value_and_grad(
            log_prob_loss)(policy_state.params)

        baseline_state = baseline_state.apply_gradients(
            grads=grads_baseline)
        policy_state = policy_state.apply_gradients(
            grads=grads_policy)
        return loss_policy+loss_baseline, policy_state, baseline_state

    def train_single_step(self):
        state = self.env.reset(seed=self.seed)[0]
        key = self.rng

        episode_rewards = []
        episode_states = []
        episode_actions = []

        for _ in range(500):
            _, key = jax.random.split(key=key)
            action = self.sample(np.expand_dims(state, axis=0))
            episode_actions.append(action)
            episode_states.append(state)
            next_state, reward, done, truncated, info = self.env.step(action)
            episode_rewards.append(reward)
            state = next_state
            if done or truncated:
                break

        discounted_rewards = jnp.array([sum(reward * (GAMMA ** t) for t, reward in enumerate(episode_rewards[start:]))
                                        for start in range(len(episode_rewards))])
        gamma_t = jnp.array([sum(GAMMA ** t for t, reward in enumerate(episode_rewards[start:]))
                             for start in range(len(episode_rewards))])
        discounted_rewards = (
            discounted_rewards-discounted_rewards.mean())/(discounted_rewards.std()+1e-8)
        episode_states = jnp.array(episode_states)
        episode_actions = jnp.array(episode_actions)
        loss, self.policy_state, self.baseline_state = self.update(
            self.policy_state, self.baseline_state, episode_states, episode_actions, discounted_rewards, gamma_t)
        gc.collect()
        jax.clear_caches()
        return loss, np.sum(episode_rewards)


class Simulation:
    def __init__(self, env_name, algorithm) -> None:
        self.env_name = env_name
        self.algorithm = algorithm
        self.env = gym.make(self.env_name)
        self.num_actions = self.env.action_space.n
        self.observation_shape = self.env.observation_space.shape

    def train(self, episodes=1000):
        self.losses, self.rewards = np.zeros(
            (5, episodes)), np.zeros((5, episodes))

        for seed in range(5):
            self.algo = self.algorithm(
                self.env, self.num_actions, self.observation_shape, seed=seed)
            for ep in tqdm.tqdm(range(1, episodes+1)):
                loss, reward = self.algo.train_single_step()
                self.losses[seed][ep-1] = loss
                self.rewards[seed][ep-1] = reward


if __name__ == '__main__':
    cartpole_baseline = Simulation('CartPole-v1', algorithm=MC_Baseline)
    cartpole_baseline.train()
    rewards_cartpole_baseline = cartpole_baseline.rewards
    mean_rcb = np.mean(rewards_cartpole_baseline, axis=0)
    std_rcb = np.std(rewards_cartpole_baseline, axis=0)
    plot_data(mean_rcb, std_rcb, name='Cartpole PG Baseline')

    acrobot_baseline = Simulation('Acrobot-v1', algorithm=MC_Baseline)
    acrobot_baseline.train()
    rewards_acrobot_baseline = acrobot_baseline.rewards
    mean_rab = np.mean(rewards_acrobot_baseline, axis=0)
    std_rab = np.std(rewards_acrobot_baseline, axis=0)
    plot_data(mean_rab, std_rab, name='Acrobot PG Baseline')
