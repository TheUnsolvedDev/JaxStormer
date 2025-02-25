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
    """
    Smooths out rewards using a moving average window defined by window_size.
    
    Parameters:
    rewards (array): The array of rewards to be smoothed.
    window_size (int): The size of the moving average window (default is 10).
    
    Returns:
    array: Smoothed rewards array after applying the moving average.
    """
    smoothed_rewards = np.zeros_like(rewards)
    for i in range(len(rewards)):
        window_start = max(0, i - window_size // 2)
        window_end = min(len(rewards), i + window_size // 2 + 1)
        smoothed_rewards[i] = np.mean(rewards[window_start:window_end])
    return smoothed_rewards


def plot_data(mean, std, name):
    """
    Plots the data with mean, standard deviation, and name.

    Parameters:
    mean (array): Array containing the mean values.
    std (array): Array containing the standard deviation values.
    name (str): Name of the plot.

    Returns:
    None
    """
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
        """
        A function that represents the forward pass of the neural network.

        Parameters:
            x (ndarray): The input tensor to the network.

        Returns:
            ndarray: The output tensor of the network.
        """
        x = flax.linen.Dense(16)(x)
        x = flax.linen.gelu(x)
        x = flax.linen.Dense(16)(x)
        x = flax.linen.gelu(x)
        x = flax.linen.Dense(self.action_dim)(x)
        x = flax.linen.softmax(x)
        return x


class MC_Reinforce:
    def __init__(self, env, num_actions, observation_shape, seed=0):
        """
        Initializes the MC_Reinforce class with the provided environment, number of actions, observation shape, and optional seed.
        Sets up the random number generator, policy network, and its state.
        """
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
        print(self.policy.tabulate(self.rng, jnp.ones(
            self.observation_shape)))

    @functools.partial(jax.jit, static_argnums=(0,))
    def policy_prob(self, q_state, state):
        """
        Compute the probability of a given state under a policy.

        Args:
            q_state (object): The current state of the Q-function.
            state (object): The state for which to compute the probability.

        Returns:
            object: The probability of the given state under the policy.
        """
        probs = self.policy.apply(q_state.params, state)
        return probs

    def sample(self, state):
        """
        Samples an action from the policy probability distribution given a state.

        Parameters:
            state (object): The current state.

        Returns:
            int: The sampled action.
        """
        probs = self.policy_prob(self.policy_state, state)[0]
        action = np.random.choice(self.num_actions, p=np.array(probs))
        return action

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(self, policy_state, states, actions, discounted_rewards):
        def log_prob_loss(params):
            probs = self.policy.apply(params, states)
            log_probs = jnp.log(probs)
            actions_new = jax.nn.one_hot(actions, num_classes=self.num_actions)
            prob_reduce = -jnp.sum(log_probs*actions_new, axis=1)
            loss = jnp.mean(prob_reduce*discounted_rewards)
            return loss

        loss, grads = jax.value_and_grad(
            log_prob_loss)(policy_state.params)
        policy_state = policy_state.apply_gradients(grads=grads)
        return loss, policy_state

    def train_single_step(self,reward_shape = 0):
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
            reward = reward_shape if done or truncated else reward
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
        loss, self.policy_state = self.update(self.policy_state, episode_states, episode_actions,
                                              discounted_rewards*gamma_t)
        gc.collect()
        jax.clear_caches()
        return loss, np.sum(episode_rewards)


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
    cartpole_reinforce = Simulation('CartPole-v1', algorithm=MC_Reinforce, reward_shape=-3)
    cartpole_reinforce.train()
    rewards_cartpole_reinforce = cartpole_reinforce.rewards
    mean_rcr = np.mean(rewards_cartpole_reinforce, axis=0)
    std_rcr = np.std(rewards_cartpole_reinforce, axis=0)
    plot_data(mean_rcr, std_rcr, name='Cartpole PG Reinforce')

    acrobot_reinforce = Simulation('Acrobot-v1', algorithm=MC_Reinforce, reward_shape=3)
    acrobot_reinforce.train()
    rewards_acrobot_reinforce = acrobot_reinforce.rewards
    mean_rar = np.mean(rewards_acrobot_reinforce, axis=0)
    std_rar = np.std(rewards_acrobot_reinforce, axis=0)
    plot_data(mean_rar, std_rar, name='Acrobot PG Reinforce')
