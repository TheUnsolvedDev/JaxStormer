import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

NUM_BANDITS = 10
MEANS = [i for i in range(NUM_BANDITS)]
key = jax.random.PRNGKey(NUM_BANDITS)


class Bandit:
    def __init__(self, num_arms: int, mean_of_arms: list) -> None:
        assert len(mean_of_arms) == num_arms
        self.num_arms = num_arms
        self.mean_of_arms = mean_of_arms

    def best_arm(self) -> int:
        return np.argmax(self.mean_of_arms), np.max(self.mean_of_arms)

    def sample(self, action: int) -> float:
        return np.random.normal(loc=self.mean_of_arms[action], scale=1)

    def plot_distribution(self, num_samples: int = 10000) -> None:
        samples = np.zeros((self.num_arms, num_samples))
        fig, ax = plt.subplots(1, 1, figsize=(9, 5))
        for arm in range(self.num_arms):
            for i in range(num_samples):
                samples[arm][i] = self.sample(arm)
            sns.histplot(samples[arm, :], ax=ax, stat='density',
                         kde=True, bins=100, label=f'arm{arm}')
            ax.legend()
        plt.show()


class RandomPolicy:
    def __init__(self, num_arms: int) -> None:
        self.num_arms = num_arms

    def reset(self) -> None:
        pass

    def update(self, action: int, reward: float) -> None:
        pass

    def select_action(self) -> int:
        return np.random.randint(self.num_arms)


class EpsilonGreedyPolicy:
    def __init__(self, num_arms: int, epsilon: float) -> None:
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.reset()

    def reset(self) -> None:
        self.Q = jnp.zeros((self.num_arms))
        self.arm_count = jnp.zeros(self.num_arms)

    def update(self, action: int, reward: float) -> None:
        self.arm_count.at[action].add(1)
        self.Q.at[action].add(1/self.arm_count[action]*(reward-self.Q[action]))

    def select_action(self) -> int:
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.randint(self.num_arms)
        else:
            action = jnp.argmax(self.Q)
        return action


class SoftmaxPolicy:
    def __init__(self, num_arms: int, tau: float) -> None:
        self.num_arms = num_arms
        self.tau = tau
        self.reset()

    def reset(self) -> None:
        self.Q = jnp.zeros((self.num_arms))
        self.arm_count = jnp.zeros(self.num_arms)

    def update(self, action: int, reward: float) -> None:
        self.arm_count.at[action].add(1)
        self.Q.at[action].add(1/self.arm_count[action]*(reward-self.Q[action]))

    def select(self) -> int:
        Q = self.Q.copy()
        exp_x = np.exp(Q/self.tau - Q.max()/self.tau)
        probs = exp_x/exp_x(axis=1)
        action = jax.random.choice(key, action, p=probs)
        return action


def simulate(agent, num_trials: int = 5, num_episodes: int = 5000):
    game_history = np.zeros((num_trials, num_episodes))
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    for game in range(num_trials):
        agent.reset()
        bandit = Bandit(num_arms=NUM_BANDITS, mean_of_arms=MEANS)
        for episode in tqdm.tqdm(range(num_episodes)):
            action = agent.select_action()
            reward = bandit.sample(action)
            agent.update(action, reward)
            game_history[game][episode] += reward
        ax.plot(game_history[game, :].cumsum() /
                np.arange(1, num_episodes+1), label=f'game_{game}')
    ax.set_ylabel('avg_rewards')
    ax.set_xlabel('episode')
    ax.legend()
    plt.show()


# implement :
#     1. Random policy
#     2. Epsilon Greedy policy
#     3. Softmax policy
#     4. UCB policy

if __name__ == '__main__':
    bandit = Bandit(num_arms=NUM_BANDITS, mean_of_arms=MEANS)
    bandit.plot_distribution()

    simulate(EpsilonGreedyPolicy(num_arms=NUM_BANDITS, epsilon=0.9))
