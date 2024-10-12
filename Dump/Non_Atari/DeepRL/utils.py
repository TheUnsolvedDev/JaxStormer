import jax
import numpy as np
import matplotlib.pyplot as plt


@jax.jit
def linear_schedule(start_e: float, end_e: float, duration: int, t: int) -> float:
    """
    Calculates a linear schedule.

    Args:
        start_e (float): The starting value of the schedule.
        end_e (float): The end value of the schedule.
        duration (int): The duration of the schedule.
        t (int): The current time step of the schedule.

    Returns:
        e (float): The value of the schedule at time step t.
    """
    cond1 = (end_e - start_e) / duration * t + start_e
    cond2 = end_e
    return jax.lax.cond(cond1 > cond2, lambda: cond1, lambda: cond2)


def smooth_rewards(rewards: np.ndarray, window_size: int = 10) -> np.ndarray:
    """
    Smooths out rewards using a moving average window defined by window_size.

    Parameters:
        rewards (np.ndarray): The array of rewards to be smoothed.
        window_size (int): The size of the moving average window (default is 10).

    Returns:
        np.ndarray: Smoothed rewards array after applying the moving average.
    """
    smoothed_rewards = np.zeros_like(rewards)
    for i in range(len(rewards)):
        window_start = max(0, i - window_size // 2)
        window_end = min(len(rewards), i + window_size // 2 + 1)
        smoothed_rewards[i] = np.mean(rewards[window_start:window_end])
    return smoothed_rewards


def plot_data(mean: np.ndarray, std: np.ndarray, name: str) -> None:
    """
    Plots the data with mean, standard deviation, and name.

    Parameters:
    mean (array): Array containing the mean values.
    std (array): Array containing the standard deviation values.
    name (str): Name of the plot.

    Returns:
    None
    """
    x = np.arange(len(mean))

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
