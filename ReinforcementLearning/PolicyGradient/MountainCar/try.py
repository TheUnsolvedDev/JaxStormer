# import numpy as np

import numpy as np
import jax.numpy as jnp
import jax


def print_traj(trajectories):
    for trajectory in trajectories:
        print(trajectory)

# def compute_discounted_rewards(trajectories, gamma=0.99):
#     discounted_rewards = []
#     for trajectory in trajectories:
#         rewards = np.array(trajectory, dtype=np.float16)
#         discounted = np.zeros_like(rewards)
#         running_sum = 0
#         for t in reversed(range(len(rewards))):
#             running_sum = rewards[t] + gamma * running_sum
#             discounted[t] = running_sum
#         discounted_rewards.append(discounted.tolist())
#     return discounted_rewards

# if __name__ == '__main__':
#     sizes = np.random.randint(1,10,10)
#     x = [[j for j in range(sizes[i])] for i in range(10)]
#     print_traj(x)

#     new_x = compute_discounted_rewards(x)
#     print_traj(new_x)

# import numpy as np

# def compute_discounted_rewards(rewards_buffer, dones_buffer, gamma=0.99):
#     num_envs = len(rewards_buffer)
#     discounted_rewards = [[] for _ in range(num_envs)]
#     running_sums = np.zeros(num_envs, dtype=np.float32)

#     for t in reversed(range(len(rewards_buffer[0]))):  # Assuming equal length buffers
#         running_sums = rewards_buffer[:, t] + gamma * running_sums * (1 - dones_buffer[:, t])
#         for i in range(num_envs):
#             discounted_rewards[i].insert(0, running_sums[i])

#     return discounted_rewards

# # Example rewards and done signals for multiple environments (auto-resetting)
# rewards_buffer = np.array([
#     [0, 1, 2, 3, 0, 1, 2],
#     [0, 1, 2, 3, 4, 5, 6],
#     [0, 1, 2, 0, 1, 2, 3],
#     [0, 1, 0, 1, 2, 3, 4]
# ], dtype=np.float32)

# dones_buffer = np.array([
#     [0, 0, 0, 1, 0, 0, 1],
#     [0, 0, 0, 0, 0, 0, 1],
#     [0, 0, 1, 0, 0, 0, 1],
#     [0, 1, 0, 0, 0, 0, 1]
# ], dtype=np.float32)

# discounted = compute_discounted_rewards(rewards_buffer, dones_buffer, gamma=0.99)
# print_traj(discounted)

# import jax.numpy as jnp
# import jax.lax as lax
# import jax

# @jax.jit
# def compute_discounted_rewards(rewards_buffer, dones_buffer, gamma=0.99):
#     # num_envs, seq_len = rewards_buffer.shape

#     def scan_fn(carry, inputs):
#         reward, done = inputs
#         carry = reward + gamma * carry * (1 - done)
#         return carry, carry

#     def process_env(rewards, dones):
#         _, discounted = lax.scan(scan_fn, 0.0, (rewards[::-1], dones[::-1]))
#         return discounted[::-1]

#     discounted_rewards = jnp.vstack(jax.vmap(process_env)(rewards_buffer, dones_buffer))
#     return discounted_rewards

# # Example rewards and done signals for multiple environments (auto-resetting)
# rewards_buffer = jnp.array([
#     [0, 1, 2, 3, 0, 1, 2],
#     [0, 1, 2, 3, 4, 5, 6],
#     [0, 1, 2, 0, 1, 2, 3],
#     [0, 1, 0, 1, 2, 3, 4]
# ], dtype=jnp.float32)

# dones_buffer = jnp.array([
#     [0, 0, 0, 1, 0, 0, 1],
#     [0, 0, 0, 0, 0, 0, 1],
#     [0, 0, 1, 0, 0, 0, 1],
#     [0, 1, 0, 0, 0, 0, 1]
# ], dtype=jnp.float32)

# discounted = compute_discounted_rewards(rewards_buffer, dones_buffer, gamma=0.99)
# print(discounted)

# import jax.numpy as jnp
# import jax

# def compute_total_rewards(rewards_buffer, dones_buffer):
#     def process_env(rewards, dones):
#         dones_shifted = jnp.concatenate([jnp.zeros((1,), dtype=jnp.float32), dones[:-1]])
#         episode_starts = jnp.where(dones_shifted, 1, 0)
#         episode_ids = jnp.cumsum(episode_starts)
#         total_rewards = jax.ops.segment_sum(rewards, episode_ids)
#         return total_rewards

#     total_rewards = jax.vmap(process_env)(rewards_buffer, dones_buffer)
#     return total_rewards

# # Example rewards and done signals for multiple environments (auto-resetting)
# rewards_buffer = jnp.array([
#     [0, 1, 2, 3, 0, 1, 2],
#     [0, 1, 2, 3, 4, 5, 6],
#     [0, 1, 2, 0, 1, 2, 3],
#     [0, 1, 0, 1, 2, 3, 4]
# ], dtype=jnp.float32)

# dones_buffer = jnp.array([
#     [0, 0, 0, 1, 0, 0, 1],
#     [0, 0, 0, 0, 0, 0, 1],
#     [0, 0, 1, 0, 0, 0, 1],
#     [0, 1, 0, 0, 0, 0, 1]
# ], dtype=jnp.float32)

# total_rewards = compute_total_rewards(rewards_buffer, dones_buffer)
# print(total_rewards)

import functools
class TrajectoryBuffer:
    def __init__(self, input_size, num_envs, trajectory_size):
        self.input_size = input_size
        self.trajectory_size = trajectory_size
        self.num_envs = num_envs
        self.states = jnp.zeros(
            (self.trajectory_size, self.num_envs, self.input_size))
        self.current_idx = 0

    @functools.partial(jax.jit, static_argnums = (0,))
    def add(self, state):
        idx = self.current_idx % self.trajectory_size

        def body_fun(i, states):
            return states.at[idx, i].set(state[i])

        self.states = jax.lax.fori_loop(0, self.num_envs, body_fun, self.states)
        self.current_idx += 1

    def get_buffer(self):
        return self.states.reshape((self.trajectory_size, self.num_envs * self.input_size))



if __name__ == '__main__':
    input_size = 4
    num_envs = 5
    trajectory_size = 10
    buffer = TrajectoryBuffer(input_size, num_envs, trajectory_size)
    for i in range(trajectory_size):
        states = jnp.ones((num_envs,input_size))*(i+1)
        buffer.add(states)
        print(f'After adding trajectory {i+1}:')
        print(buffer.states)

# print(rewards_buffer.shape)