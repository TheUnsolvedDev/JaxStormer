# import numpy as np

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

import jax 
import jax.numpy as jnp
import flashbax

buffer = flashbax.make_trajectory_buffer(
    add_batch_size=1,
    sample_batch_size=16,
    sample_sequence_length=16,
    period=32,
    min_length_time_axis=32,
    max_size=64,
)
state = buffer.init({'states': jnp.ones((5,)), 'rewards':jnp.array(1.0)})
# print(state)
for i in range(40):
    state = buffer.add(state, {'states': i*jnp.ones((5,)), 'rewards':i*jnp.array(1.0)})
print(state)

print(buffer.sample(state,jax.random.PRNGKey(0)))