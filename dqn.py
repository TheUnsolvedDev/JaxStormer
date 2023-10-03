import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import flax
import optax
import orbax
import tqdm
import csv
import os
import time
from flax.training.train_state import TrainState

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
key = jax.random.PRNGKey(0)

ENV = 'CartPole-v1'
ALPHA = 2.5e-4
GAMMA = 0.99
TOTAL_TIME_STEPS = int(5e+5)
LEARNING_START = int(1e+4)
BUFFER_SIZE = int(1e+4)
EPSILON = 1
TRAIN_FREQUENCY = 10
UPDATE_TARGET_FREQUENCY = 500
BATCH_SIZE = 32
NUM_ENVS = 1
TAU = 0.9


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
        return jax.random.choice(jax.random.PRNGKey(np.random.randint(low=0, high=100)), len(self.buffer), shape=(batch_size,), replace=False)

    def get_batch(self, indices):
        states, actions, rewards, next_states, dones = zip(
            *[self.buffer[i] for i in indices])
        print(np.sum(actions),len(actions))
        return jnp.array(states).squeeze(), jnp.array(actions).squeeze(), jnp.array(rewards).squeeze(),\
            jnp.array(next_states).squeeze(), jnp.array(dones).squeeze()


class Q(flax.linen.Module):
    action_dim: int

    @flax.linen.compact
    def __call__(self, x: jnp.ndarray):
        x = flax.linen.Dense(120)(x)
        x = flax.linen.relu(x)
        x = flax.linen.Dense(84)(x)
        x = flax.linen.relu(x)
        x = flax.linen.Dense(self.action_dim)(x)
        return x


def create_env(indx=0, name=ENV, video_stats=False):
    if video_stats and indx == 0:
        env = gym.make(name, max_episode_steps=BUFFER_SIZE//10,
                       render_mode='rgb_array')
        env = gym.wrappers.RecordVideo(
            env, f"RL_updates/{name}_{int(time.time())}", step_trigger=lambda x: x % int(1e+4) == 0)
    else:
        env = gym.make(name, max_episode_steps=BUFFER_SIZE//10)
    return env


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def test(policy, q_state, num_games=10):
    env = create_env()
    rewards = []

    for game in (range(num_games)):
        sum_reward = 0
        state, _ = env.reset()
        done = False
        while not done:
            action = policy(q_state, state, epsilon=0.01)
            action = jax.device_get(action)
            next_state, reward, done, truncated, infos = env.step(action)
            sum_reward += reward
            state = next_state
        rewards.append(sum_reward)
    return np.mean(rewards)


def main():
    key = jax.random.PRNGKey(0)
    env = create_env()
    replay_buffer = ReplayBuffer(BUFFER_SIZE)
    q_network = Q(env.action_space.n)
    q_state = TrainState.create(
        apply_fn=q_network.apply,
        params=q_network.init(key, jnp.zeros(env.observation_space.shape)),
        target_params=q_network.init(
            key, jnp.zeros(env.observation_space.shape)),
        tx=optax.adam(learning_rate=ALPHA),
    )
    q_network.apply = jax.jit(q_network.apply)
    q_state = q_state.replace(target_params=optax.incremental_update(
        q_state.params, q_state.target_params, 1))
    print(q_network.tabulate(key, jnp.ones(
        env.observation_space.shape)))

    @jax.jit
    def policy(q_state, state, epsilon=0.1, key=jax.random.PRNGKey(0)):
        prob = jax.random.uniform(key)
        q_values = q_network.apply(q_state.params, state)
        action = q_values.argmax(axis=-1)
        a = jax.lax.cond(prob < epsilon, lambda x: np.random.randint(2),
                         lambda x: action, None)
        return a

    # @jax.jit
    def update(q_state, states, actions, rewards, next_states,  dones):
        q_next_target = q_network.apply(q_state.target_params, next_states)
        q_next_target = jnp.max(q_next_target, axis=-1)
        next_q_value = (rewards + (1 - dones) * GAMMA * q_next_target)

        def mse_loss(params):
            q_pred = q_network.apply(params, states)
            q_pred = q_pred[jnp.arange(q_pred.shape[0]), actions.squeeze()]
            print(actions.squeeze())
            return ((jax.lax.stop_gradient(next_q_value) - q_pred) ** 2).mean(), q_pred

        (loss_value, q_pred), grads = jax.value_and_grad(
            mse_loss, has_aux=True)(q_state.params)
        q_state = q_state.apply_gradients(grads=grads)
        return loss_value, jnp.mean(q_pred), q_state

    state, _ = env.reset()
    start = time.time()
    epsilon = EPSILON
    for step in tqdm.tqdm(range(TOTAL_TIME_STEPS+1)):

        new_key, key = jax.random.split(key)
        action = policy(q_state, state, epsilon, key=new_key)
        action = jax.device_get(action)
        
        next_state, reward, done, truncated, infos = env.step(action)
        replay_buffer.push([state, action, reward, next_state, done])

        state = next_state
        if done:
            state, _ = env.reset()

        if step > LEARNING_START:
            epsilon = linear_schedule(
                start_e=EPSILON, end_e=0.05, duration=TOTAL_TIME_STEPS*0.25, t=step)
            if not step % TRAIN_FREQUENCY:
                indices = replay_buffer.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = replay_buffer.get_batch(
                    indices)
                loss_values, q_pred, q_state = update(
                    q_state, states, actions, rewards, next_states,  dones)

            if not step % UPDATE_TARGET_FREQUENCY:
                q_state = q_state.replace(target_params=optax.incremental_update(
                    q_state.params, q_state.target_params, 1))
                test_results = test(policy, q_state)
                if not step % (UPDATE_TARGET_FREQUENCY):
                    print("td_loss:", jax.device_get(loss_values),
                          "Q_value:", q_pred, 'Position:', test_results, 'Epsilon:', epsilon)
                with open(f'RL_updates/logs_{start}.txt', 'a', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(
                        [loss_values, q_pred, test_results, epsilon])

                from flax.training import orbax_utils
                orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                ckpt = {'q_state': q_state}
                save_args = orbax_utils.save_args_from_target(ckpt)
                orbax_checkpointer.save(
                    f'RL_updates/single_save/{step}/{int(time.time())}', ckpt, save_args=save_args)
    env.close()


if __name__ == '__main__':
    main()
