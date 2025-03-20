
import jax
import flax
import jax.numpy as jnp
import numpy as np
import optax
import functools
import flax.training.train_state

from config import *


class Policy(flax.linen.Module):
    num_actions: int

    @flax.linen.compact
    def __call__(self, x):
        x = flax.linen.Dense(64)(x)
        x = flax.linen.leaky_relu(x)
        x = flax.linen.Dense(64)(x)
        x = flax.linen.leaky_relu(x)
        x = flax.linen.Dense(self.num_actions)(x)
        x = flax.linen.softmax(x)
        return x


class TrajectoryBuffer:
    def __init__(self, input_size, num_envs, trajectory_size):
        self.input_size = input_size
        self.trajectory_size = trajectory_size
        self.num_envs = num_envs

        # Initial buffer state (should be passed explicitly)
        self.init_state = {
            "states": jnp.zeros((self.trajectory_size, self.num_envs, self.input_size)),
            "actions": jnp.zeros((self.trajectory_size, self.num_envs)),
            "rewards": jnp.zeros((self.trajectory_size, self.num_envs)),
            "dones": jnp.zeros((self.trajectory_size, self.num_envs)),
            "current_idx": jnp.array(0),
        }

    @functools.partial(jax.jit, static_argnums=(0,))
    def add(self, buffer_state, state, action, reward, done):
        """JIT-compatible add function with explicit state updates."""
        idx = buffer_state["current_idx"] % self.trajectory_size

        new_states = buffer_state["states"].at[idx].set(state)
        new_actions = buffer_state["actions"].at[idx].set(action)
        new_rewards = buffer_state["rewards"].at[idx].set(reward)
        new_dones = buffer_state["dones"].at[idx].set(done)

        new_state = {
            "states": new_states,
            "actions": new_actions,
            "rewards": new_rewards,
            "dones": new_dones,
            "current_idx": buffer_state["current_idx"] + 1,
        }
        return new_state

    @functools.partial(jax.jit, static_argnums=(0,))
    def compute_discounted_rewards(self, rewards_buffer, dones_buffer, gamma=0.99):
        def scan_fn(carry, inputs):
            reward, done = inputs
            carry = reward + gamma * carry * (1 - done)
            return carry, carry

        def process_env(rewards, dones):
            _, discounted = jax.lax.scan(
                scan_fn, 0.0, (rewards[::-1], dones[::-1]))
            return discounted[::-1]

        discounted_rewards = jnp.vstack(
            jax.vmap(process_env)(rewards_buffer, dones_buffer))
        return discounted_rewards

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_buffer(self, buffer_state):
        """Extracts flattened buffer data."""
        states = buffer_state["states"].reshape(
            (self.trajectory_size * self.num_envs, self.input_size))
        actions = buffer_state["actions"].reshape(
            (self.trajectory_size * self.num_envs, 1))

        rewards = buffer_state["rewards"].T
        dones = buffer_state["dones"].T
        discounted_rewards = self.compute_discounted_rewards(rewards, dones).T
        discounted_rewards = discounted_rewards.reshape(
            (self.trajectory_size * self.num_envs, 1))

        return states, actions, discounted_rewards


class PolicyGradientAgent:
    def __init__(self, input_shape=5, output_shape=2):
        self.input_shape = input_shape
        self.num_actions = output_shape
        self.policy = Policy(self.num_actions)
        self.policy_state = flax.training.train_state.TrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(jax.random.PRNGKey(
                0), jnp.ones(self.input_shape)),
            tx=optax.adam(learning_rate=LEARNING_RATE)
        )
        self.policy.apply = jax.jit(self.policy.apply)

    @functools.partial(jax.jit, static_argnums=(0,))
    def act(self, rng, policy_params, state):
        probs = self.policy.apply(policy_params, state)
        action_probs = jax.random.categorical(rng, probs)
        return action_probs

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(self, policy_state, experiences):
        states, actions, discounted_rewards = experiences

        def log_prob_loss(params):
            probs = self.policy.apply(params, states)
            log_probs = jnp.log(probs)
            actions_new = jax.nn.one_hot(
                actions, num_classes=self.num_actions)
            prob_reduce = -jnp.sum(log_probs*actions_new, axis=1)
            loss = jnp.mean(prob_reduce*discounted_rewards)
            return loss

        loss, grads = jax.value_and_grad(log_prob_loss)(policy_state.params)
        policy_state = policy_state.apply_gradients(grads=grads)
        return policy_state, loss


if __name__ == '__main__':
    agent = PolicyGradientAgent()
    x = jnp.ones((4, 5))
    rng = jax.random.PRNGKey(0)
    params = agent.policy_state.params
    print(agent.act(rng, params, x))
