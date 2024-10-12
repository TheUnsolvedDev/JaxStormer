import jax
import numpy as np
import jax.numpy as jnp
import gymnax
import optax
import flax
import functools
from flax.training.train_state import TrainState

ALPHA = 0.001
GAMMA = 0.99


class PolicyNetwork(flax.linen.Module):
    action_dim: int

    @flax.linen.compact
    def __call__(self, x):
        x = flax.linen.Dense(16)(x)
        x = flax.linen.gelu(x)
        x = flax.linen.Dense(16)(x)
        x = flax.linen.gelu(x)
        x = flax.linen.Dense(self.action_dim)(x)
        probs = flax.linen.softmax(x)
        return probs


class Buffer:
    def __init__(self, state_shape, max_size=500) -> None:
        self.max_size = max_size
        self.buffer_state = {
            'states': jnp.empty((max_size, state_shape), dtype=jnp.float32),
            'actions': jnp.empty((max_size,), dtype=jnp.int32),
            'rewards': jnp.empty((max_size,), dtype=jnp.int32),
        }
        self.current_idx = 0

    @functools.partial(jax.jit, static_argnums=(0,))
    def add(self, buffer_state, experience):
        state, action, reward = experience
        idx = self.current_idx % self.max_size
        buffer_state['states'] = buffer_state['states'].at[idx].set(state)
        buffer_state['actions'] = buffer_state['actions'].at[idx].set(action)
        buffer_state['rewards'] = buffer_state['rewards'].at[idx].set(reward)
        self.current_idx += 1
        return buffer_state


class Reinforce:
    def __init__(self, num_actions, observation_shape, seed=0) -> None:
        self.seed = seed
        self.rng = jax.random.PRNGKey(seed)
        self.num_actions = num_actions
        self.observation_shape = observation_shape

        self.policy = PolicyNetwork(action_dim=num_actions)
        self.policy_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(self.rng, jnp.ones(observation_shape)),
            tx=optax.adam(learning_rate=ALPHA),
        )
        self.policy.apply = jax.jit(self.policy.apply)

    @functools.partial(jax.jit, static_argnums=(0,))
    def act(self, rng, policy_params, state):
        probs = self.policy.apply(policy_params, state)
        action = jax.random.choice(rng, jnp.arange(
            self.num_actions), shape=(NUM_ENVS,), p=probs)
        rng, subkey = jax.random.split(rng)
        return action, rng

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_discounted_rewards(self, rewards):
        return jax.vmap(lambda x: jnp.sum(GAMMA**np.arange(len(x)) * x))(rewards)

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(self, policy_state, experience):
        states, actions, discounted_rewards = experience

        @jax.jit
        def _batch_loss_fn(params, states, actions, discounted_rewards):
            @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0))
            def _loss_fn(params, state, action, discounted_reward):
                prob = self.policy.apply(params, state)
                log_prob = jnp.log(prob)
                actions_new = jax.nn.one_hot(
                    action, num_classes=self.num_actions)
                prob_reduce = -jnp.sum(log_prob*actions_new, axis=1)
                loss = prob_reduce*discounted_reward
                return loss
            return jnp.mean(_loss_fn(params, states, actions, discounted_rewards))

        loss, grads = jax.value_and_grad(_batch_loss_fn)(
            policy_state.params, states, actions, discounted_rewards)
        policy_state = policy_state.apply_gradients(grads=grads)
        return loss, policy_state


def main():
    pass


if __name__ == "__main__":
    a = jnp.empty((16, 4))
    print(a)
