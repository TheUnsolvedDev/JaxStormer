
import jax
import jax.numpy as jnp
import numpy as np
import flax
import functools
import optax
from flax.training.train_state import TrainState

from Models import Policy


class ReinforceAlgorithm:
    def __init__(self, env, env_params, num_actions, observation_shape, gamma=0.99, alpha=0.001, weight_decay=0.01, tau=0.01, max_episode_length=1000):
        self.num_actions = num_actions
        self.env_params = env_params
        self.observation_shape = observation_shape
        self.gamma = gamma
        self.alpha = alpha
        self.env = env
        self.weight_decay = weight_decay
        self.tau = tau
        self.max_episode_length = max_episode_length
        self.policy = Policy.Pi(self.num_actions)
        self.policy_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(jax.random.PRNGKey(
                0), jnp.ones(self.observation_shape)),
            tx=optax.adam(learning_rate=self.alpha)
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def act(self, rng, policy_params, state):
        state = jnp.expand_dims(state, axis=0)
        probs = self.policy.apply(policy_params, state)
        probs = jnp.squeeze(probs, axis=0)
        action = jax.random.choice(rng, jnp.arange(
            self.num_actions), shape=(1,), p=probs)
        rng, subkey = jax.random.split(rng)
        return action[0], rng

    @functools.partial(jax.jit, static_argnums=(0,))
    def generate_trajectory(self, rng=jax.random.PRNGKey(0)):
        rng_reset, rng_episode = jax.random.split(rng, 2)
        state, obs = self.env.reset(rng_reset, self.env_params)

        @jax.jit
        def policy_step(obs_input, tmp):
            state, obs, rng = obs_input
            rng, rng_step = jax.random.split(rng)
            action, _ = self.act(rng_step, self.policy_state.params, state)
            next_state, next_obs, reward, done, truncated = self.env.step(
                rng_step, obs, action, self.env_params)
            carry = [next_state, next_obs, rng]
            return carry, [state, action, reward, next_state, done]

        _, trajectory = jax.lax.scan(
            policy_step, [state, obs, rng_episode], (), self.max_episode_length)
        states, actions, rewards, next_states, dones = trajectory
        mask = jnp.array(1-jnp.array(dones, dtype=jnp.float32)).cumsum() <= 1
        mask = jnp.array(dones, dtype=jnp.float32).cumsum(axis=0)
        length_mask = jnp.sum(mask < 1)+1
        length_mask = jax.lax.stop_gradient(length_mask).astype(int)

        # print(length_mask, self.observation_shape[0])

        # states = jax.lax.dynamic_slice(
        #     states, (0, 0), (length_mask, self.observation_shape[0]))
        # actions = jax.lax.dynamic_slice(actions, (0,), (length_mask,))
        # rewards = jax.lax.dynamic_slice(rewards, (0,), (length_mask,))
        # dones = jax.lax.dynamic_slice(dones, (0,), (length_mask,))
        return states, actions, rewards, dones, length_mask
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def get_discounted_rewards(self, rewards):
        reversed_rewards = rewards[::-1]
        discounted_rewards = jax.numpy.cumsum(reversed_rewards * (self.gamma ** jnp.arange(len(rewards))))
        discounted_rewards = discounted_rewards[::-1]
        return discounted_rewards

    @functools.partial(jax.jit, static_argnums=(0,))
    def train_single_step(self, experience, policy_state):
        states, actions, rewards = experience
        @jax.jit
        def loss_fn(params, states, actions, rewards):
            probs = self.policy.apply(params, states)
            log_probs = jnp.log(probs)
            actions_new = jax.nn.one_hot(
                actions, num_classes=self.num_actions)
            prob_reduce = -jnp.sum(log_probs*actions_new, axis=1)
            loss = prob_reduce*rewards
            return loss.mean()

        loss, grads = jax.value_and_grad(loss_fn)(
            policy_state.params, states, actions, rewards)
        policy_state = policy_state.apply_gradients(grads=grads)
        return policy_state, loss.mean()

    def train(self, iteration=10000):
        for i in range(iteration):
            rng = jax.random.PRNGKey(i)
            states, actions, rewards, dones, length_mask = self.generate_trajectory(rng)
            states,actions,rewards,dones = states[:length_mask],actions[:length_mask],rewards[:length_mask],dones[:length_mask]
            discounted_rewards = self.get_discounted_rewards(rewards*(1-dones))
            policy_state, loss = self.train_single_step(
                (states, actions, discounted_rewards), self.policy_state)
            print(f'\r Game [{i}\{iteration}] loss:\t {loss:.4f}\t reward:\t {rewards.sum()}')
            self.policy_state = policy_state
        print()    
        
        # def looping_function(i, policy_state):
        #     rng = jax.random.PRNGKey(i)
        #     states, actions, rewards = self.generate_trajectory(
        #         rng)
        #     policy_state, loss, rewards_sum = self.train_single_step(
        #         (states, actions, rewards, dones, length_mask), policy_state)
        #     return policy_state

        # val = jax.lax.fori_loop(
        #     0, iteration, looping_function, self.policy_state)

    def evaluate(self):
        pass
