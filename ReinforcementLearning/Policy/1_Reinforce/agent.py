import jax
import jax.numpy as jnp
import numpy as np
import optax
import functools
from flax.training.train_state import TrainState

from config import *
from model import *
from buffer import *


class Agent:
    def __init__(self, input_shape, output_shape) -> None:
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.transition_buffer = TransitionBuffer(state_shape=input_shape)
        self.policy = PolicyNetwork(action_dim=self.output_shape)
        self.policy_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(rng, jnp.ones(self.input_shape)),
            tx=optax.adam(learning_rate=ALPHA),
        )
        self.policy.apply = jax.jit(self.policy.apply)

    # @functools.partial(jax.jit, static_argnums=(0,))
    def act(self, rng, policy_params, state):
        probs = self.policy.apply(policy_params, state)
        action = jax.random.choice(rng, jnp.arange(self.output_shape), p=probs[0])
        return action

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_discounted_rewards(self, rewards):
        def discount_step(carry, reward):
            discounted_sum = reward + GAMMA * carry
            return discounted_sum, discounted_sum
        
        _, discounted_rewards_array = jax.lax.scan(discount_step, 0.0, rewards[::-1])
        return discounted_rewards_array[::-1]
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def update(self, policy_state, experiences):
        states, actions, discounted_rewards = experiences
        def log_prob_loss(params):
            probs = self.policy.apply(params, states)
            log_probs = jnp.log(probs)
            actions_new = jax.nn.one_hot(actions, num_classes=self.output_shape)
            prob_reduce = -jnp.sum(log_probs*actions_new, axis=1)
            loss = jnp.mean(prob_reduce*discounted_rewards)
            return loss

        loss, grads = jax.value_and_grad(
            log_prob_loss)(policy_state.params)
        policy_state = policy_state.apply_gradients(grads=grads)
        return loss, policy_state
