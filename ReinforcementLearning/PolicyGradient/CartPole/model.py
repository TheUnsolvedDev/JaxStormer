import flax.linen
import flax.training
import flax.training.train_state
import jax
import flax
import jax.numpy as jnp
import numpy as np
import optax
import functools

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
    
class PolicyGradientAgent:
    def __init__(self,input_shape = 5,output_shape=2):
        self.input_shape = input_shape
        self.num_actions = output_shape
        self.policy = Policy(self.num_actions)
        self.policy_state = flax.training.train_state.TrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(jax.random.PRNGKey(0), jnp.ones(self.input_shape)),
            tx=optax.adam(learning_rate=LEARNING_RATE)
        )
        self.policy.apply = jax.jit(self.policy.apply)
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def act(self, rng, policy_params,state):
        probs = self.policy.apply(policy_params,state)
        action_probs = jax.random.categorical(rng, probs)
        return action_probs
        
        
    @functools.partial(jax.jit,static_argnums=(0,))
    def compute_discounted_rewards(self,rewards_buffer, dones_buffer, gamma=0.99):
        
        def scan_fn(carry, inputs):
            reward, done = inputs
            carry = reward + gamma * carry * (1 - done)
            return carry, carry
        
        def process_env(rewards, dones):
            _, discounted = jax.lax.scan(scan_fn, 0.0, (rewards[::-1], dones[::-1]))
            return discounted[::-1]
        
        discounted_rewards = jnp.vstack(jax.vmap(process_env)(rewards_buffer, dones_buffer))
        return discounted_rewards
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def update(self,policy_state,experiences):
        action_probs, states, rewards, dones = experiences
        discounted_rewards = self.compute_discounted_rewards(rewards, dones)
        def log_prob_loss(params):
            probs = self.policy.apply(params, policy_state.tx.init(x))
            log_probs = jnp.log(probs)
            action_probs = jnp.expand_dims(action_probs, axis=-1)
            loss = -jnp.sum(log_probs * action_probs,axis =1)
            return jnp.mean(loss)
        
        loss, grads = jax.value_and_grad(log_prob_loss)(policy_state.params)
        policy_state = policy_state.apply_gradients(grads)
        return policy_state, loss
    
if __name__ == '__main__':
    agent = PolicyGradientAgent()
    x = jnp.ones((4,5))
    rng = jax.random.PRNGKey(0)
    params = agent.policy_state.params
    print(agent.act(rng, params,x))