import flax.linen
import jax
import flax
import optax
import jax.numpy as jnp
from typing import *


class QNetwork(flax.linen.Module):
    action_dim: int

    @flax.linen.compact
    def __call__(self, x: jnp.ndarray):
        x = flax.linen.Dense(16)(x)
        x = flax.linen.selu(x)
        x = flax.linen.Dense(16)(x)
        x = flax.linen.selu(x)
        q_values = flax.linen.Dense(self.action_dim)(x)
        return q_values


class AdvantageNetwork(flax.linen.Module):
    action_dim: int

    @flax.linen.compact
    def __call__(self, x: jnp.ndarray):
        x = flax.linen.Dense(16)(x)
        x = flax.linen.selu(x)
        x = flax.linen.Dense(16)(x)
        x = flax.linen.selu(x)
        value_stream = flax.linen.Dense(1)(x)
        advantage_stream = flax.linen.Dense(self.action_dim)(x)
        q_values = value_stream + \
            (advantage_stream - jnp.mean(advantage_stream, axis=-1, keepdims=True))
        return q_values


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


class ValueNetwork(flax.linen.Module):
    action_dim: int

    @flax.linen.compact
    def __call__(self, x: jnp.ndarray):
        x = flax.linen.Dense(16)(x)
        x = flax.linen.selu(x)
        x = flax.linen.Dense(16)(x)
        x = flax.linen.selu(x)
        values = flax.linen.Dense(1)(x)
        return values
