import flax.linen
import jax
import jax.numpy as jnp
import numpy as np
import flax
import config
import distrax

rng = jax.random.PRNGKey(0)


class Q(flax.linen.Module):
    num_actions: int

    @flax.linen.compact
    def __call__(self, state):
        x = flax.linen.Dense(config.HIDDEN)(state)
        x = flax.linen.silu(x)
        x = flax.linen.Dense(config.HIDDEN)(x)
        x = flax.linen.silu(x)
        x = flax.linen.Dense(self.num_actions)(x)
        return x


class Q_adv(flax.linen.Module):
    num_actions: int

    @flax.linen.compact
    def __call__(self, state):
        x = flax.linen.Dense(config.HIDDEN)(state)
        x = flax.linen.silu(x)
        x = flax.linen.Dense(config.HIDDEN)(x)
        x = flax.linen.silu(x)
        value_stream = flax.linen.Dense(1)(x)
        advantage_stream = flax.linen.Dense(self.num_actions)(x)
        q_values = value_stream + \
            (advantage_stream - jnp.mean(advantage_stream, axis=-1, keepdims=True))
        return q_values


class V(flax.linen.Module):
    @flax.linen.compact
    def __call__(self, state):
        x = flax.linen.Dense(config.HIDDEN)(state)
        x = flax.linen.silu(x)
        x = flax.linen.Dense(config.HIDDEN)(x)
        x = flax.linen.silu(x)
        x = flax.linen.Dense(1)(x)
        return x


class Q_SA(flax.linen.Module):
    @flax.linen.compact
    def __call__(self, state, action):
        combined_input = jnp.concat([state, action], axis=-1)
        x = flax.linen.Dense(config.HIDDEN)(combined_input)
        x = flax.linen.silu(x)
        x = flax.linen.Dense(config.HIDDEN)(x)
        x = flax.linen.silu(x)
        x = flax.linen.Dense(1)(x)
        return x


class Pi(flax.linen.Module):
    num_actions: int

    @flax.linen.compact
    def __call__(self, state, seed=jax.random.PRNGKey(0)):
        x = flax.linen.Dense(config.HIDDEN)(state)
        x = flax.linen.silu(x)
        x = flax.linen.Dense(config.HIDDEN)(x)
        x = flax.linen.silu(x)
        x = flax.linen.Dense(self.num_actions)(x)
        distribution = distrax.Categorical(logits=x)
        sample = distribution.sample(seed=seed)
        return sample


class Pi_cont(flax.linen.Module):
    @flax.linen.compact
    def __call__(self, state, seed=jax.random.PRNGKey(0)):
        x = flax.linen.Dense(config.HIDDEN)(state)
        x = flax.linen.silu(x)
        x = flax.linen.Dense(config.HIDDEN)(x)
        x = flax.linen.silu(x)
        mu = flax.linen.Dense(1)(x)
        sigma = flax.linen.Dense(1)(x)
        disribution = distrax.Independent(distrax.Normal(
            loc=mu, scale=sigma), reinterpreted_batch_ndims=1)
        sample = disribution.sample(seed=seed)
        return sample


def show_summary(Q, x):
    print(flax.linen.tabulate(Q, rng, compute_flops=True, compute_vjp_flops=True)(*x))


if __name__ == "__main__":
    x = jnp.ones((1, 4))
    show_summary(Q(4), [x])
    show_summary(Q_adv(4), [x])
    show_summary(V(), [x])
    show_summary(Q_SA(), [x, x])
    show_summary(Pi(4), [x])
    show_summary(Pi_cont(), [x])
