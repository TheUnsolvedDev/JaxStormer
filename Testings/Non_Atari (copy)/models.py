from import_packages import *

key = jax.random.PRNGKey(0)


class Q_model_small(flax.linen.Module):
    action_dim: int
    activation: str = 'relu'

    @flax.linen.compact
    def __call__(self, x: jnp.ndarray):
        if self.activation == 'relu':
            act = jax.nn.relu
        elif self.activation == 'selu':
            act = jax.nn.selu
        elif self.activation == 'gelu':
            act = jax.nn.gelu
        else:
            act = jax.nn.tanh

        x = flax.linen.Dense(16)(x)
        x = act(x)
        x = flax.linen.Dense(16)(x)
        x = act(x)
        x = flax.linen.Dense(self.action_dim)(x)
        return x


class Q_model_medium(flax.linen.Module):
    action_dim: int
    activation: str = 'relu'

    @flax.linen.compact
    def __call__(self, x: jnp.ndarray):
        if self.activation == 'relu':
            act = jax.nn.relu
        elif self.activation == 'selu':
            act = jax.nn.selu
        elif self.activation == 'gelu':
            act = jax.nn.gelu
        else:
            act = jax.nn.tanh

        x = flax.linen.Dense(32)(x)
        x = act(x)
        x = flax.linen.Dense(64)(x)
        x = act(x)
        x = flax.linen.Dense(self.action_dim)(x)
        return x


class Q_model_large(flax.linen.Module):
    action_dim: int
    activation: str = 'relu'

    @flax.linen.compact
    def __call__(self, x: jnp.ndarray):
        if self.activation == 'relu':
            act = jax.nn.relu
        elif self.activation == 'selu':
            act = jax.nn.selu
        elif self.activation == 'gelu':
            act = jax.nn.gelu
        else:
            act = jax.nn.tanh

        x = flax.linen.Dense(128)(x)
        x = act(x)
        x = flax.linen.Dense(128)(x)
        x = act(x)
        x = flax.linen.Dense(self.action_dim)(x)
        return x


if __name__ == '__main__':
    q_network = Q_model_small(10)
    print(q_network.tabulate(key, jnp.ones((128, 5))))
