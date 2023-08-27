import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tqdm
import optax
import flax

key = jax.random.PRNGKey(0)


@jax.jit
def normalize(data):
    mean = jnp.mean(data, axis=0)
    std = jnp.std(data, axis=0)
    normalized_data = (data - mean) / std
    return normalized_data


class Dataset:
    def __init__(self):
        self.train, self.test = tf.keras.datasets.mnist.load_data()

    def get_data(self):

        @jax.jit
        def scale(data):
            data = jnp.array(data)
            return data/255.0

        X_train = scale(self.train[0].astype(np.float32)).reshape(-1, 784)
        y_train = jax.nn.one_hot(self.train[1], 10)
        X_test = scale(self.test[0].astype(np.float32)).reshape(-1, 784)
        y_test = jax.nn.one_hot(self.test[1], 10)
        return (X_train, y_train), (X_test, y_test)


class NN(flax.linen.Module):
    hidden_layers: list
    num_classes: int

    @flax.linen.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        for hidden_dim in self.hidden_layers:
            x = flax.linen.Dense(hidden_dim)(x)
            x = jax.nn.relu(x)
        x = flax.linen.Dense(self.num_classes)(x)
        return x


class NeuralNetwork:
    def __init__(self, hidden_layers=[128, 64], alpha=0.01, batch_size=512, n_epochs=1000):
        self.hidden_layers = hidden_layers
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.optimizer = optax.adam(self.alpha)

    def _init_values(self):
        self.model = NN(self.hidden_layers, num_classes=10)
        self.model_params = self.model.init(key, jnp.ones(
            (self.batch_size, 784)))

    def train(self, train, test):
        X, y = train
        X_test, y_test = test
        self.num_batches = len(X) // self.batch_size + 1
        self._init_values()
        self.optimizer_state = self.optimizer.init(self.model_params)

        @jax.jit
        def loss_fn(params, model, batch):
            inputs, targets = batch
            logits = model.apply(params, inputs)
            loss = nn.softmax_cross_entropy(logits, targets)
            return loss.mean()

        @jax.jit
        def train_step(optimizer, model_params, batch):
            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(model_params, self.model, batch)
            updates, new_optimizer_state = optimizer.update(
                grads, optimizer.target)
            new_model_params = optax.apply_updates(model_params, updates)
            return new_optimizer_state, new_model_params, loss

        for _ in tqdm.tqdm(range(self.n_epochs)):
            for i in range(self.num_batches):
                X_data = X[i * (self.batch_size):(i + 1) * self.batch_size]
                y_data = y[i * (self.batch_size):(i + 1) * self.batch_size]

                self.optimizer, self.model_params, loss = train_step(
                    self.optimizer, self.model_params, (X_data,y_data))

            # if _ % 50 == 0:
            #     train_loss = cross_entropy_loss(
            #         self.params, X_data, y_data)
            #     test_loss = cross_entropy_loss(
            #         self.params, X_test, y_test).block_until_ready()
            #     multi_test = accuracy(
            #         y_test, X_test, self.params).block_until_ready()
            #     print('Train Loss', train_loss, 'Test Loss',
            #           test_loss, 'Multi Class accuracy', round(multi_test * 100, 3), '%')

            # if train_loss <= 0.5 * 1e-4:
            #     break


if __name__ == '__main__':
    dataset = Dataset()
    train_data, test_data = dataset.get_data()

    nn = NeuralNetwork()
    nn.train(train_data, test_data)
