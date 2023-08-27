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

    def setup(self):
        self.layers = [flax.linen.Dense(
            i) for i in self.hidden_layers]

    @flax.linen.compact
    def __call__(self, x):
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            x = flax.linen.relu(x)
        x = flax.linen.Dense(self.num_classes)(x)
        return x


class NeuralNetwork:
    def __init__(self, hidden_layers=[128, 64], alpha=0.01, batch_size=512, n_epochs=1000):
        self.hidden_layers = hidden_layers
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        linear_decay_scheduler = optax.linear_schedule(init_value=alpha, end_value=0.001,
                                                       transition_steps=n_epochs,
                                                       transition_begin=int(n_epochs*0.2))
        self.optimizer = optax.adam(learning_rate=linear_decay_scheduler)

    def _init_values(self):
        self.model = NN(self.hidden_layers, num_classes=10)
        self.model_params = self.model.init(key, jnp.ones(
            (1, 784)))
        self.model.tabulate(key, jnp.ones(
            (1, 784)))

    def train(self, train, test):
        X, y = train
        X_test, y_test = test
        self._init_values()

        state, params = flax.core.pop(self.model_params, 'params')
        optimizer_state = self.optimizer.init(params)

        @jax.jit
        def batch_loss(params, batch_X, batch_Y, state):
            def loss_fn(x, y):
                pred, updated_state = self.model.apply(
                    {'params': params, **state},
                    x, mutable=list(state.keys())
                )
                loss = optax.softmax_cross_entropy(pred, y)
                return loss, updated_state

            loss, updated_state = jax.vmap(
                loss_fn, out_axes=(0, None),
                axis_name='batch'
            )(batch_X, batch_Y)
            return jnp.mean(loss), updated_state

        @jax.jit
        def update_step(x_batch, y_batch, opt_state, params, state):
            (loss, updated_state), grads = jax.value_and_grad(
                batch_loss, has_aux=True
            )(params, x_batch, y_batch, state)
            updates, opt_state = self.optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return opt_state, params, updated_state, loss

        @ jax.jit
        def accuracy(y_true, x, params):
            y_pred = self.model.apply(params, x)
            predicted_classes = jnp.argmax(y_pred, axis=-1)
            true_classes = y_true
            true_classes = jnp.argmax(true_classes, axis=-1)
            correct_predictions = jnp.sum(predicted_classes == true_classes)
            total_samples = y_true.shape[0]
            acc = correct_predictions / total_samples
            return acc

        for _ in tqdm.tqdm(range(self.n_epochs+1)):
            if self.batch_size is None:
                optimizer_state, params, state, loss = update_step(
                    X, y, optimizer_state, params, state)
            else:
                self.num_batches = len(X) // self.batch_size + 1
                for i in range(self.num_batches):
                    X_data = X[i*(self.batch_size):(i+1)*self.batch_size]
                    y_data = y[i*(self.batch_size):(i+1)*self.batch_size]
                    optimizer_state, params, state, loss = update_step(
                        X_data, y_data, optimizer_state, params, state)

            if _ % 50 == 0:
                train_loss = optax.softmax_cross_entropy(
                    self.model.apply({'params': params}, X), y).mean()
                test_loss = optax.softmax_cross_entropy(
                    self.model.apply({'params': params}, X_test), y_test).mean()
                multi_test = accuracy(
                    y_test, X_test, {'params': params}).block_until_ready()
                print('Train Loss', train_loss, 'Test Loss',
                      test_loss, 'Multi Class accuracy', round(multi_test * 100, 3), '%')

            if train_loss <= 0.5 * 1e-4:
                break


if __name__ == '__main__':
    dataset = Dataset()
    train_data, test_data = dataset.get_data()

    nn = NeuralNetwork(batch_size=None)
    nn.train(train_data, test_data)
