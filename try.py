import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tqdm
import optax
import flax.linen as nn

key = jax.random.PRNGKey(0)

class NeuralNetwork(nn.Module):
    hidden_layers: list
    num_classes: int

    def setup(self):
        self.layers = []
        for hidden_units in self.hidden_layers:
            self.layers.append(nn.Dense(hidden_units))
            self.layers.append(nn.tanh)
        self.layers.append(nn.Dense(self.num_classes))
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


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
# ... (same as your code)

if __name__ == '__main__':
    dataset = Dataset()
    train_data, test_data = dataset.get_data()

    hidden_layers = [64, 64]
    num_classes = 10
    alpha = 0.01
    batch_size = 512
    n_epochs = 1000

    model = NeuralNetwork(hidden_layers=hidden_layers, num_classes=num_classes)
    optimizer = optax.adam(alpha)
    rng = jax.random.PRNGKey(0)

    @jax.jit
    def cross_entropy_loss(params, x, y):
        logits = model.apply(params, x)
        softmax_logits = jnp.log(jax.nn.softmax(logits))
        loss = -jnp.sum(softmax_logits * y)
        return loss

    @jax.jit
    def accuracy(y_true, x, params):
        logits = model.apply(params, x)
        predicted_classes = jnp.argmax(logits, axis=-1)
        true_classes = jnp.argmax(y_true, axis=-1)
        correct_predictions = jnp.sum(predicted_classes == true_classes)
        total_samples = y_true.shape[0]
        acc = correct_predictions / total_samples
        return acc

    @jax.jit
    def train_step(params, optimizer_state, batch_X, batch_y):
        gradient_fn = jax.grad(cross_entropy_loss)
        grads = gradient_fn(params, batch_X, batch_y)
        updates, optimizer_state = optimizer.update(grads, optimizer_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, optimizer_state

    model_state = model.init(rng, jnp.ones((batch_size, 784)))
    optimizer_state = optimizer.init(model_state)

    num_batches = len(train_data[0]) // batch_size + 1

    for _ in tqdm.tqdm(range(n_epochs)):
        for i in range(num_batches):
            X_data = train_data[0][i * batch_size: (i + 1) * batch_size]
            y_data = train_data[1][i * batch_size: (i + 1) * batch_size]

            model_state, optimizer_state = train_step(
                model_state, optimizer_state, X_data, y_data)

        if _ % 50 == 0:
            train_loss = cross_entropy_loss(
                model_state, X_data, y_data)
            test_loss = cross_entropy_loss(
                model_state, test_data[0], test_data[1]).block_until_ready()
            multi_test = accuracy(
                test_data[1], test_data[0], model_state).block_until_ready()
            print('Train Loss', train_loss, 'Test Loss',
                  test_loss, 'Multi Class accuracy', round(multi_test * 100, 3), '%')

        if train_loss <= 0.5 * 1e-4:
            break
