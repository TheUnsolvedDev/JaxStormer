import jax
import jax.numpy as jnp
import tqdm

key = jax.random.PRNGKey(0)


@jax.jit
def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


class Dataset:
    def __init__(self) -> None:
        self.num_data = 10000
        self.num_dims = 10
        self.weights = jnp.arange(1, 11)
        self.bias = 11

    def get_data(self):
        X = jax.random.normal(key, shape=(self.num_data, self.num_dims))
        logits = jnp.dot(X, self.weights).block_until_ready() + self.bias
        probabilities = sigmoid(logits)
        y = jax.random.bernoulli(key, probabilities)
        shuffled_indices = jax.random.permutation(
            key, len(X), independent=True)
        shuffled_data = X[shuffled_indices]
        shuffled_labels = y[shuffled_indices]

        X_train = shuffled_data[:int(self.num_data*0.8)]
        y_train = shuffled_labels[:int(self.num_data*0.8)]
        X_test = shuffled_data[int(self.num_data*0.8):]
        y_test = shuffled_labels[int(self.num_data*0.8):]

        return (X_train, y_train), (X_test, y_test)


class LogisticRegression:
    def __init__(self, alpha=0.01, batch_size=512, n_epochs=1000):
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def _init_values(self, X, y):
        self.init_weights = jnp.zeros(X.shape[1], dtype=jnp.float32)
        self.init_bias = 0.
        self.params = [self.init_weights, self.init_bias]

    def train(self, train, test):
        X, y = train
        X_test, y_test = test
        self._init_values(X, y)
        params = self.params

        @jax.jit
        def binary_cross_entropy(params, batch_X, batch_y):
            def nll(x, y):
                y_pred = sigmoid(jnp.dot(
                    x, params[0]) + params[1])
                y_true = y
                epsilon = 1e-10
                y_pred = jnp.clip(y_pred, epsilon, 1.0 - epsilon)
                loss = - (y_true * jnp.log(y_pred) +
                          (1 - y_true) * jnp.log(1 - y_pred))
                return loss
            return jnp.mean(jax.vmap(nll)(batch_X, batch_y), axis=0)

        @jax.jit
        def update_step(X_batch, y_batch, params, learning_rate):
            loss, grads = jax.value_and_grad(
                binary_cross_entropy)(params, X_batch, y_batch)
            params = jax.tree_util.tree_map(
                lambda p, g: p - learning_rate * g, params, grads)
            return params, loss

        @jax.jit
        def accuracy(y_true, x, params):
            y_pred = sigmoid(jnp.dot(x, params[0]) + params[1])
            y_pred_binary = (y_pred >= 0.5).astype(jnp.int32)
            correct_predictions = jnp.sum(y_true == y_pred_binary)
            total_predictions = y_true.shape[0]
            accuracy = correct_predictions / total_predictions
            return accuracy

        for _ in tqdm.tqdm(range(self.n_epochs)):
            if self.batch_size is None:
                params, loss = update_step(
                    X, y, params, self.alpha)
            else:
                self.num_batches = len(X) // self.batch_size + 1
                for i in range(self.num_batches):
                    X_data = X[i*(self.batch_size):(i+1)*self.batch_size]
                    y_data = y[i*(self.batch_size):(i+1)*self.batch_size]
                    params, loss = update_step(
                        X_data, y_data, params, self.alpha)

            if _ % 50 == 0:
                self.alpha *= 0.5
                train_loss = binary_cross_entropy(params, X, y)
                test_loss = binary_cross_entropy(params, X_test, y_test)
                train_accuracy = accuracy(y, X, params)
                test_accuracy = accuracy(y_test, X_test, params)
                print('Train Loss', train_loss, 'Test Loss', test_loss)
                print('Train Accuracy', round(train_accuracy*100, 3), '%',
                      'Test Accuracy', round(test_accuracy*100, 3), '%')

            if jnp.abs(loss) <= 0.5*1e-2:
                print(params)
                break


if __name__ == '__main__':
    dataset = Dataset()
    train, test = dataset.get_data()

    lr_model = LogisticRegression(batch_size=None)
    lr_model.train(train, test)
