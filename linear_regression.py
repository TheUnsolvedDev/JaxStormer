import jax
import jax.numpy as jnp
import tqdm

key = jax.random.PRNGKey(0)


class Dataset:
    def __init__(self, train=0.8) -> None:
        self.num_data = 10000
        self.num_dims = 10
        self.weights = jnp.arange(1, 11)
        self.bias = 11

    def get_data(self):
        X = jax.random.normal(key, shape=(self.num_data, self.num_dims))
        y = jnp.dot(X, self.weights).block_until_ready() + self.bias

        shuffled_indices = jax.random.permutation(
            key, len(X), independent=True)
        shuffled_data = X[shuffled_indices]
        shuffled_labels = y[shuffled_indices]

        X_train = shuffled_data[:int(self.num_data*0.8)]
        y_train = shuffled_labels[:int(self.num_data*0.8)].reshape(-1, 1)
        X_test = shuffled_data[int(self.num_data*0.8):]
        y_test = shuffled_labels[int(self.num_data*0.8):].reshape(-1, 1)

        return (X_train, y_train), (X_test, y_test)


class LinearRegression:
    def __init__(self, alpha=0.01, batch_size=1024, n_epochs=1000):
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

        @jax.jit
        def mse(params, batch_X, batch_y):
            def squared_error(x, y):
                pred = jnp.dot(
                    x, params[0]) + params[1]
                return jnp.inner(y-pred, y-pred) / 2.0
            return jnp.mean(jax.vmap(squared_error)(batch_X, batch_y), axis=0)

        @jax.jit
        def update_params(params, learning_rate, grads):
            params = jax.tree_util.tree_map(
                lambda p, g: p - learning_rate * g, params, grads)
            return params

        loss_grad_fn = jax.value_and_grad(mse)
        self.num_batches = len(X)//self.batch_size + 1

        for _ in tqdm.tqdm(range(self.n_epochs)):
            for i in range(self.num_batches):
                X_data = X[i*(self.batch_size):(i+1)*self.batch_size]
                y_data = y[i*(self.batch_size):(i+1)*self.batch_size]
                loss_val, grads = loss_grad_fn(self.params, X_data, y_data)
                self.params = update_params(self.params, self.alpha, grads)

            if _ % 100 == 0:
                self.alpha = self.alpha*0.5
                test_loss, _ = loss_grad_fn(self.params, X_test, y_test)
                print('Train Loss', loss_val, 'Test Loss', test_loss)

            if loss_val <= 0.5*1e-4:
                print(self.params)
                break


if __name__ == "__main__":
    dataset = Dataset()
    train, test = dataset.get_data()

    lr_model = LinearRegression()
    lr_model.train(train, test)
