from import_packages import *
key = jax.random.PRNGKey(0)


class Dataset:
    def __init__(self):
        self.train, self.test = tf.keras.datasets.mnist.load_data()

    def get_data(self):

        @jax.jit
        def scale(data):
            data = jnp.array(data)
            return data/255.0

        X_train = scale(self.train[0].astype(
            np.float32)).reshape(-1, 28, 28, 1)
        y_train = jax.nn.one_hot(self.train[1], 10)
        X_test = scale(self.test[0].astype(np.float32)).reshape(-1, 28, 28, 1)
        y_test = jax.nn.one_hot(self.test[1], 10)
        return (X_train, y_train), (X_test, y_test)


class LeNet5(flax.linen.Module):

    @flax.linen.compact
    def __call__(self, x):
        x = flax.linen.Conv(features=6, kernel_size=(5, 5), strides=2)(x)
        x = flax.linen.relu(x)
        x = flax.linen.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = flax.linen.Conv(
            features=16, kernel_size=(5, 5), strides=2)(x)
        x = flax.linen.relu(x)
        x = flax.linen.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = jnp.reshape(x, (x.shape[0], -1))
        x = jnp.squeeze(x)

        x = flax.linen.Dense(features=120)(x)
        x = flax.linen.relu(x)

        x = flax.linen.Dense(features=84)(x)
        x = flax.linen.relu(x)

        x = flax.linen.Dense(features=10)(x)
        return x


class NeuralNetwork:
    def __init__(self, alpha=0.01, batch_size=512, n_epochs=1000):
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        linear_decay_scheduler = optax.linear_schedule(init_value=alpha, end_value=0.001,
                                                       transition_steps=n_epochs,
                                                       transition_begin=int(n_epochs*0.2))
        self.optimizer = optax.adam(learning_rate=linear_decay_scheduler)

    def _init_values(self):
        self.model = LeNet5()
        self.model_params = self.model.init(key, jnp.ones(
            (1, 28, 28, 1)))
        print(self.model.tabulate(key, jnp.ones(
            (1, 28, 28, 1))))

    def train(self, train, test):
        X, y = train
        X_test, y_test = test
        self._init_values()

        state, params = flax.core.pop(self.model_params, 'params')
        optimizer_state = self.optimizer.init(params)

        @jax.jit
        def cross_entropy_loss(params, batch_X, batch_Y, state):
            def nll(x, y):
                pred, updated_state = self.model.apply(
                    {'params': params, **state},
                    x, mutable=list(state.keys())
                )
                loss = optax.softmax_cross_entropy(pred, y)
                return loss, updated_state

            loss, updated_state = jax.vmap(
                nll, out_axes=(0, None),
                axis_name='batch'
            )(batch_X, batch_Y)
            return jnp.mean(loss), updated_state

        @jax.jit
        def update_step(x_batch, y_batch, opt_state, params, state):
            (loss, updated_state), grads = jax.value_and_grad(
                cross_entropy_loss, has_aux=True
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
                X = jnp.expand_dims(X, axis=1)
                optimizer_state, params, state, loss = update_step(
                    X, y, optimizer_state, params, state)
            else:
                self.num_batches = len(X) // self.batch_size + 1
                for i in range(self.num_batches):
                    X_data = jnp.expand_dims(
                        X[i*(self.batch_size):(i+1)*self.batch_size], axis=1)
                    y_data = y[i*(self.batch_size):(i+1)*self.batch_size]
                    optimizer_state, params, state, loss = update_step(
                        X_data, y_data, optimizer_state, params, state)

            if _ % 50 == 0:
                train_loss = optax.softmax_cross_entropy(
                    self.model.apply({'params': params}, X), y).mean()
                test_loss = optax.softmax_cross_entropy(
                    self.model.apply({'params': params}, X_test), y_test).mean()
                train_accuracy = accuracy(
                    y, X, {'params': params}).block_until_ready()
                test_accuracy = accuracy(
                    y_test, X_test, {'params': params}).block_until_ready()
                print('Train Loss', train_loss, 'Test Loss', test_loss)
                print('Train Accuracy', round(train_accuracy*100, 3), '%',
                      'Test Accuracy', round(test_accuracy*100, 3), '%')

            if train_loss <= 0.5 * 1e-4:
                break


if __name__ == '__main__':
    dataset = Dataset()
    train_data, test_data = dataset.get_data()

    nn = NeuralNetwork()
    nn.train(train_data, test_data)
