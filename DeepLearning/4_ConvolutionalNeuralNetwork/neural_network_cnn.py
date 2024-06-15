from import_packages import *
import import_packages
key = jax.random.PRNGKey(0)


class Dataset:
    def __init__(self) -> None:
        """
        Initialize the Dataset class.

        Args:
            None

        Returns:
            None
        """
        import_packages.init()
        self.train, self.test = import_packages.load()

    def get_data(self) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Get the training and testing data.

        Returns:
            Tuple[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]: The scaled training and testing data.
        """
        @jax.jit
        def scale(data: np.ndarray) -> jnp.ndarray:
            """
            Scale the data by dividing it by 255.0.

            Args:
                data (np.ndarray): The input data.

            Returns:
                jnp.ndarray: The scaled data.
            """
            data = jnp.array(data)
            return data/255.0

        X_train = scale(self.train[0].astype(np.float32)).reshape(-1, 28, 28, 1)
        y_train = jax.nn.one_hot(self.train[1], 10)
        X_test = scale(self.test[0].astype(np.float32)).reshape(-1, 28, 28, 1)
        y_test = jax.nn.one_hot(self.test[1], 10)
        return (X_train, y_train), (X_test, y_test)


class CNN(flax.linen.Module):

    @flax.linen.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Apply the convolutional neural network to the input data.

        Args:
            x (jnp.ndarray): The input data, of shape (batch_size, height, width, channels).

        Returns:
            jnp.ndarray: The output of the neural network, of shape (batch_size, num_classes).
        """
        # First convolutional layer
        x = flax.linen.Conv(features=6, kernel_size=(5, 5), strides=2)(x)
        x = flax.linen.relu(x)
        x = flax.linen.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Second convolutional layer
        x = flax.linen.Conv(
            features=16, kernel_size=(5, 5), strides=2)(x)
        x = flax.linen.relu(x)
        x = flax.linen.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Flatten the output
        x = jnp.reshape(x, (x.shape[0], -1))
        x = jnp.squeeze(x)

        # Dense layers
        x = flax.linen.Dense(features=120)(x)
        x = flax.linen.relu(x)

        x = flax.linen.Dense(features=10)(x)
        return x


class NeuralNetwork:
    def __init__(
        self,
        alpha: float = 0.01,  # learning rate
        batch_size: int = 512,  # size of the mini-batches for training
        n_epochs: int = 1000,  # number of training epochs
    ) -> None:
        """
        Initialize the NeuralNetwork class.

        Args:
            alpha (float, optional): The learning rate. Defaults to 0.01.
            batch_size (int, optional): The size of the mini-batches for training. Defaults to 512.
            n_epochs (int, optional): The number of training epochs. Defaults to 1000.

        Returns:
            None
        """
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        linear_decay_scheduler = optax.linear_schedule(
            init_value=alpha,
            end_value=0.001,
            transition_steps=n_epochs,
            transition_begin=int(n_epochs * 0.2),
        )
        self.optimizer = optax.adam(learning_rate=linear_decay_scheduler)

    def _init_values(self) -> None:
        """
        Initialize the model and model parameters.

        Args:
            self: The instance of the NeuralNetwork class.

        Returns:
            None
        """
        self.model = CNN()
        self.model_params = self.model.init(
            key=key,
            input_shape=(1, 28, 28, 1)
        )  # type: flax.core.frozen_dict.FrozenDict[str, jnp.ndarray]
        print(self.model.tabulate(
            key=key,
            input_shape=(1, 28, 28, 1)
        ))

    def train(self, train: Tuple[jnp.ndarray, jnp.ndarray],
              test: Tuple[jnp.ndarray, jnp.ndarray]) -> None:
        """
        Train the neural network model.

        Args:
            train (Tuple[jnp.ndarray, jnp.ndarray]): The training data.
            test (Tuple[jnp.ndarray, jnp.ndarray]): The test data.

        Returns:
            None
        """
        X, y = train
        X_test, y_test = test
        self._init_values()

        state, params = flax.core.pop(self.model_params, 'params')
        optimizer_state = self.optimizer.init(params)

        @jax.jit
        def cross_entropy_loss(params: jnp.ndarray, batch_X: jnp.ndarray,
                                batch_Y: jnp.ndarray,
                                state: MutableMapping[str, jnp.ndarray]
                                ) -> Tuple[jnp.ndarray, MutableMapping[str, jnp.ndarray]]:
            """
            Calculate the cross entropy loss for a batch of data.

            Args:
                params (jnp.ndarray): The model parameters.
                batch_X (jnp.ndarray): The input data.
                batch_Y (jnp.ndarray): The target data.
                state (MutableMapping[str, jnp.ndarray]): The model state.

            Returns:
                Tuple[jnp.ndarray, MutableMapping[str, jnp.ndarray]]: The mean loss and updated state.
            """
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
        def update_step(x_batch: jnp.ndarray, y_batch: jnp.ndarray,
                        opt_state: optax.OptState, params: jnp.ndarray,
                        state: MutableMapping[str, jnp.ndarray]
                        ) -> Tuple[optax.OptState, jnp.ndarray,
                                   MutableMapping[str, jnp.ndarray], jnp.ndarray]:
            """
            Update the model parameters based on the gradient of the loss.

            Args:
                x_batch (jnp.ndarray): The input data for a batch.
                y_batch (jnp.ndarray): The target data for a batch.
                opt_state (optax.OptState): The optimizer state.
                params (jnp.ndarray): The model parameters.
                state (MutableMapping[str, jnp.ndarray]): The model state.

            Returns:
                Tuple[optax.OptState, jnp.ndarray, MutableMapping[str, jnp.ndarray], jnp.ndarray]:
                    The updated optimizer state, model parameters, updated state, and loss.
            """
            loss, grads = jax.value_and_grad(
                cross_entropy_loss)(params, x_batch, y_batch, state)
            updates, opt_state = self.optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return opt_state, params, state, loss

        @jax.jit
        def accuracy(y_true: jnp.ndarray, x: jnp.ndarray,
                     params: MutableMapping[str, jnp.ndarray]
                     ) -> jnp.ndarray:
            """
            Calculate the accuracy of the model on the given data.

            Args:
                y_true (jnp.ndarray): The true labels.
                x (jnp.ndarray): The input data.
                params (MutableMapping[str, jnp.ndarray]): The model parameters.

            Returns:
                jnp.ndarray: The accuracy of the model on the given data.
            """
            y_pred = self.model.apply(params, x)
            predicted_classes = jnp.argmax(y_pred, axis=-1)
            true_classes = jnp.argmax(y_true, axis=-1)
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
