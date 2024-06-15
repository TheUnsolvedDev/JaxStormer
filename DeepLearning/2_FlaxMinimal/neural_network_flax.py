from import_packages import *
import import_packages
key = jax.random.PRNGKey(0)


def normalize(data: jnp.ndarray) -> jnp.ndarray:
    """
    Normalize the data by subtracting the mean and dividing by the standard deviation along the first axis.

    Args:
        data (jnp.ndarray): The input data.

    Returns:
        jnp.ndarray: The normalized data.
    """
    mean = jnp.mean(data, axis=0)
    std = jnp.std(data, axis=0)
    normalized_data = (data - mean) / std
    return normalized_data


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

        X_train = scale(self.train[0].astype(np.float32)).reshape(-1, 784)
        y_train = jax.nn.one_hot(self.train[1], 10)
        X_test = scale(self.test[0].astype(np.float32)).reshape(-1, 784)
        y_test = jax.nn.one_hot(self.test[1], 10)
        return (X_train, y_train), (X_test, y_test)


class NN(flax.linen.Module):
    hidden_layers: list
    num_classes: int

    def setup(self, hidden_layers: List[int], num_classes: int) -> None:
        """
        Set up the layers of the neural network.

        Args:
            hidden_layers (List[int]): The list of number of neurons in each hidden layer.
            num_classes (int): The number of classes.

        Returns:
            None
        """
        self.hidden_layers = hidden_layers
        self.num_classes = num_classes
        self.layers = [flax.linen.Dense(i) for i in self.hidden_layers]

    @flax.linen.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass of the neural network.

        Args:
            x (jnp.ndarray): The input data.

        Returns:
            jnp.ndarray: The output of the neural network.
        """
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            x = flax.linen.relu(x)
        x = flax.linen.Dense(self.num_classes)(x)
        return x


class NeuralNetwork:
    def __init__(
        self,
        hidden_layers: List[int] = [128, 64],
        alpha: float = 0.01,
        batch_size: int = 512,
        n_epochs: int = 1000,
    ) -> None:
        """
        Initialize the NeuralNetwork class.

        Args:
            hidden_layers (List[int], optional): The list of number of neurons in each hidden layer. Defaults to [128, 64].
            alpha (float, optional): The learning rate. Defaults to 0.01.
            batch_size (int, optional): The batch size. Defaults to 512.
            n_epochs (int, optional): The number of epochs. Defaults to 1000.

        Returns:
            None
        """
        self.hidden_layers = hidden_layers
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
        self.model = NN(self.hidden_layers, num_classes=10)
        self.model_params = self.model.init(
            key=key,
            input_shape=(1, 784)
        )  # type: flax.core.frozen_dict.FrozenDict[str, jnp.ndarray]
        print(self.model.tabulate(
            key=key,
            input_shape=(1, 784)
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
                        opt_state: Mapping[str, jnp.ndarray],
                        params: jnp.ndarray,
                        state: MutableMapping[str, jnp.ndarray]
                        ) -> Tuple[Mapping[str, jnp.ndarray], jnp.ndarray,
                                    MutableMapping[str, jnp.ndarray], jnp.ndarray]:
            """
            Perform a single update step.

            Args:
                x_batch (jnp.ndarray): The input data for the batch.
                y_batch (jnp.ndarray): The target data for the batch.
                opt_state (Mapping[str, jnp.ndarray]): The optimizer state.
                params (jnp.ndarray): The model parameters.
                state (MutableMapping[str, jnp.ndarray]): The model state.

            Returns:
                Tuple[Mapping[str, jnp.ndarray], jnp.ndarray,
                      MutableMapping[str, jnp.ndarray], jnp.ndarray]:
                    The updated optimizer state, model parameters, updated state, and loss.
            """
            (loss, updated_state), grads = jax.value_and_grad(
                cross_entropy_loss, has_aux=True
            )(params, x_batch, y_batch, state)
            updates, opt_state = self.optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return opt_state, params, updated_state, loss

        @ jax.jit
        def accuracy(y_true: jnp.ndarray, x: jnp.ndarray,
                     params: jnp.ndarray
                     ) -> jnp.ndarray:
            """
            Calculate the accuracy of the model.

            Args:
                y_true (jnp.ndarray): The true labels.
                x (jnp.ndarray): The input data.
                params (jnp.ndarray): The model parameters.

            Returns:
                jnp.ndarray: The accuracy.
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

    nn = NeuralNetwork(batch_size=None)
    nn.train(train_data, test_data)
