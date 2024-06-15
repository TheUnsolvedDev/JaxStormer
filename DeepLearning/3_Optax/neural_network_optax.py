import import_packages
from import_packages import *

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


class NeuralNetwork:
    def __init__(
        self,
        hidden_layers: List[int] = [64, 64],
        alpha: float = 0.01,
        batch_size: int = 512,
        n_epochs: int = 1000,
    ) -> None:
        """
        Initialize the NeuralNetwork class.

        Args:
            hidden_layers (List[int], optional): The list of number of neurons in each hidden layer. Defaults to [64, 64].
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
        self.optimizer = optax.adam(self.alpha)


    def _init_values(self, X: jnp.ndarray, y: jnp.ndarray) -> List[jnp.ndarray]:
        """
        Initialize the values of the neural network.

        Args:
            X (jnp.ndarray): The input data.
            y (jnp.ndarray): The target data.

        Returns:
            List[jnp.ndarray]: The initialized parameters.
        """
        keys = jax.random.split(
            key, num=2*(len(self.hidden_layers)+1)+1)
        input_dim = X.shape[1]
        output_dim = y.shape[1]

        self.params: List[jnp.ndarray] = [
            jax.random.normal(keys[1], (input_dim, self.hidden_layers[0])),
            jax.random.normal(keys[2], (self.hidden_layers[0],))
        ]
        for i in range(1, len(self.hidden_layers)):
            self.params.extend([
                jax.random.normal(keys[2*i], (self.hidden_layers[i-1], self.hidden_layers[i])),
                jax.random.normal(keys[2*i+1], (self.hidden_layers[i],))
            ])
        self.params.extend([
            jax.random.normal(keys[-2], (self.hidden_layers[-1], output_dim)),
            jax.random.normal(keys[-1], (output_dim,))
        ])
        return self.params

    def train(self, train: Tuple[jnp.ndarray, jnp.ndarray],
              test: Tuple[jnp.ndarray, jnp.ndarray]) -> None:
        """
        Train the neural network.

        Args:
            train (Tuple[jnp.ndarray, jnp.ndarray]): The training data.
            test (Tuple[jnp.ndarray, jnp.ndarray]): The test data.

        Returns:
            None
        """
        X, y = train
        X_test, y_test = test
        self._init_values(X, y)
        params: List[jnp.ndarray] = self.params
        optimizer_state = self.optimizer.init(params)

        @jax.jit
        def neural_network(params: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
            """
            Compute the output of the neural network.

            Args:
                params (jnp.ndarray): The model parameters.
                x (jnp.ndarray): The input data.

            Returns:
                jnp.ndarray: The output of the neural network.
            """
            num_hidden_layers = len(params) // 2
            hidden = x
            for i in range(num_hidden_layers):
                w = params[2 * i]
                b = params[2 * i + 1]
                if i <= num_hidden_layers - 2:
                    hidden = jax.nn.tanh(jnp.dot(hidden, w) + b)
                else:
                    hidden = jnp.dot(hidden, w) + b
            return jax.nn.softmax(hidden)

        @jax.jit
        def cross_entropy_loss(params: jnp.ndarray, batch_x: jnp.ndarray,
                               batch_y: jnp.ndarray) -> jnp.ndarray:
            """
            Calculate the cross-entropy loss for a batch of data.

            Args:
                params (jnp.ndarray): The model parameters.
                batch_x (jnp.ndarray): The input data.
                batch_y (jnp.ndarray): The target data.

            Returns:
                jnp.ndarray: The mean loss.
            """
            def nll(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
                probs = neural_network(params, x)
                softmax_logits = jnp.log(probs)
                loss = -jnp.sum(softmax_logits * y)
                return loss
            return jnp.mean(jax.vmap(nll)(batch_x, batch_y), axis=0)

        @jax.jit
        def update_step(x_batch: jnp.ndarray, y_batch: jnp.ndarray,
                        opt_state: optax.OptState, params: jnp.ndarray) -> Tuple[optax.OptState, jnp.ndarray, jnp.ndarray]:
            """
            Update the model parameters based on the gradient of the loss.

            Args:
                x_batch (jnp.ndarray): The input data for a batch.
                y_batch (jnp.ndarray): The target data for a batch.
                opt_state (optax.OptState): The optimizer state.
                params (jnp.ndarray): The model parameters.

            Returns:
                Tuple[optax.OptState, jnp.ndarray, jnp.ndarray]: The updated optimizer state, model parameters, and loss.
            """
            loss, grads = jax.value_and_grad(
                cross_entropy_loss)(params, x_batch, y_batch)
            updates, opt_state = self.optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return opt_state, params, loss

        @ jax.jit
        def accuracy(y_true: jnp.ndarray, x: jnp.ndarray,
                     params: jnp.ndarray) -> jnp.ndarray:
            """
            Calculate the accuracy of the model.

            Args:
                y_true (jnp.ndarray): The true labels.
                x (jnp.ndarray): The input data.
                params (jnp.ndarray): The model parameters.

            Returns:
                jnp.ndarray: The accuracy of the model.
            """
            y_pred = neural_network(params, x)
            predicted_classes = jnp.argmax(y_pred, axis=-1)
            true_classes = y_true
            true_classes = jnp.argmax(true_classes, axis=-1)
            correct_predictions = jnp.sum(predicted_classes == true_classes)
            total_samples = y_true.shape[0]
            acc = correct_predictions / total_samples
            return acc

        for _ in tqdm.tqdm(range(self.n_epochs)):
            if self.batch_size is None:
                optimizer_state, params, loss = update_step(
                    X, y, optimizer_state, params)
            else:
                self.num_batches = len(X) // self.batch_size + 1
                for i in range(self.num_batches):
                    X_data = X[i*(self.batch_size):(i+1)*self.batch_size]
                    y_data = y[i*(self.batch_size):(i+1)*self.batch_size]
                    optimizer_state, params, loss = update_step(
                        X_data, y_data, optimizer_state, params)

            if _ % 50 == 0:
                train_loss = cross_entropy_loss(params, X, y)
                test_loss = cross_entropy_loss(params, X_test, y_test)
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
    train_data, test_data = dataset.get_data()

    nn = NeuralNetwork(batch_size=None)
    nn.train(train_data, test_data)
