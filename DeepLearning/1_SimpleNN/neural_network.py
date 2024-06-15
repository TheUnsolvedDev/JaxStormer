import import_packages
from import_packages import *

key = jax.random.PRNGKey(0)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

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

        X_train = scale(self.train[0].astype(jnp.float32)).reshape(-1, 784)
        y_train = jax.nn.one_hot(self.train[1], 10)
        X_test = scale(self.test[0].astype(jnp.float32)).reshape(-1, 784)
        y_test = jax.nn.one_hot(self.test[1], 10)
        return (X_train, y_train), (X_test, y_test)


class NeuralNetwork:
    def __init__(
        self,
        hidden_layers: List[int] = [64, 64],
        alpha: float = 0.1,
        batch_size: int = 512,
        n_epochs: int = 10000
    ) -> None:
        """
        Initialize the NeuralNetwork class.

        Args:
            hidden_layers (List[int]): The hidden layers of the neural network.
            alpha (float): The learning rate.
            batch_size (int): The batch size.
            n_epochs (int): The number of epochs.

        Returns:
            None
        """
        self.hidden_layers: List[int] = hidden_layers
        self.alpha: float = alpha
        self.n_epochs: int = n_epochs
        self.batch_size: int = batch_size

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
        params = self.params

        @jax.jit
        def neural_network(params: List[jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
            """
            Perform a forward pass through the neural network.

            Args:
                params (List[jnp.ndarray]): The network parameters.
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
        def cross_entropy_loss(params: List[jnp.ndarray],
                               batch_x: jnp.ndarray,
                               batch_y: jnp.ndarray) -> jnp.ndarray:
            """
            Compute the cross entropy loss.

            Args:
                params (List[jnp.ndarray]): The network parameters.
                batch_x (jnp.ndarray): The input data.
                batch_y (jnp.ndarray): The target data.

            Returns:
                jnp.ndarray: The cross entropy loss.
            """
            def nll(x, y):
                probs = neural_network(params, x)
                softmax_logits = jnp.log(probs)
                loss = -jnp.sum(softmax_logits * y)
                return loss
            return jnp.mean(jax.vmap(nll)(batch_x, batch_y), axis=0)

        @jax.jit
        def update_step(X_batch: jnp.ndarray,
                        y_batch: jnp.ndarray,
                        params: List[jnp.ndarray],
                        learning_rate: float) -> Tuple[List[jnp.ndarray], jnp.ndarray]:
            """
            Perform an update step of the neural network.

            Args:
                X_batch (jnp.ndarray): The input data batch.
                y_batch (jnp.ndarray): The target data batch.
                params (List[jnp.ndarray]): The network parameters.
                learning_rate (float): The learning rate.

            Returns:
                Tuple[List[jnp.ndarray], jnp.ndarray]: The updated parameters and the loss.
            """
            loss, grads = jax.value_and_grad(
                cross_entropy_loss)(params, X_batch, y_batch)
            params = jax.tree_util.tree_map(
                lambda p, g: p - learning_rate * g, params, grads)
            return params, loss

        @jax.jit
        def accuracy(y_true: jnp.ndarray,
                     x: jnp.ndarray,
                     params: List[jnp.ndarray]) -> float:
            """
            Compute the accuracy of the neural network.

            Args:
                y_true (jnp.ndarray): The true labels.
                x (jnp.ndarray): The input data.
                params (List[jnp.ndarray]): The network parameters.

            Returns:
                float: The accuracy of the neural network.
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
                self.alpha = self.alpha
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
