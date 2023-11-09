import import_packages
from import_packages import *

key = jax.random.PRNGKey(0)


@jax.jit
def normalize(data):
    mean = jnp.mean(data, axis=0)
    std = jnp.std(data, axis=0)
    normalized_data = (data - mean) / std
    return normalized_data


class Dataset:
    def __init__(self):
        import_packages.init()
        self.train, self.test = import_packages.load()

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


class NeuralNetwork:
    def __init__(self, hidden_layers=[64, 64], alpha=0.01, batch_size=512, n_epochs=1000):
        self.hidden_layers = hidden_layers
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.optimizer = optax.adam(self.alpha)

    def _init_values(self, X, y):
        keys = jax.random.split(
            key, num=2*(len(self.hidden_layers)+1)+1)
        input_dim = X.shape[1]
        output_dim = y.shape[1]

        self.params = [jax.random.normal(
            keys[1], (input_dim, self.hidden_layers[0])), jax.random.normal(keys[2], (self.hidden_layers[0],))]
        for i in range(1, len(self.hidden_layers)):
            self.params.extend([jax.random.normal(
                keys[2*i], (self.hidden_layers[i-1], self.hidden_layers[i])), jax.random.normal(keys[2*i+1], (self.hidden_layers[i],))])
        self.params.extend([jax.random.normal(
            keys[-2], (self.hidden_layers[-1], output_dim)), jax.random.normal(keys[-1], (output_dim,))])
        return self.params

    def train(self, train, test):
        X, y = train
        X_test, y_test = test
        self._init_values(X, y)
        params = self.params
        optimizer_state = self.optimizer.init(params)

        @jax.jit
        def neural_network(params, x):
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

        @ jax.jit
        def cross_entropy_loss(params, batch_x, batch_y):
            def nll(x, y):
                probs = neural_network(params, x)
                softmax_logits = jnp.log(probs)
                loss = -jnp.sum(softmax_logits * y)
                return loss
            return jnp.mean(jax.vmap(nll)(batch_x, batch_y), axis=0)

        @jax.jit
        def update_step(x_batch, y_batch, opt_state, params):
            loss, grads = jax.value_and_grad(
                cross_entropy_loss)(params, x_batch, y_batch)
            updates, opt_state = self.optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return opt_state, params, loss

        @ jax.jit
        def accuracy(y_true, x, params):
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
