import jax
import jax.numpy as jnp

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


@jax.jit
def gaussian_pdf(x, mean, std):
    exponent = -0.5 * ((x - mean) / std) ** 2
    return (1.0 / (std * jnp.sqrt(2 * jnp.pi))) * jnp.exp(exponent)


def train_naive_bayes(X, y):
    num_samples, num_features = X.shape
    num_classes = len(jnp.unique(y))
    class_priors = jnp.bincount(y.astype(jnp.int32)) / num_samples

    means = jnp.stack([jnp.mean(X[y == c], axis=0)
                      for c in range(num_classes)])
    stds = jnp.stack([jnp.std(X[y == c], axis=0) for c in range(num_classes)])
    return class_priors, means, stds


@jax.jit
def predict_naive_bayes(class_priors, means, stds, x):
    likelihoods = gaussian_pdf(x, means, stds)
    posteriors = class_priors * jnp.prod(likelihoods, axis=1)
    return jnp.argmax(posteriors)


@jax.jit
def accuracy(y_true, y_pred):
    y_pred_binary = y_pred.astype(jnp.int32)
    correct_predictions = jnp.sum(y_true == y_pred_binary)
    total_predictions = y_true.shape[0]
    accuracy = correct_predictions / total_predictions
    return accuracy


if __name__ == '__main__':
    dataset = Dataset()
    train, test = dataset.get_data()

    class_priors, means, stds = train_naive_bayes(train[0], train[1])
    train_predictions = jax.vmap(lambda x: predict_naive_bayes(
        class_priors, means, stds, x))(train[0])
    test_predictions = jax.vmap(lambda x: predict_naive_bayes(
        class_priors, means, stds, x))(test[0])
    print('Train Accuracy:', round(
        accuracy(train[1], train_predictions)*100, 3), '%')
    print('Test Accuracy:', round(
        accuracy(test[1], test_predictions))*100, '%')
