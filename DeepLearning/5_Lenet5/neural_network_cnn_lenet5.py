import import_packages
from import_packages import *


def scale(data: np.ndarray) -> jnp.ndarray:
    """
    Scale the data by dividing it by 255.0.

    Args:
        data (np.ndarray): The input data.

    Returns:
        jnp.ndarray: The scaled data.
    """
    data = jnp.array(data)
    return data / 255.0


class Dataset:
    def __init__(self, batch_size: int = 128) -> None:
        """
        Initialize the Dataset class.

        Args:
            batch_size (int): The size of the mini-batches for training.

        Returns:
            None
        """
        # import_packages.init()
        self.batch_size = batch_size
        self.train, self.test = import_packages.load()

    def get_data(self) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Get the training and testing data.

        Returns:
            Tuple[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]: The scaled training and testing data.
        """
        self.X_train = scale(self.train[0].astype(np.float32)).reshape(-1, 28, 28, 1)  # type: jnp.ndarray
        self.y_train = jax.nn.one_hot(self.train[1], 10)  # type: jnp.ndarray
        self.X_test = scale(self.test[0].astype(np.float32)).reshape(-1, 28, 28, 1)  # type: jnp.ndarray
        self.y_test = jax.nn.one_hot(self.test[1], 10)  # type: jnp.ndarray
        return (self.X_train, self.y_train), (self.X_test, self.y_test)

    def train_data_loader(self) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Generate the training data batch by batch.

        Yields:
            Iterator[Tuple[jnp.ndarray, jnp.ndarray]]: The training data and labels.
        """
        indices = np.arange(len(self.train[0])//self.batch_size)
        np.random.shuffle(indices)
        (self.X_train, self.y_train), _ = self.get_data()
        for ind in indices:
            data = self.X_train[(ind)*self.batch_size:(ind+1)*self.batch_size]
            labels = self.y_train[(ind)*self.batch_size:(ind+1)*self.batch_size]
            yield data, labels

    def val_data_loader(self) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Generate the validation data batch by batch.

        Yields:
            Iterator[Tuple[jnp.ndarray, jnp.ndarray]]: The validation data and labels.
        """
        indices = np.arange(len(self.test[0])//self.batch_size)
        np.random.shuffle(indices)
        _, (self.X_test, self.y_test) = self.get_data()
        for ind in indices:
            data = self.X_test[(ind)*self.batch_size:(ind+1)*self.batch_size]  # type: jnp.ndarray
            labels = self.y_test[(ind)*self.batch_size:(ind+1)*self.batch_size]  # type: jnp.ndarray
            yield data, labels


class LeNet5(flax.linen.Module):

    @flax.linen.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Apply the LeNet5 convolutional neural network to the input data.

        Args:
            x (jnp.ndarray): The input data, of shape (batch_size, height, width, channels).

        Returns:
            jnp.ndarray: The output of the neural network, of shape (batch_size, num_classes).
        """
        x = flax.linen.Conv(features=6, kernel_size=(5, 5), strides=2)(x)
        x = flax.linen.relu(x)
        x = flax.linen.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = flax.linen.Conv(features=16, kernel_size=(5, 5), strides=2)(x)
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


class ConvolutionalNeuralNetwork:
    def __init__(
        self,
        alpha: float = 0.01,  # type: ignore
        batch_size: int = 128,  # type: ignore
        n_epochs: int = 100,  # type: ignore
        model_name: str = 'LeNet5',  # type: ignore
    ) -> None:
        """
        Initialize a ConvolutionalNeuralNetwork object.

        Args:
            alpha (float, optional): The learning rate. Defaults to 0.01.
            batch_size (int, optional): The batch size. Defaults to 128.
            n_epochs (int, optional): The number of epochs. Defaults to 100.
            model_name (str, optional): The name of the model. Defaults to 'LeNet5'.

        Returns:
            None
        """
        self.alpha = alpha
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.model_name = model_name
        self.log_dir = os.path.join('/tmp/DL_updates', model_name)
        self.init_model()

    def init_model(self, dummy_image: np.ndarray = np.ones((10, 28, 28, 1))) -> None:
        """
        Initialize the model.

        Args:
            dummy_image (np.ndarray, optional): The dummy input image.
                Defaults to np.ones((10, 28, 28, 1)).

        Returns:
            None
        """
        init_rng = jax.random.PRNGKey(0)
        self.model = LeNet5()
        print(self.model.tabulate(init_rng, dummy_image))
        variables = self.model.init(init_rng, dummy_image)
        self.init_params: flax.core.frozen_dict.FrozenDict = variables
        self.state: Optional[TrainState] = None

    def init_optimizer(self) -> None:
        """
        Initialize the optimizer.

        Returns:
            None
        """
        opt_class = optax.adam
        num_steps_per_epoch = 60_000//self.batch_size
        lr_schedule = optax.piecewise_constant_schedule(
            init_value=self.alpha,  # type: float
            boundaries_and_scales={int(num_steps_per_epoch*self.n_epochs*0.6): 0.1,
                                   int(num_steps_per_epoch*self.n_epochs*0.85): 0.1}
        )
        transf = [optax.clip(1.0)]
        optimizer = optax.chain(
            *transf,
            opt_class(lr_schedule)
        )
        self.model.apply = jax.jit(self.model.apply)  # type: Callable[[flax.core.frozen_dict.FrozenDict, jnp.ndarray], jnp.ndarray]
        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=self.init_params if self.state is None else self.state.params,  # type: flax.core.frozen_dict.FrozenDict
            tx=optimizer
        )

    def train_model(self, train: Callable[[], Iterable[Any]],  # type: ignore
                    val: Callable[[], Iterable[Any]]) -> None:
        """
        Train the model.

        Args:
            train (Callable[[], Iterable[Any]]): A function that yields batches of training data.
            val (Callable[[], Iterable[Any]]): A function that yields batches of validation data.

        Returns:
            None
        """
        self.init_optimizer()
        best_eval = 0.0
        for epoch_idx in tqdm.tqdm(range(1, self.n_epochs+1)):
            self.train_epoch(train(), epoch=epoch_idx)
            if epoch_idx % 5 == 0:
                eval_acc = self.eval_model(val())
                print('Val Acc:', round(eval_acc, 3))
                if eval_acc >= best_eval:
                    best_eval = eval_acc
                    self.save_model(step=epoch_idx)

    def train_epoch(self, train_loader: Iterable[Any], epoch: int) -> None:
        """
        Train the model for one epoch.

        Args:
            train_loader (Iterable[Any]): An iterable that yields batches of training data.
            epoch (int): The current epoch index.

        Returns:
            None
        """
        metrics = {'loss': [], 'acc': []}
        for batch in tqdm.tqdm(train_loader, desc='Training', leave=True):
            self.state, loss, acc = self.train_step(self.state, batch)
            metrics['loss'].append(loss)
            metrics['acc'].append(acc)

        avg_acc_loss = np.array(jax.device_get(metrics['loss'])).mean()
        avg_acc_val = np.array(jax.device_get(metrics['acc'])).mean()
        print('\t\t', [epoch], '/', [self.n_epochs], '\t Loss',
              round(avg_acc_loss, 4), '\t Acc', round(avg_acc_val, 4), end='\t')

    @partial(jax.jit, static_argnums=(0,))
    def train_step(self, state: TrainState, batch: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[TrainState, float, float]:
        """
        Train the model for one step.

        Args:
            state (flax.training.train_state.TrainState): The training state.
            batch (Tuple[jnp.ndarray, jnp.ndarray]): A tuple containing the batch of images and labels.

        Returns:
            Tuple[flax.training.train_state.TrainState, float, float]: A tuple containing the updated training state, the loss, and the accuracy.
        """
        def loss_fn(params: jnp.ndarray) -> Tuple[float, float]:
            """
            Compute the loss and accuracy for a given set of parameters.

            Args:
                params (jnp.ndarray): The parameters.

            Returns:
                Tuple[float, float]: A tuple containing the loss and accuracy.
            """
            imgs, labels = batch
            logits = self.model.apply(params, imgs)
            loss = optax.softmax_cross_entropy(
                logits, labels).mean()
            acc = (logits.argmax(axis=-1) == labels.argmax(axis=-1)).mean()
            return jnp.squeeze(loss), jnp.squeeze(acc)

        ret, grads = jax.value_and_grad(
            loss_fn, has_aux=True)(state.params)
        loss, acc = ret[0], ret[1]
        state = state.apply_gradients(
            grads=grads)
        return state, loss, acc

    @partial(jax.jit, static_argnums=(0,))
    def eval_step(self, state: TrainState, batch: Tuple[jnp.ndarray, jnp.ndarray]) -> float:
        """
        Compute the evaluation accuracy for a given set of parameters and batch.

        Args:
            state (flax.training.train_state.TrainState): The training state.
            batch (Tuple[jnp.ndarray, jnp.ndarray]): A tuple containing the batch of images and labels.

        Returns:
            float: The evaluation accuracy.
        """
        def acc_fn(params: jnp.ndarray, imgs: jnp.ndarray, labels: jnp.ndarray) -> float:
            """
            Compute the accuracy for a given set of parameters, images, and labels.

            Args:
                params (jnp.ndarray): The parameters.
                imgs (jnp.ndarray): The images.
                labels (jnp.ndarray): The labels.

            Returns:
                float: The accuracy.
            """
            logits = self.model.apply(params, imgs)
            acc = (logits.argmax(axis=-1) == labels.argmax(axis=-1)).mean()
            return jnp.squeeze(acc)
        
        imgs, labels = batch
        acc = acc_fn(state.params, imgs, labels)
        return acc

    def eval_model(self, data_loader: DataLoader) -> float:
        """
        Compute the evaluation accuracy for a given data loader.

        Args:
            data_loader (DataLoader): The data loader.

        Returns:
            float: The evaluation accuracy.
        """
        correct_class: float = 0
        count: int = 0
        for batch in data_loader:
            batch_acc: float = self.eval_step(self.state, batch)
            correct_class += batch_acc * batch[0].shape[0]
            count += batch[0].shape[0]
        eval_acc: float = correct_class / count
        return eval_acc

    def save_model(self, step: int = 0) -> None:
        """
        Save the model parameters to a checkpoint file.

        Args:
            step (int): The step number associated with the checkpoint.

        Returns:
            None
        """
        checkpoints.save_checkpoint(
            ckpt_dir=self.log_dir,
            target={'params': self.state.params},
            step=step,
            overwrite=True)

    def load_model(self, pretrained: bool = False) -> None:
        """
        Load a pre-trained model from a checkpoint.

        Args:
            pretrained (bool, optional): Whether to load the pre-trained model. Defaults to False.

        Returns:
            None
        """
        if not pretrained:
            state_dict = checkpoints.restore_checkpoint(
                ckpt_dir=self.log_dir, target=None)
        else:
            state_dict = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(
                'DL_updates', f'{self.model_name}.ckpt'), target=None)
        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=state_dict['params'],
            tx=self.state.tx if self.state else optax.sgd(0.1)
        )

    def checkpoint_exists(self) -> bool:
        """
        Check if a checkpoint file for the model exists.

        Returns:
            bool: True if the checkpoint file exists, False otherwise.
        """
        return os.path.isfile(os.path.join('DL_updates', f'{self.model_name}.ckpt'))


if __name__ == '__main__':
    data = Dataset(64)
    model = ConvolutionalNeuralNetwork(batch_size=64)
    model.train_model(data.train_data_loader, data.val_data_loader)
