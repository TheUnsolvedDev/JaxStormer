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
        indices = np.arange(len(self.train[0]) // self.batch_size)
        np.random.shuffle(indices)
        (self.X_train, self.y_train), _ = self.get_data()
        for ind in indices:
            data = self.X_train[(ind) * self.batch_size:(ind + 1) * self.batch_size]  # type: jnp.ndarray
            labels = self.y_train[(ind) * self.batch_size:(ind + 1) * self.batch_size]  # type: jnp.ndarray
            yield data, labels

    def val_data_loader(self) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Generate the validation data batch by batch.

        Yields:
            Iterator[Tuple[jnp.ndarray, jnp.ndarray]]: The validation data and labels.
        """
        indices = np.arange(len(self.test[0]) // self.batch_size)
        np.random.shuffle(indices)
        _, (self.X_test, self.y_test) = self.get_data()
        for ind in indices:
            data = self.X_test[(ind) * self.batch_size:(ind + 1) * self.batch_size]  # type: jnp.ndarray
            labels = self.y_test[(ind) * self.batch_size:(ind + 1) * self.batch_size]  # type: jnp.ndarray
            yield data, labels


resnet_kernel_init = flax.linen.initializers.variance_scaling(
    2.0, mode='fan_out', distribution='normal')


class ResNetBlock(flax.linen.Module):
    act_fn: callable  
    c_out: int  
    subsample: bool = False  

    @flax.linen.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """
        Apply a ResNet block to the input tensor.

        Args:
            x (jnp.ndarray): The input tensor.
            train (bool): Whether the model is in training mode.

        Returns:
            jnp.ndarray: The output tensor after applying the ResNet block.
        """
        z = flax.linen.Conv(self.c_out, kernel_size=(3, 3),
                            strides=(1, 1) if not self.subsample else (2, 2),
                            kernel_init=resnet_kernel_init,
                            use_bias=False)(x)
        z = flax.linen.BatchNorm()(z, use_running_average=not train)
        z = self.act_fn(z)
        z = flax.linen.Conv(self.c_out, kernel_size=(3, 3),
                            kernel_init=resnet_kernel_init,
                            use_bias=False)(z)
        z = flax.linen.BatchNorm()(z, use_running_average=not train)

        if self.subsample:
            x = flax.linen.Conv(self.c_out, kernel_size=(1, 1), strides=(
                2, 2), kernel_init=resnet_kernel_init)(x)

        x_out = self.act_fn(z + x)
        return x_out


class PreActResNetBlock(ResNetBlock):

    @flax.linen.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """
        Apply the Pre-activation ResNet block to the input tensor.

        Args:
            x (jnp.ndarray): The input tensor.
            train (bool, optional): If True, use batch normalization with running average. Defaults to True.

        Returns:
            jnp.ndarray: The output tensor after applying the Pre-activation ResNet block.
        """
        z = flax.linen.BatchNorm()(x, use_running_average=not train)
        z = self.act_fn(z)
        z = flax.linen.Conv(self.c_out, kernel_size=(3, 3),
                            strides=(1, 1) if not self.subsample else (2, 2),
                            kernel_init=resnet_kernel_init,
                            use_bias=False)(z)
        z = flax.linen.BatchNorm()(z, use_running_average=not train)
        z = self.act_fn(z)
        z = flax.linen.Conv(self.c_out, kernel_size=(3, 3),
                            kernel_init=resnet_kernel_init,
                            use_bias=False)(z)

        if self.subsample:
            x = flax.linen.BatchNorm()(x, use_running_average=not train)
            x = self.act_fn(x)
            x = flax.linen.Conv(self.c_out,
                                kernel_size=(1, 1),
                                strides=(2, 2),
                                kernel_init=resnet_kernel_init,
                                use_bias=False)(x)

        x_out = z + x
        return x_out


class ResNet(flax.linen.Module):
    num_classes: int = 10
    act_fn: callable = flax.linen.relu
    block_class: flax.linen.Module = PreActResNetBlock
    num_blocks: tuple = (3, 3, 3)
    c_hidden: tuple = (16, 32, 64)

    @flax.linen.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """
        Apply the ResNet model to the input tensor.

        Args:
            x (jnp.ndarray): The input tensor of shape (batch_size, height, width, channels).
            train (bool, optional): If True, use batch normalization with running average. Defaults to True.

        Returns:
            jnp.ndarray: The output tensor of shape (batch_size, num_classes).
        """
        x = flax.linen.Conv(self.c_hidden[0], kernel_size=(
            3, 3), kernel_init=resnet_kernel_init, use_bias=False)(x)
        if self.block_class == ResNetBlock: 
            x = flax.linen.BatchNorm()(x, use_running_average=not train)
            x = self.act_fn(x)

        for block_idx, block_count in enumerate(self.num_blocks):
            for bc in range(block_count):
                subsample = (bc == 0 and block_idx > 0)
                x = self.block_class(c_out=self.c_hidden[block_idx],
                                     act_fn=self.act_fn,
                                     subsample=subsample)(x, train=train)

        x = x.mean(axis=(1, 2))
        x = jnp.reshape(x, (x.shape[0], -1))
        x = flax.linen.Dense(self.num_classes)(x)
        return x


class TrainStateNew(TrainState):
    batch_stats: flax.core.FrozenDict


class ConvolutionalNeuralNetwork:
    def __init__(
        self,
        alpha: float = 0.01,  # type: ignore
        batch_size: int = 128,  # type: ignore
        n_epochs: int = 100,  # type: ignore
        model: ResNet = ResNet(),  # type: ignore
        model_name: str = 'ResNet'  # type: ignore
    ) -> None:
        """
        Initialize a ConvolutionalNeuralNetwork object.

        Args:
            alpha (float, optional): The learning rate. Defaults to 0.01.
            batch_size (int, optional): The batch size. Defaults to 128.
            n_epochs (int, optional): The number of epochs. Defaults to 100.
            model (ResNet, optional): The neural network model. Defaults to ResNet().
            model_name (str, optional): The name of the model. Defaults to 'ResNet'.

        Returns:
            None
        """
        self.alpha = alpha
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.model_name = model_name
        self.log_dir = os.path.join('/tmp/DL_updates', self.model_name)
        self.model = model
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
        print(self.model.tabulate(init_rng, dummy_image))
        variables = self.model.init(init_rng, dummy_image, train=True)
        self.init_params = variables['params']  # type: flax.core.frozen_dict.FrozenDict
        self.init_batch_stats = variables['batch_stats']  # type: flax.core.frozen_dict.FrozenDict
        self.state = None

    def init_optimizer(self) -> None:
        """
        Initialize the optimizer.

        Returns:
            None
        """
        opt_class = optax.adam
        num_steps_per_epoch = 60_000 // self.batch_size
        lr_schedule = optax.piecewise_constant_schedule(
            init_value=self.alpha,  # type: float
            boundaries_and_scales={int(num_steps_per_epoch * self.n_epochs * 0.6): 0.1,
                                   int(num_steps_per_epoch * self.n_epochs * 0.85): 0.1}
        )
        transf = [optax.clip(1.0)]
        optimizer = optax.chain(
            *transf,
            opt_class(lr_schedule)
        )

        self.state = TrainStateNew.create(
            apply_fn=self.model.apply,  # type: Callable[[flax.core.frozen_dict.FrozenDict, jnp.ndarray], jnp.ndarray]
            params=self.init_params if self.state is None else self.state.params,  # type: flax.core.frozen_dict.FrozenDict
            batch_stats=self.init_batch_stats if self.state is None else self.state.batch_stats,
            tx=optimizer
        )

    def train_model(self, train: Callable[[], Iterable[Tuple[jnp.ndarray, jnp.ndarray]]],  # type: ignore
                    val: Callable[[], Iterable[Tuple[jnp.ndarray, jnp.ndarray]]]) -> None:
        """
        Train the model.

        Args:
            train (Callable[[], Iterable[Tuple[jnp.ndarray, jnp.ndarray]]]): A function that yields batches of training data.
                Each batch is a tuple of (image, label), where image is a jnp.ndarray of shape (batch_size, height, width, channels)
                and label is a jnp.ndarray of shape (batch_size,).
            val (Callable[[], Iterable[Tuple[jnp.ndarray, jnp.ndarray]]]): A function that yields batches of validation data.
                Each batch is a tuple of (image, label), where image is a jnp.ndarray of shape (batch_size, height, width, channels)
                and label is a jnp.ndarray of shape (batch_size,).

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

    def train_epoch(self, train_loader: Iterable[Tuple[jnp.ndarray, jnp.ndarray]], epoch: int) -> None:
        """
        Train the model for one epoch.

        Args:
            train_loader (Iterable[Tuple[jnp.ndarray, jnp.ndarray]]): An iterable that yields batches of training data.
                Each batch is a tuple of (image, label), where image is a jnp.ndarray of shape (batch_size, height, width, channels)
                and label is a jnp.ndarray of shape (batch_size,).
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
            state (TrainState): The training state.
            batch (Tuple[jnp.ndarray, jnp.ndarray]): A tuple containing the batch of images and labels.
                Each batch is a tuple of (image, label), where image is a jnp.ndarray of shape (batch_size, height, width, channels)
                and label is a jnp.ndarray of shape (batch_size,).

        Returns:
            Tuple[TrainState, float, float]: A tuple containing the updated training state, the loss, and the accuracy.
        """
        def loss_fn(params: jnp.ndarray) -> Tuple[float, float, Any]:
            """
            Compute the loss and accuracy for a given set of parameters.

            Args:
                params (jnp.ndarray): The parameters.

            Returns:
                Tuple[float, float, Any]: A tuple containing the loss, accuracy, and the new model state.
            """
            return self.calculate_loss(params, state.batch_stats, batch, train=True)

        ret, grads = jax.value_and_grad(
            loss_fn, has_aux=True)(state.params)
        loss, acc, new_model_state = ret[0], *ret[1]
        state = state.apply_gradients(grads=grads)
        state = state.replace(batch_stats=new_model_state['batch_stats'])
        return state, loss, acc

    def calculate_loss(
        self,
        params: jnp.ndarray,
        batch_stats: Any,
        batch: Tuple[jnp.ndarray, jnp.ndarray],
        train: bool
    ) -> Tuple[float, Tuple[float, Any]]:
        """
        Compute the loss and accuracy for a given set of parameters.

        Args:
            params (jnp.ndarray): The parameters.
            batch_stats (Any): The batch statistics.
            batch (Tuple[jnp.ndarray, jnp.ndarray]): A tuple containing the batch of images and labels.
                Each batch is a tuple of (image, label), where image is a jnp.ndarray of shape (batch_size, height, width, channels)
                and label is a jnp.ndarray of shape (batch_size,).
            train (bool): Whether the model is in training mode.

        Returns:
            Tuple[float, Tuple[float, Any]]: A tuple containing the loss and accuracy.
        """
        imgs, labels = batch
        outs = self.model.apply(
            {'params': params, 'batch_stats': batch_stats},
            imgs,
            train=train,
            mutable=['batch_stats'] if train else False
        )
        logits, new_model_state = outs if train else (outs, None)
        loss = optax.softmax_cross_entropy(logits, labels).mean()
        acc = (logits.argmax(axis=-1) == labels.argmax(axis=-1)).mean()
        return loss, (acc, new_model_state)

    @partial(jax.jit, static_argnums=(0,))
    def eval_step(
        self,
        state: TrainState,
        batch: Tuple[jnp.ndarray, jnp.ndarray]
    ) -> float:
        """
        Compute the evaluation accuracy for a given set of parameters and batch.

        Args:
            state (TrainState): The training state.
            batch (Tuple[jnp.ndarray, jnp.ndarray]): A tuple containing the batch of images and labels.
                Each batch is a tuple of (image, label), where image is a jnp.ndarray of shape (batch_size, height, width, channels)
                and label is a jnp.ndarray of shape (batch_size,).

        Returns:
            float: The evaluation accuracy.
        """
        _, (acc, _) = self.calculate_loss(
            state.params,
            state.batch_stats,
            batch,
            train=False
        )
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
            acc = self.eval_step(self.state, batch)
            correct_class += acc * batch[0].shape[0]
            count += batch[0].shape[0]
        eval_acc: float = (correct_class / count).item()
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
            pretrained (bool, optional): Whether to load the pre-trained model.
                Defaults to False.

        Returns:
            None: This function does not return anything.
        """
        if not pretrained:
            state_dict = checkpoints.restore_checkpoint(
                ckpt_dir=self.log_dir, target=None)
        else:
            state_dict = checkpoints.restore_checkpoint(
                ckpt_dir=os.path.join('DL_updates', f'{self.model_name}.ckpt'),
                target=None)
        self.state = TrainStateNew.create(
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
