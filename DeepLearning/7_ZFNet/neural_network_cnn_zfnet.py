import import_packages
from import_packages import *

IMG_SHAPE = [48, 48]


class Dataset:
    def __init__(self, batch_size=128):
        # import_packages.init()
        self.batch_size = batch_size
        self.train, self.test = import_packages.load()
        self.data_aug = DataAugmentation()

    def get_data(self):
        self.X_train = self.train[0].astype(
            np.float32).reshape(-1, 28, 28, 1)
        self.y_train = jax.nn.one_hot(self.train[1], 10)
        self.X_test = self.test[0].astype(np.float32).reshape(-1, 28, 28, 1)
        self.y_test = jax.nn.one_hot(self.test[1], 10)
        return (self.X_train, self.y_train), (self.X_test, self.y_test)

    def train_data_loader(self):
        indices = np.arange(len(self.train[0])//self.batch_size)
        np.random.shuffle(indices)
        (self.X_train, self.y_train), _ = self.get_data()
        for ind in indices:
            data = self.X_train[(ind)*self.batch_size:(ind+1)*self.batch_size]
            labels = self.y_train[(ind)*self.batch_size:(ind+1)*self.batch_size]
            data = self.data_aug.resize(data, shape=[data.shape[0], *IMG_SHAPE, data.shape[-1]])
            data = self.data_aug.rescale(data)
            yield data, labels

    def val_data_loader(self):
        indices = np.arange(len(self.test[0])//self.batch_size)
        np.random.shuffle(indices)
        _, (self.X_test, self.y_test) = self.get_data()
        for ind in indices:
            data = self.X_test[(ind)*self.batch_size:(ind+1)*self.batch_size]
            labels = self.y_test[(ind)*self.batch_size:(ind+1)*self.batch_size]
            data = self.data_aug.resize(data, shape=[data.shape[0], *IMG_SHAPE, data.shape[-1]])
            data = self.data_aug.rescale(data)
            yield data, labels


class ZFNet(flax.linen.Module):
    num_classes: int = 10

    @flax.linen.compact
    def __call__(self, x, train=True):
        x = flax.linen.Conv(features=96, kernel_size=(7, 7), strides=2, padding=1)(x)
        x = flax.linen.BatchNorm()(x, use_running_average=not train)
        x = flax.linen.relu(x)
        x = flax.linen.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        x = flax.linen.Conv(features=256, kernel_size=(5, 5))(x)
        x = flax.linen.BatchNorm()(x, use_running_average=not train)
        x = flax.linen.relu(x)
        x = flax.linen.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        x = flax.linen.Conv(features=512, kernel_size=(3, 3))(x)
        x = flax.linen.BatchNorm()(x, use_running_average=not train)
        x = flax.linen.relu(x)
        x = flax.linen.Conv(features=1024, kernel_size=(3, 3))(x)
        x = flax.linen.BatchNorm()(x, use_running_average=not train)
        x = flax.linen.relu(x)
        x = flax.linen.Conv(features=512, kernel_size=(3, 3))(x)
        x = flax.linen.BatchNorm()(x, use_running_average=not train)
        x = flax.linen.relu(x)
        x = flax.linen.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        x = jnp.reshape(x, (x.shape[0], -1))
        x = flax.linen.Dense(features=1024)(x)
        x = flax.linen.relu(x)
        x = flax.linen.Dropout(0.5, deterministic=not train)(x)
        x = flax.linen.Dense(features=1024)(x)
        x = flax.linen.relu(x)
        x = flax.linen.Dropout(0.5, deterministic=not train)(x)
        x = flax.linen.Dense(features=self.num_classes)(x)
        return x


class TrainStateNew(TrainState):
    batch_stats: flax.core.FrozenDict


class ConvolutionalNeuralNetwork:
    def __init__(self, alpha=0.01, batch_size=128, n_epochs=100, model=ZFNet(), model_name='ZFNet'):
        self.alpha = alpha
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.model_name = model_name
        self.log_dir = os.path.join('/tmp/DL_updates', self.model_name)
        self.model = model
        self.init_model()

    def init_model(self, dummy_image=np.ones([10, *IMG_SHAPE, 1])):
        init_rng = {'params': jax.random.PRNGKey(0), 'dropout': jax.random.PRNGKey(1)}
        variables = self.model.init(init_rng, dummy_image, train=True)
        print(self.model.tabulate(init_rng, dummy_image, train=False))
        self.init_params, self.init_batch_stats = variables['params'], variables['batch_stats']
        self.state = None

    def init_optimizer(self):
        opt_class = optax.adam
        num_steps_per_epoch = 60_000//self.batch_size
        lr_schedule = optax.piecewise_constant_schedule(
            init_value=self.alpha,
            boundaries_and_scales={int(num_steps_per_epoch*self.n_epochs*0.6): 0.1,
                                   int(num_steps_per_epoch*self.n_epochs*0.85): 0.1}
        )
        transf = [optax.clip(1.0)]
        optimizer = optax.chain(
            *transf,
            opt_class(lr_schedule)
        )

        self.state = TrainStateNew.create(apply_fn=self.model.apply,
                                          params=self.init_params if self.state is None else self.state.params,
                                          batch_stats=self.init_batch_stats if self.state is None else self.state.batch_stats,
                                          tx=optimizer)

    def train_model(self, train, val):
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

    def train_epoch(self, train_loader, epoch):
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
    def train_step(self, state, batch):
        def loss_fn(params): return self.calculate_loss(params, state.batch_stats, batch, train=True)

        ret, grads = jax.value_and_grad(
            loss_fn, has_aux=True)(state.params)
        loss, acc, new_model_state = ret[0], *ret[1]
        state = state.apply_gradients(grads=grads)
        state = state.replace(batch_stats=new_model_state['batch_stats'])
        return state, loss, acc

    def calculate_loss(self, params, batch_stats, batch, train):
        imgs, labels = batch
        outs = self.model.apply({'params': params, 'batch_stats': batch_stats},
                                imgs,
                                rngs={'dropout': jax.random.PRNGKey(0)},
                                train=train,
                                mutable=['batch_stats'] if train else False)
        logits, new_model_state = outs if train else (outs, None)
        loss = optax.softmax_cross_entropy(logits, labels).mean()
        acc = (logits.argmax(axis=-1) == labels.argmax(axis=-1)).mean()
        return loss, (acc, new_model_state)

    @partial(jax.jit, static_argnums=(0,))
    def eval_step(self, state, batch):
        _, (acc, _) = self.calculate_loss(state.params, state.batch_stats, batch, train=False)
        return acc

    def eval_model(self, data_loader):
        correct_class, count = 0, 0
        for batch in data_loader:
            acc = self.eval_step(self.state, batch)
            correct_class += acc * batch[0].shape[0]
            count += batch[0].shape[0]
        eval_acc = (correct_class / count).item()
        return eval_acc

    def save_model(self, step=0):
        checkpoints.save_checkpoint(ckpt_dir=self.log_dir,
                                    target={'params': self.state.params},
                                    step=step,
                                    overwrite=True)

    def load_model(self, pretrained=False):
        if not pretrained:
            state_dict = checkpoints.restore_checkpoint(
                ckpt_dir=self.log_dir, target=None)
        else:
            state_dict = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(
                'DL_updates', f'{self.model_name}.ckpt'), target=None)
        self.state = TrainStateNew.create(apply_fn=self.model.apply,
                                          params=state_dict['params'],
                                          tx=self.state.tx if self.state else optax.sgd(
                                              0.1)
                                          )

    def checkpoint_exists(self):
        return os.path.isfile(os.path.join('DL_updates', f'{self.model_name}.ckpt'))


if __name__ == '__main__':
    data = Dataset(64)
    model = ConvolutionalNeuralNetwork(batch_size=64)
    model.train_model(data.train_data_loader, data.val_data_loader)
