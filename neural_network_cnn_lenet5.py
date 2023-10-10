
from import_packages import *
from torch.utils.tensorboard import SummaryWriter


class TrainerModule:

    def __init__(self,
                 model_name: str,
                 model_class: flax.linen.Module,
                 model_hparams: dict,
                 optimizer_name: str,
                 optimizer_hparams: dict,
                 exmp_imgs,
                 seed=42):
        super().__init__()
        self.model_name = model_name
        self.model_class = model_class
        self.model_hparams = model_hparams
        self.optimizer_name = optimizer_name
        self.optimizer_hparams = optimizer_hparams
        self.seed = seed
        self.model = self.model_class(**self.model_hparams)
        self.log_dir = os.path.join('RL_updates', self.model_name)
        self.logger = SummaryWriter(
            log_dir=self.log_dir)
        self.create_functions()
        self.init_model(exmp_imgs)

    def create_functions(self):
        def calculate_loss(params, batch, train):
            imgs, labels = batch
            outs = self.model.apply({'params': params},
                                    imgs,
                                    train=train,
                                    mutable=['params'])
            logits, new_model_state = outs if train else (outs, None)
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, labels).mean()
            acc = (logits.argmax(axis=-1) == labels).mean()
            return loss, acc

        def train_step(state, batch):
            def loss_fn(params): return calculate_loss(
                params, batch, train=True)
            ret, grads = jax.value_and_grad(
                loss_fn, has_aux=True)(state.params)
            loss, acc = ret[0], ret[1]
            state = state.apply_gradients(
                grads=grads)
            return state, loss, acc

        def eval_step(state, batch):
            _, acc = calculate_loss(state.params, batch, train=not False)
            return acc
        self.train_step = jax.jit(train_step)
        self.eval_step = jax.jit(eval_step)

    def init_model(self, exmp_imgs):
        init_rng = jax.random.PRNGKey(self.seed)
        variables = self.model.init(init_rng, exmp_imgs, train=True)
        self.init_params = variables['params']
        self.state = None

    def init_optimizer(self, num_epochs, num_steps_per_epoch):
        if self.optimizer_name.lower() == 'adam':
            opt_class = optax.adam
        elif self.optimizer_name.lower() == 'adamw':
            opt_class = optax.adamw
        elif self.optimizer_name.lower() == 'sgd':
            opt_class = optax.sgd
        else:
            assert False, f'Unknown optimizer "{opt_class}"'
        lr_schedule = optax.piecewise_constant_schedule(
            init_value=self.optimizer_hparams.pop('lr'),
            boundaries_and_scales={int(num_steps_per_epoch*num_epochs*0.6): 0.1,
                                   int(num_steps_per_epoch*num_epochs*0.85): 0.1}
        )
        transf = [optax.clip(1.0)]
        if opt_class == optax.sgd and 'weight_decay' in self.optimizer_hparams:
            transf.append(optax.add_decayed_weights(
                self.optimizer_hparams.pop('weight_decay')))
        optimizer = optax.chain(
            *transf,
            opt_class(lr_schedule, **self.optimizer_hparams)
        )

        self.state = TrainState.create(apply_fn=self.model.apply,
                                       params=self.init_params if self.state is None else self.state.params,
                                       tx=optimizer)

    def train_model(self, train_loader, val_loader, num_epochs=200):
        self.init_optimizer(num_epochs, len(train_loader))
        best_eval = 0.0
        for epoch_idx in tqdm.tqdm(range(1, num_epochs+1)):
            self.train_epoch(train_loader, epoch=epoch_idx)
            if epoch_idx % 2 == 0:
                eval_acc = self.eval_model(val_loader)
                self.logger.add_scalar(
                    'val/acc', eval_acc, global_step=epoch_idx)
                if eval_acc >= best_eval:
                    best_eval = eval_acc
                    self.save_model(step=epoch_idx)
                self.logger.flush()

    def train_epoch(self, train_loader, epoch):
        metrics = collections.defaultdict(list)
        for batch in tqdm.tqdm(train_loader, desc='Training', leave=False):
            self.state, loss, acc = self.train_step(self.state, batch)
            metrics['loss'].append(loss)
            metrics['acc'].append(acc)
        for key in metrics.keys():
            avg_val = np.stack(jax.device_get(metrics[key])).mean()
            self.logger.add_scalar('train/'+key, avg_val, global_step=epoch)

    def eval_model(self, data_loader):
        # Test model on all images of a data loader and return avg loss
        correct_class, count = 0, 0
        for batch in data_loader:
            acc = self.eval_step(self.state, batch)
            correct_class += acc * batch[0].shape[0]
            count += batch[0].shape[0]
        eval_acc = (correct_class / count).item()
        return eval_acc

    def save_model(self, step=0):
        # Save current model at certain training iteration
        checkpoints.save_checkpoint(ckpt_dir=self.log_dir,
                                    target={'params': self.state.params},
                                    step=step,
                                    overwrite=True)

    def load_model(self, pretrained=False):
        # Load model. We use different checkpoint for pretrained models
        if not pretrained:
            state_dict = checkpoints.restore_checkpoint(
                ckpt_dir=self.log_dir, target=None)
        else:
            state_dict = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(
                'RL_updates', f'{self.model_name}.ckpt'), target=None)
        self.state = TrainState.create(apply_fn=self.model.apply,
                                       params=state_dict['params'],
                                       tx=self.state.tx if self.state else optax.sgd(
                                           0.1)   # Default optimizer
                                       )

    def checkpoint_exists(self):
        # Check whether a pretrained model exist for this autoencoder
        return os.path.isfile(os.path.join('RL_updates', f'{self.model_name}.ckpt'))


def train_classifier(*args, num_epochs=200, **kwargs):
    trainer = TrainerModule(*args, **kwargs)
    if not trainer.checkpoint_exists():
        trainer.train_model(train_loader, val_loader, num_epochs=num_epochs)
        trainer.load_model()
    else:
        trainer.load_model(pretrained=True)
    # Test trained model
    val_acc = trainer.eval_model(val_loader)
    test_acc = trainer.eval_model(test_loader)
    return trainer, {'val': val_acc, 'test': test_acc}


def image_to_numpy(img):
    img = np.array(img, dtype=np.float32)
    img = (img / 255. - 0) / 1
    return img


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class LeNet5(flax.linen.Module):
    num_classes: int

    @flax.linen.compact
    def __call__(self, x, train=True):
        x = flax.linen.Conv(features=6, kernel_size=(5, 5), strides=2)(x)
        x = flax.linen.relu(x)
        x = flax.linen.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = flax.linen.Conv(
            features=16, kernel_size=(5, 5), strides=2)(x)
        x = flax.linen.relu(x)
        x = flax.linen.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = jnp.reshape(x, (x.shape[0], -1))
        # x = jnp.squeeze(x)

        x = flax.linen.Dense(features=120)(x)
        x = flax.linen.relu(x)

        x = flax.linen.Dense(features=84)(x)
        x = flax.linen.relu(x)

        x = flax.linen.Dense(features=self.num_classes)(x)
        return x


if __name__ == '__main__':

    test_transform = image_to_numpy
    train_transform = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                                      torchvision.transforms.RandomResizedCrop(
                                                          (32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                                                      image_to_numpy
                                                      ])
    train_dataset = torchvision.datasets.CIFAR10(
        root='Dataset', train=True, transform=train_transform, download=True)
    val_dataset = torchvision.datasets.CIFAR10(
        root='Dataset', train=True, transform=test_transform, download=True)
    train_set, _ = torch.utils.data.random_split(
        train_dataset, [45000, 5000], generator=torch.Generator().manual_seed(42))
    _, val_set = torch.utils.data.random_split(
        val_dataset, [45000, 5000], generator=torch.Generator().manual_seed(42))

    test_set = torchvision.datasets.CIFAR10(
        root='Dataset', train=False, transform=test_transform, download=True)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=128,
                                               shuffle=True,
                                               drop_last=True,
                                               collate_fn=numpy_collate,
                                               num_workers=NUM_WORKERS,
                                               persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=128,
                                             shuffle=False,
                                             drop_last=False,
                                             collate_fn=numpy_collate,
                                             num_workers=NUM_WORKERS//2,
                                             persistent_workers=True)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=128,
                                              shuffle=False,
                                              drop_last=False,
                                              collate_fn=numpy_collate,
                                              num_workers=NUM_WORKERS//2,
                                              persistent_workers=True)

    lenet5_trainer, lenet5_results = train_classifier(model_class=LeNet5,
                                                      model_name="LeNet5",
                                                      model_hparams={
                                                          "num_classes": 10},
                                                      optimizer_name="adamw",
                                                      optimizer_hparams={"lr": 1e-3,
                                                                         "weight_decay": 1e-4},
                                                      exmp_imgs=jax.device_put(
                                                          next(iter(train_loader))[0]),
                                                      num_epochs=200)
