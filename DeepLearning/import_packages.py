import jax
import jax.numpy as jnp
import numpy as np
import flax
import optax
import tqdm
import os
import cv2
from flax.training.train_state import TrainState
from flax.training import checkpoints
from functools import partial
from urllib import request
import gzip
import pickle
from typing import Any
import matplotlib.pyplot as plt

filename = [
    ["training_images", "train-images-idx3-ubyte.gz"],
    ["test_images", "t10k-images-idx3-ubyte.gz"],
    ["training_labels", "train-labels-idx1-ubyte.gz"],
    ["test_labels", "t10k-labels-idx1-ubyte.gz"]
]


def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading "+name[1]+"...")
        request.urlretrieve(base_url+name[1], name[1])
    print("Download complete.")


def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(
                f.read(), np.uint8, offset=16).reshape(-1, 28*28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("Dataset/mnist.pkl", 'wb') as f:
        pickle.dump(mnist, f)
    print("Save complete.")


def init():
    download_mnist()
    save_mnist()


def load():
    with open("Dataset/mnist.pkl", 'rb') as f:
        mnist = pickle.load(f)
    return (mnist["training_images"], mnist["training_labels"]), (mnist["test_images"], mnist["test_labels"])


class DataAugmentation:
    def resize(self, image, shape):
        return jax.image.resize(image, shape=shape, method='nearest')

    @partial(jax.jit, static_argnums=(0,))
    def rescale(self, image):
        return jnp.divide(image, 255.0)

    @partial(jax.jit, static_argnums=(0,))
    def normalize(self, image):
        return jax.nn.normalize(image, axis=-1, mean=0, variance=1)

    @partial(jax.jit, static_argnums=(0,))
    def rotate_90(self, img):
        return jnp.rot90(img, k=1, axes=(0, 1))

    @partial(jax.jit, static_argnums=(0,))
    def identity(self, img):
        return img

    @partial(jax.jit, static_argnums=(0,))
    def flip_left_right(self, img):
        return jnp.fliplr(img)

    @partial(jax.jit, static_argnums=(0,))
    def flip_up_down(self, img):
        return jnp.flipud(img)

    @partial(jax.jit, static_argnums=(0,))
    def random_rotate(self, img, rotate):
        return jax.lax.cond(rotate, self.rotate_90, self.identity, img)

    @partial(jax.jit, static_argnums=(0,))
    def random_horizontal_flip(self, img, flip):
        return jax.lax.cond(flip, self.flip_left_right, self.identity, img)

    @partial(jax.jit, static_argnums=(0,))
    def random_vertical_flip(self, img, flip):
        return jax.lax.cond(flip, self.flip_up_down, self.identity, img)


print('Num Devices:', jax.device_count())
print('Num local devices:', jax.local_device_count())
print('Devices:', jax.devices())
# os.environ['CUDA_VISIBLE_DEVICES'] = str([args.gpu])
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

if __name__ == '__main__':
    init()
