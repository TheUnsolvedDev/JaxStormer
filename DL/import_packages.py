import jax
import jax.numpy as jnp
import numpy as np
import flax
import optax
import tqdm
import os
from flax.training.train_state import TrainState
from flax.training import checkpoints

from urllib import request
import gzip
import pickle

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
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist, f)
    print("Save complete.")


def init():
    download_mnist()
    save_mnist()


def load():
    with open("mnist.pkl", 'rb') as f:
        mnist = pickle.load(f)
    return (mnist["training_images"], mnist["training_labels"]), (mnist["test_images"], mnist["test_labels"])


print('Num Devices:', jax.device_count())
print('Num local devices:', jax.local_device_count())
print('Devices:', jax.devices())
# os.environ['CUDA_VISIBLE_DEVICES'] = str([args.gpu])
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

if __name__ == '__main__':
    init()
