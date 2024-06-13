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
from typing import *

filename = [
    ["training_images", "train-images-idx3-ubyte.gz"],
    ["test_images", "t10k-images-idx3-ubyte.gz"],
    ["training_labels", "train-labels-idx1-ubyte.gz"],
    ["test_labels", "t10k-labels-idx1-ubyte.gz"]
]


def download_mnist() -> None:
    """
    Downloads the MNIST dataset from Yann LeCun's website.

    This function iterates over the `filename` list and downloads each file
    specified by the tuple in the list. The function prints the name of the file
    being downloaded and the name of the file being saved. Once all files have
    been downloaded, it prints "Download complete."

    Args:
        None

    Returns:
        None
    """
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print(f"Downloading {name[1]}...")
        request.urlretrieve(f"{base_url}{name[1]}", name[1])
    print("Download complete.")


def save_mnist() -> None:
    """
    Saves the MNIST dataset as a pickle file.

    This function reads the MNIST dataset from the gzip files and saves it as a
    pickle file. The function iterates over the first two files in the `filename`
    list and reads each file using `gzip.open`. The function then reshapes the
    data into a 2D array of shape (-1, 28*28) and stores it in a dictionary with
    the corresponding key from the `filename` list. The function repeats this
    process for the last two files in the `filename` list. Finally, the function
    saves the dictionary as a pickle file named "mnist.pkl" in the "Dataset"
    directory. The function prints "Save complete." once the saving process is
    complete.

    Args:
        None

    Returns:
        None
    """
    mnist: dict[str, np.ndarray] = {}
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


def init() -> None:
    """
    Initializes the dataset by downloading and saving it.

    This function checks if the "Dataset" directory exists. If it does not exist,
    the function calls the `download_mnist` function to download the MNIST dataset
    from the internet. Once the download is complete, the function calls the
    `save_mnist` function to save the dataset as a pickle file. The function does
    not return anything.

    Args:
        None

    Returns:
        None
    """
    if not os.path.exists("Dataset"):
        download_mnist()
        save_mnist()


def load() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load the MNIST dataset from the pickle file.

    This function loads the MNIST dataset from the pickle file "mnist.pkl" located in the "Dataset" directory.
    The dataset is a tuple of two tuples: the first tuple contains the training images and labels, and the
    second tuple contains the test images and labels.

    Returns:
        Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]: The MNIST dataset.
    """
    with open("Dataset/mnist.pkl", 'rb') as f:
        mnist = pickle.load(f)
    return (mnist["training_images"], mnist["training_labels"]), (mnist["test_images"], mnist["test_labels"])


class DataAugmentation:
    def resize(self, image: jnp.ndarray, shape: Tuple[int, int]) -> jnp.ndarray:
        """
        Resize the image to the specified shape.

        Args:
            image (jnp.ndarray): The input image.
            shape (Tuple[int, int]): The shape to which the image should be resized.

        Returns:
            jnp.ndarray: The resized image.
        """
        return jax.image.resize(image, shape=shape, method='nearest')


    @partial(jax.jit, static_argnums=(0,))
    def rescale(self, image: jnp.ndarray) -> jnp.ndarray:
        """
        Rescale the image by dividing it by 255.0.

        Args:
            image (jnp.ndarray): The input image.

        Returns:
            jnp.ndarray: The rescaled image.
        """
        return jnp.divide(image, 255.0)


    @partial(jax.jit, static_argnums=(0,))
    def normalize(self, image: jnp.ndarray) -> jnp.ndarray:
        """
        Normalize the image by subtracting the mean and dividing by the standard deviation along the last axis.

        Args:
            image (jnp.ndarray): The input image.

        Returns:
            jnp.ndarray: The normalized image.
        """
        return jax.nn.normalize(image, axis=-1, mean=0.0, variance=1.0)

    @partial(jax.jit, static_argnums=(0,))
    def rotate_90(self, img: jnp.ndarray) -> jnp.ndarray:
        """
        Rotate the image by 90 degrees.

        Args:
            img (jnp.ndarray): The input image.

        Returns:
            jnp.ndarray: The rotated image.
        """
        return jnp.rot90(img, k=1, axes=(0, 1))

    @partial(jax.jit, static_argnums=(0,))
    def identity(self, img: jnp.ndarray) -> jnp.ndarray:
        """
        Return the input image unchanged.

        Args:
            img (jnp.ndarray): The input image.

        Returns:
            jnp.ndarray: The input image.
        """
        return img

    @partial(jax.jit, static_argnums=(0,))
    def flip_left_right(self, img: jnp.ndarray) -> jnp.ndarray:
        """
        Flip the image horizontally.

        Args:
            img (jnp.ndarray): The input image.

        Returns:
            jnp.ndarray: The flipped image.
        """
        return jnp.fliplr(img)

    @partial(jax.jit, static_argnums=(0,))
    def flip_up_down(self, img: jnp.ndarray) -> jnp.ndarray:
        """
        Flip the image vertically.

        Args:
            img (jnp.ndarray): The input image.

        Returns:
            jnp.ndarray: The flipped image.
        """
        return jnp.flipud(img)

    @partial(jax.jit, static_argnums=(0,))
    def random_rotate(self, img: jnp.ndarray, rotate: bool) -> jnp.ndarray:
        """
        Rotate the image randomly.

        Args:
            img (jnp.ndarray): The input image.
            rotate (bool): Whether to rotate the image.

        Returns:
            jnp.ndarray: The rotated image.
        """
        return jax.lax.cond(rotate, self.rotate_90, self.identity, img)

    @partial(jax.jit, static_argnums=(0,))
    def random_horizontal_flip(self, img: jnp.ndarray, flip: bool) -> jnp.ndarray:
        """
        Randomly flip the image horizontally.

        Args:
            img (jnp.ndarray): The input image.
            flip (bool): Whether to flip the image.

        Returns:
            jnp.ndarray: The flipped image.
        """
        return jax.lax.cond(flip, self.flip_left_right, self.identity, img)

    @partial(jax.jit, static_argnums=(0,))
    def random_vertical_flip(
            self,
            img: jnp.ndarray,
            flip: bool
        ) -> jnp.ndarray:
        """
        Randomly flip the image vertically.

        Args:
            img (jnp.ndarray): The input image.
            flip (bool): Whether to flip the image vertically.

        Returns:
            jnp.ndarray: The flipped image.
        """
        return jax.lax.cond(flip, self.flip_up_down, self.identity, img)



print('Num Devices:', jax.device_count())
print('Num local devices:', jax.local_device_count())
print('Devices:', jax.devices())
# os.environ['CUDA_VISIBLE_DEVICES'] = str([args.gpu])
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

if __name__ == '__main__':
    init()
