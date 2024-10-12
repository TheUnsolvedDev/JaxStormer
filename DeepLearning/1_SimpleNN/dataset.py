import jax
import jax.numpy as jnp
import numpy as np
import functools

from config import *


class DataAugmentation:
    def resize(self, image: jnp.ndarray, shape: Tuple[int, int]) -> jnp.ndarray:
        return jax.image.resize(image, shape=shape, method='nearest')

    @functools.partial(jax.jit, static_argnums=(0,))
    def rescale(self, image: jnp.ndarray) -> jnp.ndarray:
        return jnp.divide(image, 255.0)

    @functools.partial(jax.jit, static_argnums=(0,))
    def normalize(self, image: jnp.ndarray) -> jnp.ndarray:
        return jax.nn.normalize(image, axis=-1, mean=0.0, variance=1.0)

    @functools.partial(jax.jit, static_argnums=(0,))
    def rotate_90(self, img: jnp.ndarray) -> jnp.ndarray:
        return jnp.rot90(img, k=1, axes=(0, 1))

    @functools.partial(jax.jit, static_argnums=(0,))
    def identity(self, img: jnp.ndarray) -> jnp.ndarray:
        return img

    @functools.partial(jax.jit, static_argnums=(0,))
    def flip(self, img: jnp.ndarray) -> jnp.ndarray:
        return jnp.fliplr(img)

    @functools.partial(jax.jit, static_argnums=(0,))
    def transpose(self, img: jnp.ndarray) -> jnp.ndarray:
        return jnp.transpose(img)

    @functools.partial(jax.jit, static_argnums=(0,))
    def grayscale(self, img: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(img[..., :3], [0.299, 0.587, 0.114])

    @functools.partial(jax.jit, static_argnums=(0,))
    def blur(self, img: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16

    @functools.partial(jax.jit, static_argnums=(0,))
    def sharpen(self, img: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    @functools.partial(jax.jit, static_argnums=(0,))
    def edge(self, img: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])


class Dataset:
    def __init__(self,) -> None:
        init()
        self.datasets = load()
        self.aug = DataAugmentation()

    def get_data(self) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
        x_train, y_train = self.datasets[0]
        x_test, y_test = self.datasets[1]
        x_train = self.aug.rescale(x_train)
        y_train = jax.nn.one_hot(y_train, 10)
        x_test = self.aug.rescale(x_test)
        y_test = jax.nn.one_hot(y_test, 10)
        return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    dataset = Dataset()
    train_data, test_data = dataset.get_data()
    print(train_data)
    print(test_data)
