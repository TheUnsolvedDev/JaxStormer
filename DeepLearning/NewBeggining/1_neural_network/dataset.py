import jax
import jax.numpy as jnp
import flax
import numpy as np
import nvidia.dali as dali
import os

from config import *


class Dataset(object):
    def __init__(self):
        self.categories = ["COVID19", "NORMAL", "PNEUMONIA", "TURBERCULOSIS"]


if __name__ == '__main__':
    print('Hello, world!')