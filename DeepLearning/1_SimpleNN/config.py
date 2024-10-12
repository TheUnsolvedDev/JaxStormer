from urllib import request
import gzip
import pickle
import numpy as np
from typing import *
import os


filename = [
    ["training_images", "train-images-idx3-ubyte.gz"],
    ["test_images", "t10k-images-idx3-ubyte.gz"],
    ["training_labels", "train-labels-idx1-ubyte.gz"],
    ["test_labels", "t10k-labels-idx1-ubyte.gz"]
]

def download_mnist() -> None:
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print(f"Downloading {name[1]}...")
        request.urlretrieve(f"{base_url}{name[1]}", name[1])
    print("Download complete.")
    
def save_mnist() -> None:
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
    if not os.path.exists("Dataset"):
        download_mnist()
        save_mnist()

def load() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    with open("Dataset/mnist.pkl", 'rb') as f:
        mnist = pickle.load(f)
    return (mnist["training_images"], mnist["training_labels"]), (mnist["test_images"], mnist["test_labels"])
    
if __name__ == '__main__':
    init()