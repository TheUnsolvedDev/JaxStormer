# Activate the JAX_GPU conda environment
echo "**** Going to env JAX_GPU ****"
source activate JAX_GPU

if ! python3 -c "import jax" &> /dev/null; then
    echo "jax is not installed. Installing..."
    pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    echo "Jax has been installed successfully."
else
    echo "Jax is installed"
fi
python3 -c "import jax; print('GPUs Available: ', jax.devices())"

if ! python3 -c "import matplotlib.pyplot as plt" &>/dev/null; then
    echo "matplotlib is not installed. Installing..."
    pip3 install matplotlib
    echo "matplotlib has been installed successfully."
else
    echo "matplotlib is installed"
fi

if ! python3 -c "import tqdm" &>/dev/null; then
    echo "tqdm is not installed. Installing..."
    pip3 install tqdm
    echo "tqdm has been installed successfully."
else
    echo "tqdm is installed"
fi

if ! python3 -c "import flax" &> /dev/null; then
    echo "flax is not installed. Installing..."
    pip3 install flax
    echo "flax has been installed successfully."
else
    echo "Flax is installed"
fi

if ! python3 -c "import optax" &> /dev/null; then
    echo "optax is not installed. Installing..."
    pip3 install optax
    echo "optax has been installed successfully."
else
    echo "optax is installed"
fi

if ! python3 -c "import orbax" &> /dev/null; then
    echo "orbax is not installed. Installing..."
    pip3 install orbax
    echo "orbax has been installed successfully."
else
    echo "orbax is installed"
fi

if ! python3 -c "import gymnax" &> /dev/null; then
    echo "gymnax is not installed. Installing..."
    pip3 install gymnax
    echo "gymnax has been installed successfully."
else
    echo "gymnax is installed"
fi

if ! python3 -c "import rlax" &> /dev/null; then
    echo "rlax is not installed. Installing..."
    pip3 install rlax
    echo "rlax has been installed successfully."
else
    echo "rlax is installed"
fi

if ! python3 -c "import gymnasium" &> /dev/null; then
    echo "gymnasium is not installed. Installing..."
    pip3 install gymnasium
    echo "gymnasium has been installed successfully."
else
    echo "gymnasium is installed"
fi

pip3 install gymnax
pip3 install moviepy
pip3 install gymnasium[classic-control]
echo "Env Checked good to go!!"
conda deactivate
sleep 5
clear