python -m venv GPU
source GPU/bin/activate
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip3 install torch torchvision torchaudio && python3 -m pip install tensorflow[and-cuda]
pip3 install flax gymnax gymnasium orbax optax tqdm wandb