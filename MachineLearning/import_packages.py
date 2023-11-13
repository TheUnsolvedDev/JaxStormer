import os
import jax
import jax.numpy as jnp
import numpy as np
import tqdm


print('Num Devices:',jax.device_count())
print('Num local devices:',jax.local_device_count() )
print('Devices:',jax.devices())
# os.environ['CUDA_VISIBLE_DEVICES'] = str([args.gpu])
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
