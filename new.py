import jax

key = jax.random.PRNGKey(0)
for i in range(10):
    new_key, key = jax.random.split(key)
    print(key, new_key)
