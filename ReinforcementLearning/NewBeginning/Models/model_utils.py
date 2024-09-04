import jax
import flax

rng = jax.random.PRNGKey(0)

def show_summary(Q, x):
    print(flax.linen.tabulate(Q, rng, compute_flops=True, compute_vjp_flops=True)(*x))