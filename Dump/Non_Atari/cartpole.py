import math
from functools import partial
import jax.numpy as jnp
import jax
from typing import *

class CartPole:
    """
    Copied from https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/cartpole.py
    ## Description

    This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
    ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
    The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
     in the left and right direction on the cart.

    ## Action Space

    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
     of the fixed force the cart is pushed with.

    - 0: Push cart to the left
    - 1: Push cart to the right

    **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
     the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

    ## Observation Space

    The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |

    **Note:** While the ranges above denote the possible values for observation space of each element,
        it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
    -  The cart x-position (index 0) can take values between `(-4.8, 4.8)`, but the episode terminates
       if the cart leaves the `(-2.4, 2.4)` range.
    -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
       if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

    ## Rewards

    Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken,
    including the termination step, is allotted. The threshold for rewards is 500 for v1 and 200 for v0.

    ## Starting State

    All observations are assigned a uniformly jax.random value in `(-0.05, 0.05)`

    ## Episode End

    The episode ends if any one of the following occurs:

    1. Termination: Pole Angle is greater than ±12°
    2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Truncation: Episode length is greater than 500 (200 for v0)
    """

    def __init__(self) -> None:
        """Initialize the cart-pole environment.

        Returns:
            None
        """
        self.gravity: float = 9.8  # m/s^2
        self.masscart: float = 1.0  # kg
        self.masspole: float = 0.1  # kg
        self.total_mass: float = self.masspole + self.masscart  # kg
        self.length: float = 0.5  # m, half the pole's length
        self.polemass_length: float = self.masspole * self.length  # kg m
        self.force_mag: float = 10.0  # N, force applied to the cart
        self.tau: float = 0.02  # s, seconds between state updates
        self.reset_bounds: float = 0.05  # random initialization range

        # Limits defining episode termination
        self.x_limit: float = 2.4  # m, distance from cart center to the edge
        self.theta_limit_rads: float = 12 * 2 * math.pi / 360  # radians, angle the pole can be tilted

    def __repr__(self) -> str:  # noqa: D102
        """Return a string representation of the CartPole environment.

        Returns:
            A string representing the environment's state.
        """
        return str(self.__dict__)

    @partial(jax.jit, static_argnums=0)  # noqa: F811
    def _get_obs(  # type: (jnp.ndarray) -> jnp.ndarray
        self, state: jnp.ndarray  # shape=(4,), dtype=float32
    ) -> jnp.ndarray:  # shape=(4,), dtype=float32
        """
        Returns the full state of the environment as the observation.

        Parameters:
            state (jax.numpy.ndarray): The environment state.

        Returns:
            The full environment state as the observation.
        """
        return state

    @partial(jax.jit, static_argnums=0)  # noqa: F811
    def _reset(  # type: (jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]
        self, key: jnp.ndarray  # shape=(2,), dtype=uint32
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:  # shape=(4,), dtype=float32
        """
        Resets the environment state and returns the initial observation.

        Parameters:
            key: The JAX PRNG key.

        Returns:
            A tuple (env_state, key), where env_state is the new environment state and
            key is the next JAX PRNG key.
        """
        new_state = jax.random.uniform(
            key,
            shape=(4,),
            minval=-self.reset_bounds,
            maxval=self.reset_bounds,
        )
        key, sub_key = jax.random.split(key)

        return new_state, sub_key

    @partial(jax.jit, static_argnums=0)
    def _reset_if_done(  # type: (CartPole, bool, Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]
        self, env_state, done):  # type: (Tuple[jnp.ndarray, jnp.ndarray])
        """
        If done, reset the environment state using self._reset, otherwise return the
        current environment state.

        Parameters:
            env_state (Tuple[jax.numpy.ndarray, jax.numpy.ndarray]): The current environment state.
            done (bool): Whether the current episode is done.

        Returns:
            The next environment state.
        """
        key = env_state[1]

        def _reset_fn(key):
            return self._reset(key)

        def _no_reset_fn(key):
            return env_state

        return jax.lax.cond(
            done,
            _reset_fn,
            _no_reset_fn,
            operand=key,
        )

    @partial(jax.jit, static_argnums=(0))  # noqa: F811
    def step(  # type: (CartPole, Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray, float, bool]
        self, env_state, action  # type: jnp.ndarray
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray, float, bool]:  # shape=(4,), dtype=float32, scalar, scalar
        """
        Steps the environment given an action.

        Parameters:
            env_state (Tuple[jax.numpy.ndarray, jax.numpy.ndarray]): The current environment state.
            action (jax.numpy.ndarray): The action to take.
                shape=(1,), dtype=float32

        Returns:
            A tuple (env_state, obs, reward, done), where env_state is the next
            environment state, obs is the observation based on the new state, reward
            is the reward for taking the action in the previous state, and done is
            whether the episode has ended.
        """
        state, key = env_state
        x, x_dot, theta, theta_dot = state

        force = jax.lax.cond(
            jnp.all(action) == 1,
            lambda _: self.force_mag,
            lambda _: -self.force_mag,
            operand=None,
        )
        cos_theta, sin_theta = jnp.cos(theta), jnp.sin(theta)

        temp = (
            force + self.polemass_length * jnp.square(theta_dot) * sin_theta
        ) / self.total_mass
        theta_accel = (self.gravity * sin_theta - cos_theta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * jnp.square(cos_theta) / self.total_mass)
        )
        x_accel = (
            temp - self.polemass_length * theta_accel * cos_theta / self.total_mass
        )

        # euler
        x += self.tau * x_dot
        x_dot += self.tau * x_accel
        theta += self.tau * theta_dot
        theta_dot += self.tau * theta_accel

        new_state = jnp.array([x, x_dot, theta, theta_dot])

        done = (
            (x < -self.x_limit)
            | (x > self.x_limit)
            | (theta > self.theta_limit_rads)
            | (theta < -self.theta_limit_rads)
        )
        reward = jnp.float32(jnp.invert(done))

        env_state = new_state, key
        env_state = self._reset_if_done(env_state, done)
        new_state = env_state[0]

        return env_state, self._get_obs(new_state), reward, done

    @jax.jit  # noqa: F811
    def reset(self, key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Resets the environment state and returns the initial observation.

        Args:
            key: JAX PRNG key.

        Returns:
            A tuple (env_state, obs), where env_state is the new environment state and
            obs is the initial observation based on the new state.
        """
        env_state = self._reset(key)
        new_state = env_state[0]
        return env_state, self._get_obs(new_state)
