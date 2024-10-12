import jax
import flax
import optax
import gymnax
import gc
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import functools
from flax.training.train_state import TrainState
import jax_tqdm
import pandas as pd
from typing import Dict
from typing import Tuple

ALPHA = 0.001
GAMMA = 0.99
BATCH_SIZE = 64
BUFFER_SIZE = 10000
UPDATE_EVERY = 25
NUM_ENVS = 16
rng = jax.random.PRNGKey(0)
cpu = jax.devices('cpu')[0]
gpu = jax.devices('gpu')[0]


def smooth_rewards(rewards: np.ndarray, window_size: int = 10) -> np.ndarray:
    """
    Smooths out rewards using a moving average window defined by window_size.

    Parameters:
        rewards (np.ndarray): The array of rewards to be smoothed.
        window_size (int): The size of the moving average window (default is 10).

    Returns:
        np.ndarray: Smoothed rewards array after applying the moving average.
    """
    smoothed_rewards = np.zeros_like(rewards)
    for i in range(len(rewards)):
        window_start = max(0, i - window_size // 2)
        window_end = min(len(rewards), i + window_size // 2 + 1)
        smoothed_rewards[i] = np.mean(rewards[window_start:window_end])
    return smoothed_rewards


def plot_data(mean: np.ndarray, std: np.ndarray, name: str) -> None:
    """
    Plots the data with mean, standard deviation, and name.

    Parameters:
    mean (array): Array containing the mean values.
    std (array): Array containing the standard deviation values.
    name (str): Name of the plot.

    Returns:
    None
    """
    x = np.arange(len(mean))

    plt.plot(x, mean, color='blue', label='Mean')
    plt.plot(x, smooth_rewards(mean), color='orange', label='smoothed')
    plt.fill_between(x, mean - std, mean + std, color='blue',
                     alpha=0.3, label='Mean ± Std')

    plt.xlabel('Steps')
    plt.ylabel('Rewards')
    plt.title(name)
    plt.legend()
    plt.grid(True)
    plt.savefig(name.replace(' ', '_')+'.png')
    # plt.show(block = False)
    plt.close()


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


class ValueNetwork(flax.linen.Module):
    action_dim: int

    @flax.linen.compact
    def __call__(
        self, x: jnp.ndarray  # type: ignore
    ) -> jnp.ndarray:  # type: ignore
        """
        A function that represents the forward pass of the neural network.

        Parameters:
            x (jnp.ndarray): The input tensor to the network.

        Returns:
            jnp.ndarray: The output tensor of the network, representing the Q-values for each action.
        """
        x = flax.linen.Dense(16)(x)
        x = flax.linen.selu(x)
        x = flax.linen.Dense(16)(x)
        x = flax.linen.selu(x)
        q_values = flax.linen.Dense(self.action_dim)(x)
        return q_values


class UniformReplayBuffer():
    def __init__(
        self,
        buffer_size: int,  # type: ignore
        batch_size: int,  # type: ignore
        state_shape: int,  # type: ignore
    ) -> None:
        """
        Initialize a uniform replay buffer.

        Parameters:
            buffer_size (int): The maximum number of experiences to store.
            batch_size (int): The size of the batches to sample.
            state_shape (int): The shape of the state space.
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer_state = {
            "states": jnp.empty((buffer_size*NUM_ENVS, state_shape), dtype=jnp.float32),
            "actions": jnp.empty((buffer_size*NUM_ENVS,), dtype=jnp.int32),
            "rewards": jnp.empty((buffer_size*NUM_ENVS,), dtype=jnp.int32),
            "next_states": jnp.empty((buffer_size*NUM_ENVS, state_shape), dtype=jnp.float32),
            "dones": jnp.empty((buffer_size*NUM_ENVS,), dtype=jnp.bool_),
        }

    @functools.partial(jax.jit, static_argnums=(0))
    def add(self, buffer_state: Dict[str, jnp.ndarray], experience: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], idx: int, ) -> Dict[str, jnp.ndarray]:
        """
        Add an experience to the buffer.

        Parameters:
            buffer_state (Dict[str, jnp.ndarray]): The current buffer state.
            experience (Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]): The experience to add, represented as a tuple of states, actions, rewards, next states, and dones.
            idx (int): The current index to add the experience to.

        Returns:
            Dict[str, jnp.ndarray]: The updated buffer state.
        """
        state, action, reward, next_state, done = experience
        idx = (idx+NUM_ENVS) % self.buffer_size
        val_init = (buffer_state, state, action, reward, next_state, done)

        def fill(
            i: int,  # type: int
            # type: Tuple[Dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
            val: Tuple[Dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        ) -> Tuple[Dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:  # type: Tuple[Dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
            """
            Fill buffer with experience at index i.

            Parameters:
                i (int): The index to fill in the buffer.
                val (Tuple[Dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]): The current buffer state and the experience to add.

            Returns:
                Tuple[Dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]: The updated buffer state.
            """
            buffer_state, state, action, reward, next_state, done = val
            buffer_state["states"] = buffer_state["states"].at[idx + i].set(
                state[i])  # shape=(buffer_size*num_envs, state_shape)
            buffer_state["actions"] = buffer_state["actions"].at[idx +
                                                                 i].set(action[i])  # shape=(buffer_size*num_envs,)
            buffer_state["rewards"] = buffer_state["rewards"].at[idx +
                                                                 i].set(reward[i])  # shape=(buffer_size*num_envs,)
            buffer_state["next_states"] = buffer_state["next_states"].at[idx + i].set(
                next_state[i])  # shape=(buffer_size*num_envs, state_shape)
            buffer_state["dones"] = buffer_state["dones"].at[idx +
                                                             i].set(done[i])  # shape=(buffer_size*num_envs,)
            val = (buffer_state, state, action, reward, next_state, done)
            return val

        buffer_state = jax.lax.fori_loop(0, NUM_ENVS, fill, val_init)[0]
        return buffer_state

    @functools.partial(jax.jit, static_argnums=(0))
    def sample(  # type: ignore
        self,
        key: jnp.ndarray,  # shape=(2,), dtype=int32
        # type: ignore # shape=(buffer_size, num_envs, ...)
        buffer_state: Dict[str, jnp.ndarray],
        current_buffer_size: int,
    ) -> Tuple[Dict[str, jnp.ndarray], jnp.ndarray]:  # type: ignore # shape=(batch_size, num_envs, ...), shape=(2,)
        # iterate over the indexes
        @functools.partial(jax.vmap, in_axes=(0, None))
        def sample_batch(  # type: ignore
            indexes: jnp.ndarray,  # shape=(batch_size,), dtype=int32
            # type: ignore # shape=(buffer_size, num_envs, ...)
            buffer: Dict[str, jnp.ndarray],
        ) -> Dict[str, jnp.ndarray]:  # type: ignore # shape=(batch_size, num_envs, ...)
            """
            Sample a batch from the buffer.

            Parameters:
                indexes (jnp.ndarray): The indexes to sample from the buffer.
                buffer (Dict[str, jnp.ndarray]): The buffer to sample from.

            Returns:
                Dict[str, jnp.ndarray]: The sampled batch.
            """
            return jax.tree_map(lambda x: x[indexes], buffer)

        key, subkey = jax.random.split(key)
        indexes = jax.random.randint(subkey, shape=(self.batch_size,),
                                     minval=0,
                                     maxval=current_buffer_size+NUM_ENVS,
                                     dtype=jnp.int32,
                                     )
        experiences = sample_batch(indexes, buffer_state)
        return experiences, subkey


class DQN:
    def __init__(self, env, num_actions, obs_shape, seed=0):
        self.rng = jax.random.PRNGKey(seed)
        self.num_actions = num_actions
        self.obs_shape = obs_shape
        self.env = env

        self.value = ValueNetwork(self.num_actions)
        self.value_state = TrainState.create(
            apply_fn=self.value.apply,
            params=self.value.init(self.rng, jnp.ones(self.obs_shape)),
            target_params=self.value.init(self.rng, jnp.ones(self.obs_shape)),
            tx=optax.adam(ALPHA)
        )
        self.value.apply = jax.jit(self.value.apply)
        self.replay_buffer = UniformReplayBuffer(
            BUFFER_SIZE, BATCH_SIZE, self.obs_shape)
        self.counter = 1
        self.updates = 1
        self.value_state = self.soft_update(self.value_state)

    @functools.partial(jax.jit, static_argnums=(0))
    def soft_update(self, target_state: TrainState) -> TrainState:
        """
        Performs a soft update of the target network parameters towards the online network parameters.
        """
        return target_state.replace(
            target_params=optax.incremental_update(
                target_state.params, target_state.target_params, 0.9
            )
        )

    @functools.partial(jax.jit, static_argnums=(0))
    def act(self, rng: jnp.ndarray,
            value_params: TrainState,
            state: jnp.ndarray,
            epsilon: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
            rng: The JAX PRNG key.
            value_params: The training state of the value network.
            state: The current environment state.
            epsilon: The probability of choosing a random action.

        Returns:
            The action and the next PRNG key.
        """

        def _random_action(key: jnp.ndarray) -> jnp.ndarray:
            """
            Args:
                key: The JAX PRNG subkey.

            Returns:
                The action chosen by the epsilon-greedy policy.
            """
            return jax.random.choice(key, jnp.arange(self.num_actions), shape=(NUM_ENVS,))

        def _greedy_action(_):
            """
            Args:
                _: Dummy argument for `jax.lax.cond`.

            Returns:
                The action chosen by the greedy policy.
            """
            q_values = self.value.apply(value_params.params, state)
            return jnp.argmax(q_values, axis=1)

        explore = jax.random.uniform(rng) < epsilon
        rng, subkey = jax.random.split(rng)
        action = jax.lax.cond(explore, _random_action,
                              _greedy_action, operand=subkey)
        return action, rng

    @functools.partial(jax.jit, static_argnums=(0))
    def batch_act(
        self,
        key: jnp.ndarray,  # shape=(2,)
        value_state: TrainState,
        state: jnp.ndarray,  # shape=(num_envs,) + observation_shape
        epsilon: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:  # shape=(num_envs,)
        """
        Args:
            key: The JAX PRNG key.
            value_state: The training state of the value network.
            state: The current environment state.
            epsilon: The probability of choosing a random action.

        Returns:
            The actions and the next PRNG key.
        """
        return jax.vmap(self.act, in_axes=(0, None, 0, 0))(key, value_state, state, epsilon)

    @functools.partial(jax.jit, static_argnums=(0))
    def update(  # type: (DQN, TrainState, Dict[str, jnp.ndarray]) -> Tuple[float, TrainState]
            self, value_state: TrainState, experiences: Dict[str, jnp.ndarray]) -> Tuple[float, TrainState]:
        """
        Args:
            value_state: The training state of the value network.
            experiences: A dictionary of experiences with keys:
                `states`, `actions`, `rewards`, `next_states`, and `dones`.

        Returns:
            The loss and the updated training state.
        """
        @jax.jit
        def _batch_loss_fn(
                online_params: jnp.ndarray,  # shape=(num_parameters,)
                target_params: jnp.ndarray,  # shape=(num_parameters,)
                states: jnp.ndarray,  # shape=(batch_size, num_observations)
                actions: jnp.ndarray,  # shape=(batch_size,)
                rewards: jnp.ndarray,  # shape=(batch_size,)
                # shape=(batch_size, num_observations)
                next_states: jnp.ndarray,
                dones: jnp.ndarray,  # shape=(batch_size,)
        ) -> float:
            """
            Computes the batch loss.

            Args:
                online_params (jnp.ndarray): The parameters of the online network.
                target_params (jnp.ndarray): The parameters of the target network.
                states (jnp.ndarray): The states.
                actions (jnp.ndarray): The actions.
                rewards (jnp.ndarray): The rewards.
                next_states (jnp.ndarray): The next states.
                dones (jnp.ndarray): Whether the episode is complete.

            Returns:
                The batch loss.
            """
            @functools.partial(jax.vmap, in_axes=[None, None, 0, 0, 0, 0, 0])
            def _loss_fn(  # type: (jnp.ndarray, jnp.ndarray, int, int, float, jnp.ndarray, bool) -> float
                    online_params: jnp.ndarray,  # shape=(num_parameters,)
                    target_params: jnp.ndarray,  # shape=(num_parameters,)
                    state: jnp.ndarray,  # shape=(num_observations,)
                    action: int,
                    reward: float,
                    next_state: jnp.ndarray,  # shape=(num_observations,)
                    done: bool,
            ) -> float:
                """
                Computes the loss for a single experience.

                Args:
                    online_params (jnp.ndarray): The parameters of the online network.
                    target_params (jnp.ndarray): The parameters of the target network.
                    state (jnp.ndarray): The state.
                    action (int): The action.
                    reward (float): The reward.
                    next_state (jnp.ndarray): The next state.
                    done (bool): Whether the episode is complete.

                Returns:
                    The loss for the experience.
                """
                target = reward + (1-done)*GAMMA * \
                    jnp.max(self.value.apply(target_params, next_state))
                target = jax.lax.stop_gradient(target)
                prediction = self.value.apply(online_params, state)[action]
                return jnp.square(target - prediction)
            return jnp.mean(_loss_fn(online_params, target_params, states, actions, rewards, next_states, dones))

        loss, grads = jax.value_and_grad(_batch_loss_fn)(
            value_state.params, value_state.target_params, **experiences)
        value_state = value_state.apply_gradients(grads=grads)
        return loss, value_state

    @functools.partial(jax.jit, static_argnums=(0))
    def batch_update(  # type: (TrainState, Mapping[str, jnp.ndarray]) -> Tuple[float, TrainState]
            self,
            value_state: TrainState,  # shape=(num_parameters,)
            experiences,
    ) -> Tuple[float, TrainState]:
        """
        Updates the value network using a batch of experiences.

        Args:
            value_state (TrainState): The current value function.
            experiences (Mapping[str, jnp.ndarray]): The experiences.

        Returns:
            The loss and the updated value function.
        """
        return jax.vmap(self.update, in_axes=(0, 0))(value_state, experiences)


class EnvJax:
    def __init__(
        self,
        name: str,
        seed: int = 0,
        num_envs: int = 1,
    ) -> None:
        """
        Initializes the environment.

        Args:
            name (str): The name of the environment.
            seed (int, optional): The random seed. Defaults to 0.
            num_envs (int, optional): The number of environments to simulate. Defaults to 1.
        """
        self.name = name
        self.rng = jax.random.PRNGKey(seed)
        self.env, self.env_params = gymnax.make(name)
        self.state_shape: int = self.env.obs_shape[0]
        self.action_shape: int = self.env.num_actions
        self.vmap_reset = jax.vmap(self.env.reset, in_axes=(0, None))
        self.vmap_step = jax.vmap(self.env.step, in_axes=(0, 0, 0, None))
        self.vmap_keys = jax.random.split(self.rng, num_envs)

    @functools.partial(jax.jit, static_argnums=(0), device=gpu)
    def reset(  # type: (EnvJax) -> jnp.ndarray
            self,  # type: EnvJax
    ) -> jnp.ndarray:
        """
        Resets the environment.

        Returns:
            states (jnp.ndarray): The initial states of the environment.
                shape=(num_envs, num_observations)
        """
        return self.vmap_reset(self.vmap_keys, self.env_params)

    @functools.partial(jax.jit, static_argnums=(0), device=gpu)
    def step(  # type: (EnvJax, jnp.ndarray, jnp.ndarray) -> jnp.ndarray
            self,  # type: EnvJax
            rngs,
            states: jnp.ndarray,  # type: jnp.ndarray
            actions: jnp.ndarray,  # type: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Steps the environment.

        Args:
            states (jnp.ndarray): The states of the environment.
                shape=(num_envs, num_observations)
            actions (jnp.ndarray): The actions to take in the environment.
                shape=(num_envs,)

        Returns:
            next_states (jnp.ndarray): The next states of the environment.
                shape=(num_envs, num_observations)
        """
        return self.vmap_step(rngs, states, actions, self.env_params)


@jax.jit
def linear_schedule(start_e: float, end_e: float, duration: int, t: int) -> float:
    """
    Calculates a linear schedule.

    Args:
        start_e (float): The starting value of the schedule.
        end_e (float): The end value of the schedule.
        duration (int): The duration of the schedule.
        t (int): The current time step of the schedule.

    Returns:
        e (float): The value of the schedule at time step t.
    """
    cond1 = (end_e - start_e) / duration * t + start_e
    cond2 = end_e
    return jax.lax.cond(cond1 > cond2, lambda: cond1, lambda: cond2)


def deep_roll_out(name: str = 'CartPole-v1',
                  timestep: int = 1_000_000,
                  random_seed: int = 0) -> None:
    """
    Performs a deep roll out of an environment.

    Args:
        name (str): The name of the environment.
        timestep (int): The number of timesteps to perform.
        random_seed (int): The random seed to use.
    """
    init_key, action_key, buffer_key = jax.vmap(
        jax.random.PRNGKey)(jnp.arange(3) + random_seed)
    env = EnvJax(name=name, seed=random_seed, num_envs=NUM_ENVS)
    state_shape = env.state_shape
    action_shape = env.action_shape
    state, obs = env.reset()

    states = jnp.zeros([timestep, state_shape])
    actions = jnp.zeros([timestep])
    rewards = jnp.zeros([timestep])
    dones = jnp.zeros([timestep])
    losses = jnp.zeros([timestep])

    agent = DQN(env, num_actions=action_shape, obs_shape=state_shape)
    buffer_state = agent.replay_buffer.buffer_state
    rngs = jax.random.split(init_key, NUM_ENVS)

    val = (losses, rewards, states, actions, dones, action_key,
           buffer_key, state, obs, buffer_state, rngs)

    @jax_tqdm.loop_tqdm(timestep)
    @jax.jit
    def for_loop_body(
            i: int,                                       # The current time step
            val: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
                       jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Performs a single time step in the environment.

        Args:
            i (int): The current time step.
            val (Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]): 
                The current buffer state and the experience to add.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]: 
                The updated buffer state.
        """
        (losses, rewards, states, actions, dones, action_key,
         buffer_key, state, obs, buffer_state, rngs) = val
        epsilon = linear_schedule(1, 0.01, timestep/2, i)
        action, action_key = agent.act(
            action_key, agent.value_state, state, epsilon)
        next_state, next_obs, reward, done, _ = env.step(rngs,
                                                         obs, action)
        reward *= (1 - done)

        experience = (state.reshape(NUM_ENVS, state_shape), action.reshape(NUM_ENVS,), reward.reshape(NUM_ENVS,),
                      next_state.reshape(NUM_ENVS, state_shape), done.reshape(NUM_ENVS,))
        buffer_state = agent.replay_buffer.add(buffer_state, experience, i)
        current_buffer_size = jnp.min(jnp.array([i, BUFFER_SIZE]))

        experiences, buffer_key = agent.replay_buffer.sample(
            buffer_key, buffer_state, current_buffer_size)
        loss, agent.value_state = agent.update(agent.value_state, experiences)

        states = states.at[i].set(state[0].reshape(-1,))
        actions = actions.at[i].set(action[0])
        rewards = rewards.at[i].set(reward[0])
        dones = dones.at[i].set(done[0])
        losses = losses.at[i].set(loss)

        agent.value_state = jax.lax.cond(
            i % UPDATE_EVERY == 0,
            lambda _: agent.soft_update(agent.value_state),
            lambda _: agent.value_state,
            operand=None,
        )

        rngs = jax.random.split(rngs[0], NUM_ENVS)
        state = next_state
        obs = next_obs

        val = (losses, rewards, states, actions, dones, action_key,
               buffer_key, state, obs, buffer_state, rngs)
        return val

    # for i in range(timestep):
    #     val = for_loop_body(i, val)

    vals = jax.lax.fori_loop(0, timestep, for_loop_body, val)
    (losses, rewards, states, actions, dones, action_key,
     buffer_key, state, obs, buffer_state, rngs) = vals

    df = pd.DataFrame(
        data={
            'episode': np.array(dones).cumsum(),
            'reward': np.array(rewards),
        }
    )
    print(df)
    df['episode'] = df['episode'].shift().fillna(0)
    episodes_df = df.groupby("episode").agg("sum")
    print(episodes_df)

    data = np.array(episodes_df)
    plt.plot(data)
    plt.show()


if __name__ == "__main__":
    deep_roll_out(name='CartPole-v1', timestep=1_000_000, random_seed=0)
    deep_roll_out(name='Acrobot-v1', timestep=1_000_000, random_seed=0)
