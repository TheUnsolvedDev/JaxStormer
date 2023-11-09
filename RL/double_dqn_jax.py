from import_packages import *

algorithm = __file__.split('/')[-1][:-3]
key = jax.random.PRNGKey(0)
rng, key_reset, key_policy, key_step = jax.random.split(key, 4)

wandb.init(
    project=algorithm+'_'+ENV,

    config={
        'ENV': ENV,
        'ALPHA': ALPHA,
        'GAMMA': GAMMA,
        'TOTAL_TIME_STEPS': TOTAL_TIME_STEPS,
        'LEARNING_START': LEARNING_START,
        'BUFFER_SIZE': BUFFER_SIZE,
        'EPSILON': EPSILON,
        'TRAIN_FREQUENCY': TRAIN_FREQUENCY,
        'UPDATE_TARGET_FREQUENCY': UPDATE_TARGET_FREQUENCY,
        'BATCH_SIZE': BATCH_SIZE,
        'NUM_ENVS': NUM_ENVS,
        'TAU': TAU,
        'MAX_EPISODE_STEPS': MAX_EPISODE_STEPS
    }
)


class Env:
    def __init__(self, num_envs=8) -> None:
        self.env, self.env_params = gymnax.make(ENV)
        self.vmap_keys = jax.random.split(rng, num_envs)
        self.vmap_reset = jax.vmap(self.env.reset, in_axes=(0, None))
        self.vmap_step = jax.vmap(self.env.step, in_axes=(0, 0, 0, None))

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self):
        obs, state = self.vmap_reset(self.vmap_keys, self.env_params)
        return obs, state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        next_obs, next_state, reward, done, _ = self.vmap_step(
            self.vmap_keys, state, action, self.env_params)
        return next_obs, next_state, reward, done, _


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        # batch_size //= NUM_ENVS
        return jax.random.choice(jax.random.PRNGKey(np.random.randint(low=0, high=100)), len(self.buffer), shape=(batch_size,), replace=False)

    def get_batch(self, indices):
        states, actions, rewards, next_states, dones = zip(
            *[self.buffer[i] for i in indices])
        states = jnp.array(states).reshape((NUM_ENVS*len(indices), -1))
        actions = jnp.array(actions).reshape((NUM_ENVS*len(indices), -1))
        rewards = jnp.array(rewards).reshape((NUM_ENVS*len(indices), -1))
        next_states = jnp.array(next_states).reshape(
            (NUM_ENVS*len(indices), -1))
        dones = jnp.array(dones).reshape((NUM_ENVS*len(indices), -1))
        return states, actions, rewards, next_states, dones


class Q(flax.linen.Module):
    action_dim: int

    @flax.linen.compact
    def __call__(self, x: jnp.ndarray):
        x = flax.linen.Dense(120)(x)
        x = flax.linen.relu(x)
        x = flax.linen.Dense(84)(x)
        x = flax.linen.relu(x)
        x = flax.linen.Dense(self.action_dim)(x)
        return x


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def init_model(env):
    q_network = Q(env.num_actions)
    q_state = TrainState.create(
        apply_fn=q_network.apply,
        params=q_network.init(key, jnp.zeros(env.obs_shape)),
        target_params=q_network.init(
            key, jnp.zeros(env.obs_shape)),
        tx=optax.adam(learning_rate=ALPHA),
    )
    q_network.apply = jax.jit(q_network.apply)
    q_state = q_state.replace(target_params=optax.incremental_update(
        q_state.params, q_state.target_params, 1))
    print(q_network.tabulate(key, jnp.ones(
        env.obs_shape)))
    return q_network, q_state


def test(policy, q_state, step, num_games=10, name=ENV, video_stats=False):
    if video_stats:
        env = gym.make(name, max_episode_steps=BUFFER_SIZE //
                       10, render_mode='rgb_array')
        env = gym.wrappers.RecordVideo(
            env, f"RL_updates/{algorithm}/{name}/{step}/time_{int(time.time())}", episode_trigger=lambda x: x % int(5) == 0)
    else:
        env = gym.make(name, max_episode_steps=BUFFER_SIZE//10)
    rewards = []

    for game in (range(num_games)):
        sum_reward = 0
        state, _ = env.reset()
        for i in range(BUFFER_SIZE//10):
            action = policy(q_state, state, epsilon=0.01, test=True)
            next_state, reward, done, trauncated, infos = env.step(action)
            sum_reward += reward
            state = next_state
            if done:
                break
        rewards.append(sum_reward)
    env.close()
    return np.mean(rewards)


class DQN:
    def __init__(self) -> None:
        self.env = Env(num_envs=NUM_ENVS)
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.q_network, self.q_state = init_model(self.env.env)

    @functools.partial(jax.jit, static_argnums=(0,))
    def policy_greedy(self, q_state, state):
        q_values = self.q_network.apply(q_state.params, state)
        action = q_values.argmax(axis=-1)
        return action

    def policy(self, q_state, state, epsilon=0.1, test=False):
        if np.random.uniform() < epsilon:
            if test:
                return np.random.randint(0, self.env.env.num_actions)
            return jax.random.randint(key, minval=0, maxval=self.env.env.num_actions, shape=(NUM_ENVS,))
        return jax.device_get(self.policy_greedy(q_state, state))

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(self, q_state, states, actions, rewards, next_states,  dones):
        q_next_target = self.q_network.apply(
            q_state.target_params, next_states)
        q_next_pred = self.q_network.apply(q_state.params, next_states)
        q_next_actions = jnp.argmax(q_next_pred, axis=-1)
        q_next_target = q_next_target[jnp.arange(
            q_next_pred.shape[0]), q_next_actions.squeeze()]
        next_q_value = (rewards + (1 - dones) * GAMMA * q_next_target)

        def mse_loss(params):
            q_pred = self.q_network.apply(params, states)
            q_pred = q_pred[jnp.arange(q_pred.shape[0]), actions.squeeze()]
            return ((jax.lax.stop_gradient(next_q_value) - q_pred) ** 2).mean(), q_pred

        (loss_value, q_pred), grads = jax.value_and_grad(
            mse_loss, has_aux=True)(q_state.params)
        q_state = q_state.apply_gradients(grads=grads)
        return loss_value, jnp.mean(q_pred), q_state

    def train(self):
        state, obs = self.env.reset()
        epsilon = EPSILON
        start = time.time()

        for step in tqdm.tqdm(range(TOTAL_TIME_STEPS+1)):
            action = self.policy(self.q_state, state, epsilon)
            next_state, next_obs, reward, done, _ = self.env.step(
                obs, action)
            self.replay_buffer.push([state, action, reward, next_state, done])
            state = next_state
            obs = next_obs

            if step > LEARNING_START:
                epsilon = linear_schedule(
                    start_e=EPSILON, end_e=0.05, duration=TOTAL_TIME_STEPS*0.5, t=step)
                if not step % TRAIN_FREQUENCY:
                    indices = self.replay_buffer.sample(BATCH_SIZE)
                    states, actions, rewards, next_states, dones = self.replay_buffer.get_batch(
                        indices)
                    loss_values, q_pred, self.q_state = self.update(
                        self.q_state, states, actions, rewards, next_states,  dones)

                if not step % UPDATE_TARGET_FREQUENCY:
                    self.q_state = self.q_state.replace(target_params=optax.incremental_update(
                        self.q_state.params, self.q_state.target_params, 1))

                    if not step % (UPDATE_TARGET_FREQUENCY)*10:
                        test_results = test(
                            self.policy, self.q_state, step, video_stats=True)
                        # print("td_loss:", jax.device_get(loss_values),
                        #       "Q_value:", q_pred, 'Position:', test_results, 'Epsilon:', epsilon)
                        with open(f'RL_updates/{algorithm}/{ENV}/logs_{start}.txt', 'a', newline='') as csv_file:
                            csv_writer = csv.writer(csv_file)
                            csv_writer.writerow(
                                [loss_values, q_pred, test_results, epsilon])
                        wandb.log({
                            "TD Loss": loss_values,
                            "Q Values": q_pred,
                            "Rewards Gathered": test_results,
                            "Epsilon": epsilon
                        })

                    from flax.training import orbax_utils
                    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                    ckpt = {'q_state': self.q_state}
                    save_args = orbax_utils.save_args_from_target(ckpt)
                    orbax_checkpointer.save(
                        f'RL_updates/{algorithm}/{ENV}/{step}/{int(time.time())}', ckpt, save_args=save_args)


if __name__ == '__main__':
    agent = DQN()
    agent.train()
