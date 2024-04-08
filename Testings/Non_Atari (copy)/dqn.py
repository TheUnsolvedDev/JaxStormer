from import_packages import *

import models
import memory
import environment
from utils import *

key = jax.random.PRNGKey(0)

rng, key_reset, key_policy, key_step = jax.random.split(key, 4)


class DeepQNetwork:
    def __init__(self, env='CartPole-v1', num_envs=8, strategy="DQN", buffer_size=10_000, model_type='small', use_per=True, learning_start=5000, total_time_steps=1_000_000, gamma=0.99, alpha=0.001, epsilon=1, train_frequency=10, update_frequency=500, save_test=1000, batch_size=128, wandb_log=False):
        self.env_name = env
        self.num_envs = num_envs
        self.env = environment.Env(self.env_name, self.num_envs)
        self.strategy = strategy
        self.buffer_size = buffer_size
        self.model_type = model_type
        self.use_per = use_per
        self.learning_start = learning_start
        self.total_time_steps = total_time_steps

        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

        self.train_frequency = train_frequency
        self.update_frequency = update_frequency
        self.save_test = save_test
        self.batch_size = batch_size
        self.wandb_log = wandb_log

        if not self.use_per:
            self.replay_buffer = memory.ReplayBuffer(
                self.buffer_size, self.num_envs)
        else:
            self.per_buffer = memory.PERMemory(self.buffer_size)
        self.q_network, self.q_state = self._init_model(self.env.env)

    def _init_model(self, env):
        if self.model_type == 'large':
            q_network = models.Q_model_large(env.num_actions)
        elif self.model_type == 'medium':
            q_network = models.Q_model_medium(env.num_actions)
        elif self.model_type == 'small':
            q_network = models.Q_model_small(env.num_actions)

        q_state = TrainState.create(
            apply_fn=q_network.apply,
            params=q_network.init(key, jnp.zeros(env.obs_shape)),
            target_params=q_network.init(
                key, jnp.zeros(env.obs_shape)),
            tx=optax.adam(learning_rate=self.alpha),
        )
        q_network.apply = jax.jit(q_network.apply)
        q_state = q_state.replace(target_params=optax.incremental_update(
            q_state.params, q_state.target_params, 1))
        print(q_network.tabulate(key, jnp.ones(
            env.obs_shape)))
        return q_network, q_state

    @functools.partial(jax.jit, static_argnums=(0,))
    def policy_greedy(self, q_state, state):
        q_values = self.q_network.apply(q_state.params, state)
        action = q_values.argmax(axis=-1)
        return action

    def policy(self, q_state, state, epsilon=0.1, test=False):
        if np.random.uniform() < epsilon:
            if test:
                return np.random.randint(0, self.env.env.num_actions)
            return jax.random.randint(key, minval=0, maxval=self.env.env.num_actions, shape=(self.num_envs,))
        return jax.device_get(self.policy_greedy(q_state, state))

    @functools.partial(jax.jit, static_argnums=(0,))
    def td_error(self, q_state, states, actions, rewards, next_states,  dones):
        q_next_target = jnp.max(self.q_network.apply(
            q_state.target_params, next_states), axis=-1)
        q_pred = self.q_network.apply(q_state.params, states)
        td_target = rewards + self.gamma*jnp.amax(q_next_target)
        td_err = td_target - \
            q_pred[jnp.arange(q_pred.shape[0]), actions.squeeze()]
        return jnp.abs(td_err)

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(self, q_state, states, actions, rewards, next_states,  dones):
        q_next_target = self.q_network.apply(
            q_state.target_params, next_states)
        q_next_target = jnp.max(q_next_target, axis=-1)
        next_q_value = (rewards + (1 - dones) * self.gamma * q_next_target)

        def mse_loss(params):
            q_pred = self.q_network.apply(params, states)
            q_pred = q_pred[jnp.arange(q_pred.shape[0]), actions.squeeze()]
            return ((jax.lax.stop_gradient(next_q_value) - q_pred) ** 2).mean(), q_pred

        (loss_value, q_pred), grads = jax.value_and_grad(
            mse_loss, has_aux=True)(q_state.params)
        q_state = q_state.apply_gradients(grads=grads)
        if not self.use_per:
            return loss_value, jnp.mean(q_pred), q_state, None
        td_error_abs = self.td_error(
            q_state, states, actions, rewards, next_states,  dones)
        return loss_value, jnp.mean(q_pred), q_state, td_error_abs

    def train(self):
        state, obs = self.env.reset()
        epsilon = self.epsilon

        for step in tqdm.tqdm(range(self.total_time_steps+1)):
            action = self.policy(self.q_state, state, epsilon)
            next_state, next_obs, reward, done, _ = self.env.step(
                obs, action)

            if not self.use_per:
                self.replay_buffer.push(
                    [state, action, reward, next_state, done])

            if self.use_per:
                td = self.td_error(self.q_state,
                                   jnp.asarray(state),
                                   jnp.asarray(action),
                                   jnp.asarray(reward),
                                   jnp.asarray(next_state),
                                   jnp.asarray(done)
                                   )
                for i in range(self.num_envs):
                    self.per_buffer.add(
                        td[i], (state[i], action[i], reward[i], next_state[i], done[i]))

            state = next_state
            obs = next_obs

            if step > self.learning_start:
                epsilon = linear_schedule(
                    start_e=self.epsilon, end_e=0.05, duration=self.total_time_steps*0.5, t=step)
                if not step % self.train_frequency:
                    if not self.use_per:
                        indices = self.replay_buffer.sample(self.batch_size)
                        states, actions, rewards, next_states, dones = self.replay_buffer.get_batch(
                            indices)
                        loss_values, q_pred, self.q_state, _ = self.update(
                            self.q_state, states, actions, rewards, next_states,  dones)
                    if self.use_per:
                        batch = self.per_buffer.sample(
                            self.batch_size)
                        states, actions, rewards, next_states, dones = self.per_buffer.get_data(
                            batch, self.batch_size)
                        loss_values, q_pred, self.q_state, new_td = self.update(
                            self.q_state, states, actions, rewards, next_states,  dones)
                        for i in range(self.batch_size):
                            self.per_buffer.update(batch[i][0], new_td[i])

                if not step % self.update_frequency:
                    self.q_state = self.q_state.replace(target_params=optax.incremental_update(
                        self.q_state.params, self.q_state.target_params, 1))
                    if not step % self.save_test:
                        test_results = self.test(
                            self.policy, self.q_state, step, name=self.env_name, video_stats=False)
                        print("td_loss:", jax.device_get(loss_values),
                              "Q_value:", q_pred, 'Position:', test_results, 'Epsilon:', epsilon)
                        if self.wandb_log:
                            wandb.log({'results': float(test_results)})
                            wandb.log(
                                {'loss': float(jax.device_get(loss_values))})
                            wandb.log({'Q value': float(q_pred)})
                        save_data(file=f'RL_updates/{self.strategy}/{self.env_name}/logs.txt', data=[step, jax.device_get(
                            loss_values), q_pred, test_results])

    def test(self, policy, q_state, step, num_games=10, name='Acrobot-v1', video_stats=False):
        if video_stats:
            env = gym.make(name, max_episode_steps=self.buffer_size //
                           10, render_mode='rgb_array')
            env = gym.wrappers.RecordVideo(
                env, f"RL_updates/{self.strategy}/{name}/{step}/time_{int(time.time())}", episode_trigger=lambda x: x % int(5) == 0)
        else:
            env = gym.make(name, max_episode_steps=self.buffer_size//10)
        rewards = []

        for game in (range(num_games)):
            sum_reward = 0
            state, _ = env.reset()
            for i in range(self.buffer_size//10):
                action = policy(q_state, state, epsilon=0.01, test=True)
                next_state, reward, done, trauncated, infos = env.step(action)
                sum_reward += reward
                state = next_state
                if done:
                    break
            rewards.append(sum_reward)
        env.close()
        return np.mean(rewards)
