\
import Agent.Reinforce
import config
import gymnax
import jax

if __name__ == '__main__':
    cartpole_env, cartpole_env_params = gymnax.make('CartPole-v1')

    reinforce = Agent.Reinforce.ReinforceAlgorithm(
        env=cartpole_env,
        env_params = cartpole_env_params,
        num_actions=2,
        observation_shape=(4,),
        max_episode_length=1000,
        gamma=0.99,
        alpha=1e-5,
        weight_decay=0.01,
        tau=0.01
    )
    # states,rewards ,dones = reinforce.generate_trajectory()
    # print(states[:50],dones[:50],rewards[:50])
    reinforce.train()
