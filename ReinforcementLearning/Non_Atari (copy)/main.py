from import_packages import *
from colorama import Fore

import dqn
import wandb

# print('Num Devices:', jax.device_count())
# print('Num local devices:', jax.local_device_count())
# print('Devices:', jax.devices())
# os.environ['CUDA_VISIBLE_DEVICES'] = str([args.gpu])


def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Arguments')
    parser.add_argument('--env', type=str,
                        default='CartPole-v1', help='Environment name')
    parser.add_argument('--alpha', type=float, default=2.5e-4,
                        help='Learning rate (ALPHA)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (GAMMA)')
    parser.add_argument('--total_time_steps', type=int,
                        default=int(5e+5), help='Total time steps (TOTAL_TIME_STEPS)')
    parser.add_argument('--learning_start', type=int, default=int(10000),
                        help='Learning start time steps (LEARNING_START)')
    parser.add_argument('--buffer_size', type=int, default=int(1e+4),
                        help='Replay buffer size (BUFFER_SIZE)')
    parser.add_argument('--epsilon', type=float, default=1,
                        help='Exploration parameter (EPSILON)')
    parser.add_argument('--train_frequency', type=int,
                        default=10, help='Training frequency (TRAIN_FREQUENCY)')
    parser.add_argument('--update_target_frequency', type=int, default=500,
                        help='Target network update frequency (UPDATE_TARGET_FREQUENCY)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size (BATCH_SIZE)')
    parser.add_argument('--num_envs', type=int, default=8,
                        help='Number of parallel environments (NUM_ENVS)')
    parser.add_argument('--tau', type=float, default=0.9,
                        help='Soft update parameter (TAU)')
    parser.add_argument('--max_episode_steps', type=int,
                        default=1000, help='Total Number of steps in an episode')
    parser.add_argument('--gpu', type=int, default=0, help='Number of GPU')
    parser.add_argument('--n_workers', type=int, default=4,
                        help='Number of Workers')
    parser.add_argument('--save_test', type=int, default=1000,
                        help='Run the testing after n steps')
    parser.add_argument('--algorithm', type=str,
                        default='DQN', help='Choose the strategy!')
    parser.add_argument('--use_per', type=bool, default=False,
                        help='Use PER(Prioritized Experience Replay)')
    parser.add_argument('--model_type', type=str,
                        default='small', help='Choose Model size')
    parser.add_argument('--wandb_log', type=bool,
                        default=False, help='Log for Wandb')
    args = parser.parse_args()
    return args


args = parse_arguments()


def train(args):
    ENV = args.env
    ALPHA = args.alpha
    GAMMA = args.gamma
    TOTAL_TIME_STEPS = args.total_time_steps
    LEARNING_START = args.learning_start
    BUFFER_SIZE = args.buffer_size
    EPSILON = args.epsilon
    TRAIN_FREQUENCY = args.train_frequency
    UPDATE_TARGET_FREQUENCY = args.update_target_frequency
    BATCH_SIZE = args.batch_size
    NUM_ENVS = args.num_envs
    TAU = args.tau
    MAX_EPISODE_STEPS = args.max_episode_steps
    SAVE_TEST = args.save_test
    ALGORITHM = args.algorithm
    PER = args.use_per
    MODEL_TYPE = args.model_type
    WANDB_LOG = args.wandb_log

    print("************************"+("*"*len(ENV)))
    print(Fore.RED+f'Training started on ENV {Fore.BLUE + ENV}'+Fore.WHITE)
    print("************************"+("*"*len(ENV)))

    if ALGORITHM == 'DQN':
        agent = dqn.DeepQNetwork(env=ENV, num_envs=NUM_ENVS, strategy=ALGORITHM, buffer_size=BUFFER_SIZE, model_type=MODEL_TYPE, use_per=PER, learning_start=LEARNING_START,
                                 total_time_steps=TOTAL_TIME_STEPS, gamma=GAMMA, alpha=ALPHA, epsilon=EPSILON, train_frequency=TRAIN_FREQUENCY, update_frequency=UPDATE_TARGET_FREQUENCY, save_test=SAVE_TEST, batch_size=BATCH_SIZE, wandb_log=WANDB_LOG)
        agent.train()

    if ALGORITHM == 'DoubleDQN':
        raise NotImplementedError

    if ALGORITHM == 'DuelingDQN':
        raise NotImplementedError


def main():
    with open('./sweep.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    run = wandb.init(config=config)
    config = wandb.config

    ENV = config.env
    ALPHA = config.alpha
    GAMMA = config.gamma
    TOTAL_TIME_STEPS = config.total_time_steps
    LEARNING_START = config.learning_start
    BUFFER_SIZE = config.buffer_size
    EPSILON = config.epsilon
    TRAIN_FREQUENCY = config.train_frequency
    UPDATE_TARGET_FREQUENCY = config.update_target_frequency
    BATCH_SIZE = config.batch_size
    NUM_ENVS = config.num_envs
    TAU = config.tau
    MAX_EPISODE_STEPS = config.max_episode_steps
    SAVE_TEST = config.save_test
    ALGORITHM = config.algorithm
    PER = config.use_per
    MODEL_TYPE = config.model_type
    WANDB_LOG = config.wandb_log

    print("************************"+("*"*len(ENV)))
    print(Fore.RED+f'Training started on ENV {Fore.BLUE + ENV}'+Fore.WHITE)
    print("************************"+("*"*len(ENV)))

    if ALGORITHM == 'DQN':
        agent = dqn.DeepQNetwork(env=ENV, num_envs=NUM_ENVS, strategy=ALGORITHM, buffer_size=BUFFER_SIZE, model_type=MODEL_TYPE, use_per=PER, learning_start=LEARNING_START,
                                 total_time_steps=TOTAL_TIME_STEPS, gamma=GAMMA, alpha=ALPHA, epsilon=EPSILON, train_frequency=TRAIN_FREQUENCY, update_frequency=UPDATE_TARGET_FREQUENCY, save_test=SAVE_TEST, batch_size=BATCH_SIZE, wandb_log=WANDB_LOG)
        agent.train()


if __name__ == '__main__':
    main()
