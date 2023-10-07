import gymnax
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import flax
import optax
import orbax
import tqdm
import csv
import os
import time
from flax.training.train_state import TrainState
import functools
import tensorflow as tf
import wandb
import argparse


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
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (BATCH_SIZE)')
    parser.add_argument('--num_envs', type=int, default=8,
                        help='Number of parallel environments (NUM_ENVS)')
    parser.add_argument('--tau', type=float, default=0.9,
                        help='Soft update parameter (TAU)')
    parser.add_argument('--max_episode_steps', type=int,
                        default=1000, help='Total Number of steps in an episode')
    parser.add_argument('--gpu', type=int, default=0, help='Number of GPU')
    args = parser.parse_args()
    return args


args = parse_arguments()

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
jax.default_device = jax.devices('gpu')[args.gpu]
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
