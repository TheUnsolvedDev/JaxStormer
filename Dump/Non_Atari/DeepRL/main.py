import argparse
import os
import jax
import numpy as np
import jax.numpy as jnp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', '--name', type=str,
                        default='CartPole-v1', help='Environment name')
    parser.add_argument('-time', '--timestep', type=int,
                        default=1_000_000, help='Timestep')
    parser.add_argument('-seed', '--random_seed', type=int,
                        default=0, help='Random seed')
    parser.add_argument('-envs', '--num_envs', type=int,
                        default=1, help='Number of envs')
    parser.add_argument('-gpu', '--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('-agent', '--agent', type=str,
                        default='DQN', help='Agent')
    parser.add_argument('-lr', '--learning_rate', type=float,
                        default=1e-3, help='Learning rate')
    parser.add_argument('-gamma', '--gamma', type=float,
                        default=0.99, help='Gamma')
    parser.add_argument('-epsilon', '--epsilon', type=float,
                        default=1.0, help='Epsilon')
    parser.add_argument('-epsilon_decay', '--epsilon_decay',
                        type=float, default=0.999, help='Epsilon decay')
    parser.add_argument('-epsilon_min', '--epsilon_min',
                        type=float, default=0.01, help='Epsilon min')
    parser.add_argument('-batch_size', '--batch_size',
                        type=int, default=64, help='Batch size')
    parser.add_argument('-te', '--target_update', type=int,
                        default=20, help='Target update')
    parser.add_argument('-buffer_size', '--buffer_size',
                        type=int, default=100_000, help='Buffer size')


if __name__ == '__main__':
    pass
