import os
import sys

# file_dir = os.path.dirname(__file__)
# sys.path.append(file_dir)

import gym
from model import *
from dqn import trainDQN
from atari_wrapper import make_env

# print(f'device: {device}')

dqn_config = dict(
    eps_i=1,
    eps_f=0.01,
    eps_decay=200000,
    gamma=0.99,
    lr=1e-4,
    num_iter=int(2e6),
    target_update_freq=1000,
    memory_size=50000,
    batch_size=32,
    target_reward=19.5,
    training_start=10000
)

ppo_config = dict(
      clip_eps=0.1,
      p_train_iter=2,
      epoch=5,
      time_steps=10,
      mini_batch=5,
      lr=1e-3,
      gamma=0.95,
      gae_lambda=0.95
)

conv_config = dict(
    conv_filters=[
        [32, 8, 4],
        [64, 4, 2],
        [64, 3, 1]
    ],
    fc_sizes=[512],
    initialize=False
)

train_config = dict(
    train_alg='dqn',
    model='conv',
    save_freq=25,
    eval_freq=25,
    eval_num=10
)

if train_config["model"] == 'conv':
    agent = ConvNet2
    agent_config = conv_config

if train_config["train_alg"] == 'dqn':
    train_config.update(dqn_config)
    trainer = trainDQN
elif train_config["train_alg"] == 'ppo':
    train_config.update(ppo_config)
    trainer = ppo

ENV_NAME = "PongNoFrameskip-v4"

if __name__ == "__main__":
    env = make_env(ENV_NAME)
    agent_config["input_shape"] = env.observation_space.shape
    agent_config["action_space"] = env.action_space.n
    # checkpoint = None
    checkpoint = 'drive/MyDrive/rl_colab/train_results/checkpoint-550.pt'
    trainer(env, agent, agent_config, train_config, checkpoint)
