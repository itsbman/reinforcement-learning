from datetime import datetime
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from collections import deque, namedtuple
import math
from .logger import Logger
import time

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
TIME = datetime.now().strftime('%m%d%H%M')
save_dir = os.path.dirname(FILE_PATH + '/train_results/' + TIME)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


class ReplayBuffer(object):
    def __init__(self, capacity, batch_size):
        self.batch_size = batch_size
        self.memory_size = capacity
        self.memory = deque(maxlen=capacity)
        self.experience = namedtuple('Experience', ('state', 'reward', 'action',
                                                    'next_state', 'done'))

    def store(self, *transition):
        # if len(self.memory) < self.memory_size:
        experience = self.experience(*transition)
        self.memory.append(experience)

    def pick_sample(self):
        select_random_batch = random.sample(self.memory, self.batch_size)
        sample_batch = self.experience(*zip(*select_random_batch))
        # state_batch = torch.tensor(sample_batch.state)
        # action_batch = torch.LongTensor(sample_batch.action)
        # reward_batch = torch.tensor(sample_batch.reward)
        # next_state_batch = torch.tensor(sample_batch.next_state)
        # done_batch = torch.tensor(sample_batch.done)

        state_batch = np.array(sample_batch.state) / 255.
        action_batch = np.array(sample_batch.action)
        reward_batch = np.array(sample_batch.reward)
        next_state_batch = np.array(sample_batch.next_state) / 255.
        done_batch = np.array(sample_batch.done)
        return state_batch, reward_batch, action_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.memory)


def trainDQN(env, net_model, net_config, train_config, checkpoint=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eps_i, eps_f, eps_decay = train_config["eps_i"], train_config["eps_f"], train_config["eps_decay"]
    gamma = train_config["gamma"]
    n_actions = net_config["action_space"]

    mean_r, max_r, min_r = 0, 0, 0

    logger = Logger(save_dir)

    q_net = net_model(**net_config).to(device)
    target_net = net_model(**net_config).to(device)
    optimizer = optim.Adam(q_net._main_net.parameters(), lr=train_config["lr"])

    if checkpoint:
        model_checkpoint = torch.load(checkpoint)
        # q_net = net_model(**net_config)
        q_net.load_state_dict(model_checkpoint["q_state"])
        q_net.train()

        # target_net = net_model(**net_config)
        target_net.load_state_dict(model_checkpoint["target_state"])

        optimizer.load_state_dict(model_checkpoint["optim_state"])

        time_steps = model_checkpoint["timesteps"]
        eps = model_checkpoint["eps"]
    else:
        time_steps = 0
        eps = 0
        target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    memory = ReplayBuffer(train_config["memory_size"], train_config["batch_size"])

    def eps_exploration(state, episode):
        eps = eps_f + (eps_i - eps_f) * math.exp(-1. * episode / eps_decay)
        sample = random.random()

        if sample > eps:
            with torch.no_grad():
                state = np.array(state) / 255.
                state = torch.tensor(state, dtype=torch.float32).to(device)
                state = state.reshape(-1, *state.shape)
                q_action = q_net(state).max(1)[1].view(1, 1)  # max action
                action = int(q_action.item())
        else:
            # action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)  # random action
            action = random.randrange(n_actions)

        return action

    def evaluate(eval_num):

        eval_obs = env.reset()
        eval_done = False
        eval_r_list = []

        q_net.eval()
        with torch.no_grad():
            for _ in range(eval_num):
                eval_ep_r = 0
                while not eval_done:
                        eval_obs = np.array(eval_obs) / 255.
                        eval_obs = torch.tensor(eval_obs, dtype=torch.float32).to(device)
                        eval_obs = eval_obs.reshape(-1, *eval_obs.shape)
                        eval_act = q_net(eval_obs).max(1)[1].view(1, 1)
                        eval_obs, eval_r, eval_done, _ = env.step(eval_act)
                        eval_ep_r += eval_r
                eval_r_list.append(eval_ep_r)
        q_net.train()

        eval_r_list = torch.tensor(eval_r_list, dtype=torch.float32)
        mean_eval_r = eval_r_list.mean()
        max_eval_r = eval_r_list.max()
        min_eval_r = eval_r_list.min()

        return mean_eval_r, max_eval_r, min_eval_r

    def update():
        if len(memory) < train_config["training_start"]:
            return
        st = time.time()
        data_batch = memory.pick_sample()

        states, rewards, actions, next_states, dones = data_batch

        states = torch.tensor(states, dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        actions = torch.tensor(actions).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.BoolTensor(dones).to(device)
        # states = (states / 255.).type(dtype=torch.float32).to(device)
        # states = states.to(device)

        st = time.time()
        q_values = q_net(states).gather(1, actions.view(actions.size(0), -1)).view(-1)

        # next_states = (next_states / 255.).type(dtype=torch.float32).to(device)
        next_state_values = target_net(next_states).max(1)[0]
        next_state_values[dones] = 0.

        expected_q_values = gamma * next_state_values + rewards
        # TODO: expected_q_values shape

        loss = F.smooth_l1_loss(q_values, expected_q_values)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logger.store(loss=loss)

    while time_steps < train_config["num_iter"]:

        obs = env.reset()

        done = False
        ep_reward = 0

        start_steps = time_steps
        start = time.time()
        while not done:

            act = eps_exploration(obs, time_steps)

            next_obs, reward, done, _ = env.step(act)
            ep_reward += reward
            memory.store(obs, reward, act, next_obs, done)

            obs = next_obs

            update()

            time_steps += 1
            if time_steps % train_config["target_update_freq"] == 0:
                target_net.load_state_dict(q_net.state_dict())

            logger.store(TimeSteps=time_steps)
        fin = time.time() - start
        eps += 1
        sps = (time_steps - start_steps) / fin
        # print(h.heap())
        logger.store(EpRew=ep_reward, Ep=eps)
        print(f'ep: {eps}, timesteps: {time_steps}, reward: {ep_reward}, fps: {sps}')

        if eps % train_config["eval_freq"] == 0:
            mean_r, max_r, min_r = evaluate(train_config["eval_num"])
            logger.store(meanR=mean_r, maxR=max_r, minR=min_r)
            print(f'EVALUATE: mean_reward: {mean_r}, max_reward: {max_r}, min_reward: {min_r}')

        if eps % train_config["save_freq"] == 0:
            torch.save({
                "timesteps": time_steps,
                "eps": eps,
                "q_state": q_net.state_dict(),
                "target_state": target_net.state_dict(),
                "optim_state": optimizer.state_dict()
            },
                save_dir + f'/checkpoint-{eps}.pt')

        logger.write_log()
        if mean_r > train_config['target_reward']:
            break

    torch.save({
        "timesteps": time_steps,
        "eps": eps,
        "q_state": q_net.state_dict(),
        "target_state": target_net.state_dict(),
        "optim_state": optimizer.state_dict()
    }, save_dir + f'/checkpoint-final.pt')

    env.close()

