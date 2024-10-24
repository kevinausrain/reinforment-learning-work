import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from gymnasium import spaces
import torch.nn.functional as F
from collections import deque
import moviepy.editor as mpy
import datetime
import util

def init_weight(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)

class CNNNetWork(nn.Module):
    def __init__(self, config):
        super(CNNNetWork, self).__init__()
        # create network layers
        layers = nn.ModuleList()
        conv1 = (nn.Conv2d(in_channels=config['conv1'][0],
                               out_channels=config['conv1'][1],
                               kernel_size=config['conv1'][2],
                               stride=config['conv1'][3])
                      )
        conv2 = (nn.Conv2d(in_channels=config['conv2'][0],
                               out_channels=config['conv2'][1],
                               kernel_size=config['conv2'][2],
                               stride=config['conv2'][3])
                      )

        layers.append(conv1)

        if not config['env_name'].startswith('CarRacing'):
            layers.append(nn.BatchNorm2d(config['conv1'][1]))

        layers.append(nn.ReLU())
        layers.append(conv2)

        if not config['env_name'].startswith('CarRacing'):
            layers.append(nn.BatchNorm2d(config['conv2'][1]))

        layers.append(nn.ReLU())

        try:
            conv3 = (nn.Conv2d(in_channels=config['conv3'][0],
                               out_channels=config['conv3'][1],
                               kernel_size=config['conv3'][2],
                               stride=config['conv3'][3])
                        )
            layers.append(conv3)
            layers.append(nn.BatchNorm2d(config['conv3'][1]))
            layers.append(nn.ReLU())
        except Exception as e:
            pass

        layers.append(nn.Flatten(start_dim=0))

        fc1 = nn.Linear(config['fc1'][0], config['fc1'][1])
        fc2 = nn.Linear(config['fc2'][0], config['fc2'][1])

        layers.append(fc1)

        if not config['env_name'].startswith('CarRacing'):
            layers.append(nn.LeakyReLU())

        layers.append(fc2)

        self.net = nn.Sequential(*layers)

        if config['initial_weight_required']:
            self.apply(init_weight)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config['lr'])

    def forward(self, x):
        return self.net(x)

class CNNValueNetwork(CNNNetWork):
    def __init__(self, config):
        super(CNNValueNetwork, self).__init__(config)

    def update(self, inputs, targets, is_batch):
        self.optimizer.zero_grad()
        if is_batch:
            outputs = torch.stack([self.net(inp.unsqueeze(0)) for inp in inputs])
        else:
            outputs = self.net(inputs)
        loss = self.criterion(outputs, targets)
        before_params = [param.clone() for param in self.net.parameters()]
        loss.backward()
        self.optimizer.step()
        after_params = [param.clone() for param in self.net.parameters()]
        params_changed = [not torch.equal(before, after) for before, after in
                           zip(before_params, after_params)]
        #print("cnn value network parameters updated: ", any(params_changed))

        return loss


    def copy_from(self, qnetwork):
        self.load_state_dict(qnetwork.state_dict())


class CNNPolicyNetwork(CNNNetWork):
    def __init__(self, config):
        super(CNNPolicyNetwork, self).__init__(config)

    def update(self, states, actions, returns, is_batch):
        self.optimizer.zero_grad()
        #logits = self.net(states)
        if is_batch:
            action_probs = torch.stack([self.net(state.unsqueeze(0)) for state in states])
        else:
            action_probs = self.net(states)
        dist = torch.distributions.Categorical(logits=action_probs)
        #log_prob = dist.log_prob(normal_sample)
        #action = torch.tanh(normal_sample)
        # 计算tanh_normal分布的对数概率密度
        #log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        print(action_probs)
        print(-torch.sum(action_probs * torch.log(action_probs + 1e-6), dim=-1))
        loss = torch.mean(-dist.log_prob(actions) * returns)
        #before_params = [param.clone() for param in self.net.parameters()]
        loss.backward()
        self.optimizer.step()
        #after_params = [param.clone() for param in self.net.parameters()]
        #params_changed = [not torch.equal(before, after) for before, after in
        #                  zip(before_params, after_params)]

        #print("cnn policy network parameters updated: ", any(params_changed))

        return loss
