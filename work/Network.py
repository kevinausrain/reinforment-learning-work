import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from gymnasium import spaces
import torch.nn.functional as F
import gymnasium.envs.box2d.car_racing
from collections import deque
import moviepy.editor as mpy
import datetime
import util


class CNNNetWork(nn.Module):
    def __init__(self, config):
        super(CNNNetWork, self).__init__()
        # create network layers
        layers = nn.ModuleList()
        conv1 = (nn.Conv2d(config['conv1'][0],
                               config['conv1'][1],
                               config['conv1'][2],
                               config['conv1'][3],
                               config['conv1'][4])
                      )
        conv2 = (nn.Conv2d(config['conv2'][0],
                                config['conv2'][1],
                                config['conv2'][2],
                                config['conv2'][3],
                                config['conv2'][4])
                      )

        layers.append(conv1)
        layers.append(conv2)

        try:
            conv3 = (nn.Conv2d(config['conv3'][0],
                                config['conv3'][1],
                                config['conv3'][2],
                                config['conv3'][3],
                                config['conv3'][4])
                        )
            layers.append(conv3)
        except Exception as e:
            pass

        layers.append(nn.Flatten(start_dim=0))

        fc1 = nn.Linear(config['fc1'][0], config['fc1'][1])
        fc2 = nn.Linear(config['fc2'][0], config['fc2'][1])

        layers.append(fc1)
        layers.append(fc2)

        self.net = nn.Sequential(*layers)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config['lr'])

    def forward(self, x):
        return self.net(x)

class DNNNetWork(nn.Module):
    def __init__(self, config):
        super(DNNNetWork, self).__init__()
        # create network layer
        layers = nn.ModuleList()

        layers.append(nn.Linear(config['fc1'][0], config['fc1'][1]))
        layers.append(nn.Linear(config['fc2'][0], config['fc2'][1]))
        layers.append(nn.Linear(config['fc3'][0], config['fc3'][1]))

        self.net = nn.Sequential(*layers)

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
            outputs = torch.stack([self.net(inp) for inp in inputs])
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


    def copy_from(self, qnetwork):
        self.load_state_dict(qnetwork.state_dict())


class CNNPolicyNetwork(CNNNetWork):
    def __init__(self, config):
        super(CNNPolicyNetwork, self).__init__(config)

    def update(self, states, actions, returns, is_batch):
        self.optimizer.zero_grad()
        if is_batch:
            logits = torch.stack([self.net(state) for state in states])
        else:
            logits = self.net(states)

        dist = torch.distributions.Categorical(logits=logits)
        loss = torch.mean(-dist.log_prob(actions) * returns)

        before_params = [param.clone() for param in self.net.parameters()]
        loss.backward()
        self.optimizer.step()
        after_params = [param.clone() for param in self.net.parameters()]
        params_changed = [not torch.equal(before, after) for before, after in
                          zip(before_params, after_params)]

        #print("cnn policy network parameters updated: ", any(params_changed))
