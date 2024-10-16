import random
import time

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
from Network import CNNValueNetwork, CNNPolicyNetwork


class DQN():
    def __init__(self, env_name, env, config):
        self.env = env
        self.env_name = env_name
        self.qnet = CNNValueNetwork(config)
        self.target_qnet = CNNValueNetwork(config)
        self.target_qnet.copy_from(self.qnet)

        self.replay_buffer = deque(maxlen=config['replay_buffer'])
        self.minibatch_size = config['minibatch_size']
        self.target_update = config['target_update']
        self.target_update_idx = 0
        self.rng = np.random.default_rng()
        self.warm_up_steps = config['warm_up_steps']

        self.is_action_discrete = config['is_action_discrete']
        self.is_observation_space_image = config['is_observation_space_image']
        self.discount = config['discount']
        self.epsilon = config['greedy']
        self.epsilon_min = config['greedy_min']
        self.decay_speed = config['decay_speed']

        if self.is_observation_space_image:
            self.stack_num = config['stack_num']
            self.frames = deque(maxlen=self.stack_num)

        if not self.is_action_discrete:
            self.max_action = config['max_action']

        if self.is_action_discrete:
            self.preferable_action_probs = config['preferable_action_probs']
            self.actions = config['actions']

    def behaviour(self, state, steps, stochastic=True):
        state = torch.tensor(state, dtype=torch.float)

        if stochastic and (random.random() <= self.epsilon or steps <= self.warm_up_steps):
            return np.random.choice(self.actions, size = 1, p=self.preferable_action_probs)[0]
        else:
            q = self.qnet.forward(state).detach()
            if self.env_name.startswith('Pong'):
                j = self.rng.permutation(3)
            else:
                j = self.rng.permutation(self.env.action_space.n)
            return j[q[j].argmax().item()]

    def policy(self, state):
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float)

        return self.qnet(state).detach().argmax().item()

    def update(self):
        if len(self.replay_buffer) >= self.minibatch_size:
            batch = self.rng.choice(len(self.replay_buffer), size=self.minibatch_size, replace=False)
            inputs = torch.zeros((self.minibatch_size, self.stack_num, 84, 84))
            if self.env_name.startswith('Pong'):
                targets = torch.zeros((self.minibatch_size, 3))
            else:
                targets = torch.zeros((self.minibatch_size, self.env.action_space.n))

            for n, index in enumerate(batch):
                state, action, reward, next_state, terminated = self.replay_buffer[index]
                inputs[n, :] = state
                targets[n, :] = self.target_qnet(state).detach()
                if terminated:
                    targets[n, action] = reward
                else:
                    targets[n, action] = reward + self.discount * self.target_qnet(next_state).detach().max()

            self.qnet.update(inputs, targets, True)

        self.target_update_idx += 1
        if self.target_update_idx % self.target_update == 0:
            self.target_qnet.copy_from(self.qnet)

    def train(self, max_episodes, stop_criterion, criterion_episodes):
        now = time.time()
        txt = open(str(datetime.datetime.now()) + '_' + self.env_name + '_greedy_' + str(self.epsilon).replace(".", "") +'_dqn.txt', 'a')
        rewards = []
        num_steps = 0
        for episode in range(max_episodes):
            state, _ = self.env.reset()

            #if self.env_name is 'CarRacing':
            #    state = util.car_v2_image_preprocess(state)
            #if self.env_name is 'SpaceInvader' or self.env_name is 'Pong':
            #    state = util.atari_v2_image_preprocess(state)

            state = torch.tensor(state, dtype=torch.float)

            for i in range(self.stack_num):
                self.frames.append(state)
            state = torch.tensor(np.stack(self.frames))

            terminated = False
            truncated = False
            rewards.append(0)
            zero_rewards = list()
            while not (terminated or truncated):
                action = self.behaviour(state, num_steps, True)
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                #if self.env_name is 'CarRacing':
                #    next_state = util.car_v2_image_preprocess(next_state)
                #else:
                #    next_state = util.atari_v2_image_preprocess(next_state)

                next_state = torch.tensor(next_state, dtype=torch.float)

                self.frames.append(next_state)
                next_state = torch.tensor(np.stack(self.frames))
                next_state = next_state

                if self.env_name.startswith('PongXX'):
                    if reward != 0.0:
                        for zero_reward in zero_rewards:
                            self.replay_buffer.append(
                                (zero_reward['state'],
                                 zero_reward['action'],
                                 reward / len(zero_reward),
                                 zero_reward['next_state'],
                                 zero_reward['terminated']))
                        self.replay_buffer.append(
                            (state, action, reward, next_state, terminated))
                        zero_rewards.clear()
                    else:
                        zero_rewards.append({'state': state, 'action': action,
                                             'next_state': next_state,
                                             'terminated': terminated})
                else:
                    self.replay_buffer.append((state, action, reward, next_state, terminated))


                if num_steps >= self.warm_up_steps:
                    self.update()

                if num_steps % 10000 == 0:
                    self.epsilon = max(self.epsilon - self.decay_speed, self.epsilon_min)

                state = next_state
                rewards[-1] += reward
                num_steps += 1

            txt.write("Episode {} done: steps = {}, rewards = {}, consume time = {}s\n".format(
                episode + 1, num_steps, rewards[episode], time.time() - now))
            print(f'\rEpisode {episode + 1} done: steps = {num_steps}, rewards = {rewards[episode]}     ', end='')

            if episode >= criterion_episodes - 1 and episode % 50 == 0:
                if stop_criterion(rewards[-criterion_episodes:]):
                    self.save(str(datetime.datetime.now()) + '_' + self.env_name + '_greedy_' + str(self.epsilon).replace(".", "") +'_dqn')
                    print(f'\nStopping criterion satisfied after {episode} episodes')
                    break
                else:
                    self.save(str(datetime.datetime.now()) + '_' + self.env_name + '_greedy_' + str(self.epsilon).replace(".", "") +'_dqn_fail')

        plt.figure(dpi=100)
        plt.plot(range(1, len(rewards) + 1), rewards, label=f'Rewards')

        plt.xlabel('Episodes')
        plt.ylabel('Rewards per episode')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()

    def save(self, path):
        # save network weights to a file
        torch.save(self.qnet.state_dict(), path)

    def load(self, path):
        # load network weights from a file
        self.qnet.load_state_dict(torch.load(path))
        self.target_qnet.copy_from(self.qnet)


class REINFORCE():
    def __init__(self, env_name, env, config):
        self.env = env
        self.env_name = env_name
        self.preferable_action_probs = config['preferable_action_probs']
        self.is_action_discrete = config['is_action_discrete']
        self.is_observation_space_image = config['is_observation_space_image']
        self.discount = config['discount']
        self.epsilon = config['greedy']
        self.epsilon_min = config['greedy_min']
        self.decay_speed = config['decay_speed']
        self.warm_up_steps = config['warm_up_steps']

        self.policy_net = CNNPolicyNetwork(config)

        if self.is_observation_space_image:
            self.stack_num = config['stack_num']
            self.frames = deque(maxlen=self.stack_num)

        if self.is_action_discrete:
            self.preferable_action_probs = config['preferable_action_probs']
            self.actions = config['actions']

    def policy(self, state, steps, stochastic=True):
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float)

        if stochastic and (random.random() <= self.epsilon or steps <= self.warm_up_steps):
            return np.random.choice(self.actions, size = 1, p=self.preferable_action_probs)[0]
        else:
            action_probs = self.policy_net(state).detach()
            dist = torch.distributions.Categorical(logits=action_probs)
            return dist.probs.argmax().item()

    def update(self, trajectory):
        states, actions, rewards = list(zip(*trajectory))
        returns = torch.zeros((len(trajectory),))
        returns[-1] = rewards[-1]

        for t in reversed(range(len(trajectory) - 1)):
            returns[t] = rewards[t] + self.discount * returns[t + 1]

        states = torch.tensor([state.cpu().detach().numpy() for state in states])
        actions = torch.tensor([action for action in actions], dtype=torch.float)

        self.policy_net.update(states, actions, returns, True)

    def train(self, max_episodes, stop_criterion, criterion_episodes):
        txt = open(str(datetime.datetime.now()) + '_' + self.env_name + '_greedy_' + str(self.epsilon).replace(".", "") +'_rein.txt', 'a')
        now = time.time()
        num_steps = 0
        episode_rewards = []
        for episode in range(max_episodes):
            state, _ = self.env.reset()

            #if self.env_name is 'CarRacing':
            #    state = util.car_v2_image_preprocess(state)
            #if self.env_name is 'SpaceInvader' or self.env_name is 'Pong':
            #    state = util.atari_v2_image_preprocess(state)

            state = torch.tensor(state, dtype=torch.float)

            for i in range(self.stack_num):
                self.frames.append(state)
            state = torch.tensor(np.stack(self.frames))

            terminated = False
            truncated = False
            episode_rewards.append(0)
            trajectory = []

            zero_rewards = list()

            while not (terminated or truncated):
                action = self.policy(state, num_steps)
                if self.env_name.startswith('PongXX'):
                    if action != 0:
                        action = action + 1
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                episode_rewards[-1] += reward

                if self.env_name.startswith('Pong'):
                    if reward != 0.0:
                        for zero_reward in zero_rewards:
                            trajectory.append(
                                (zero_reward['state'],
                                 zero_reward['action'],
                                 reward))
                        trajectory.append((state, action, reward))
                        zero_rewards.clear()
                    else:
                        zero_rewards.append({'state': state, 'action': action})
                else:
                    trajectory.append((state, action, reward))

                #if self.env_name is 'CarRacing':
                #    next_state = util.car_v2_image_preprocess(next_state)
                #if self.env_name is 'SpaceInvader' or self.env_name is 'Pong':
                #    next_state = util.atari_v2_image_preprocess(next_state)

                next_state = torch.tensor(next_state, dtype=torch.float)
                self.frames.append(next_state)
                next_state = torch.tensor(np.stack(self.frames))

                state = next_state
                num_steps += 1

                if num_steps % 10000 == 0:
                    self.epsilon = max(self.epsilon - self.decay_speed, self.epsilon_min)

            self.update(trajectory)

            txt.write("Episode {} done: steps = {}, rewards = {}, consume time = {}s\n".format(
                episode + 1, num_steps, episode_rewards[episode], time.time() - now))
            print(f'\rEpisode {episode + 1} done: steps = {num_steps}, '
                  f'rewards = {episode_rewards[episode]}     ', end='')

            if episode >= criterion_episodes - 1 and episode % 100 == 0:
                if stop_criterion(episode_rewards[-criterion_episodes:]):
                    self.save(str(datetime.datetime.now()) + '_' + self.env_name + '_greedy_' + str(self.epsilon).replace(".", "") +'_rein')
                    print(f'\nStopping criterion satisfied after {episode} episodes')
                    break
                else:
                    self.save(str(datetime.datetime.now()) + '_' + self.env_name + '_greedy_' + str(self.epsilon).replace(".", "") +'_rein_fail')

        # plot rewards received during training
        plt.figure(dpi=100)
        plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, label=f'Rewards')

        plt.xlabel('Episodes')
        plt.ylabel('Rewards per episode')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()

    def save(self, path):
        # save network weights to a file
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        # load network weights from a file
        self.policy_net.load_state_dict(torch.load(path))

class ActorCritic():
    def __init__(self, env_name, env, policy_config, value_config):
        self.env = env
        self.env_name = env_name
        self.policy_net = CNNPolicyNetwork(policy_config)
        self.value_net = CNNValueNetwork(value_config)

        self.is_action_discrete = policy_config['is_action_discrete']
        self.is_observation_space_image = policy_config['is_observation_space_image']
        self.discount = policy_config['discount']
        self.epsilon = policy_config['greedy']
        self.epsilon_min = policy_config['greedy_min']
        self.decay_speed = policy_config['decay_speed']
        self.warm_up_steps = policy_config['warm_up_steps']

        if self.is_observation_space_image:
            self.permute = policy_config['permute']
            self.stack_num = policy_config['stack_num']
            self.frames = deque(maxlen=self.stack_num)

        if self.is_action_discrete:
            self.preferable_action_probs = policy_config['preferable_action_probs']
            self.actions = policy_config['actions']

    def policy(self, state, steps, stochastic=True):
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float)

        action_probs = self.policy_net(state).detach()
        dist = torch.distributions.Categorical(logits=action_probs)

        if stochastic and (random.random() <= self.epsilon or steps <= self.warm_up_steps):
            return np.random.choice(self.actions, size=1, p=self.preferable_action_probs)[0]
        else:
            return dist.probs.argmax().item()

    def update(self, state, action, reward, next_state, terminated):
        if terminated:
            target = reward
        else:
            next_state = torch.tensor(next_state)
            target = reward + self.discount * self.value_net(next_state).detach()

        state = torch.tensor(state)
        action = torch.tensor(action)

        target = torch.stack([target])
        delta = target - self.value_net(state).detach()

        self.policy_net.update(state, action, delta, False)
        self.value_net.update(state, target, False)

    def train(self, max_episodes, stop_criterion, criterion_episodes):
        txt = open(str(datetime.datetime.now()) + '_' + self.env_name + '_greedy_' + str(self.epsilon).replace(".", "") +'_ac.txt', 'a')
        now = time.time()
        num_steps = 0
        episode_rewards = []
        for episode in range(max_episodes):
            state, _ = self.env.reset()

            #if self.env_name is 'CarRacing':
            #    state = util.car_v2_image_preprocess(state)
            #if self.env_name is 'SpaceInvader' or self.env_name is 'Pong':
            #    state = util.atari_v2_image_preprocess(state)

            state = torch.tensor(state, dtype=torch.float)
            for i in range(self.stack_num):
                self.frames.append(state)
            state = torch.tensor(np.stack(self.frames))

            terminated = False
            truncated = False
            episode_rewards.append(0)

            while not (terminated or truncated):
                action = self.policy(state, num_steps)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                episode_rewards[-1] += reward

                #if self.env_name is 'CarRacing':
                #    next_state = util.car_v2_image_preprocess(next_state)
                #if self.env_name is 'SpaceInvader' or self.env_name is 'Pong':
                #    next_state = util.atari_v2_image_preprocess(next_state)

                next_state = torch.tensor(next_state, dtype=torch.float)
                self.frames.append(next_state)
                next_state = torch.tensor(np.stack(self.frames))

                self.update(state, action, reward, next_state, terminated)
                num_steps += 1

                if num_steps % 10000 == 0:
                    self.epsilon = max(self.epsilon - self.decay_speed, self.epsilon_min)

            txt.write("Episode {} done: steps = {}, rewards = {}. consume time = {}s\n".format(
                episode + 1, num_steps, episode_rewards[episode], time.time() - now))
            print(f'\rEpisode {episode + 1} done: steps = {num_steps}, '
                  f'rewards = {episode_rewards[episode]}     ', end='')

            if episode >= criterion_episodes - 1 and episode % 100 == 0:
                if stop_criterion(episode_rewards[-criterion_episodes:]):
                    self.save(str(datetime.datetime.now()) + '_' + self.env_name + '_greedy_' + str(self.epsilon).replace(".",
                                                                                                                    "") + '_ac')
                    print(f'\nStopping criterion satisfied after {episode} episodes')
                    break
                else:
                    self.save(str(datetime.datetime.now()) + '_' + self.env_name + '_greedy_' + str(self.epsilon).replace(".",
                                                                                                                "") + '_ac_fail')

        plt.figure(dpi=100)
        plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, label=f'Rewards')

        plt.xlabel('Episodes')
        plt.ylabel('Rewards per episode')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()

    def save(self, path):
        # save network weights to a file
        torch.save({'policy': self.policy_net.state_dict(),
                    'value': self.value_net.state_dict()}, path)

    def load(self, path):
        # load network weights from a file
        networks = torch.load(path)
        self.policy_net.load_state_dict(networks['policy'])
        self.value_net.load_state_dict(networks['value'])

