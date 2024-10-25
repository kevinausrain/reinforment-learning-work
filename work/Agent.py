import random
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from gymnasium import spaces
import torch.nn.functional as F
from collections import deque
import moviepy.editor as mpy
import datetime
from Network import *
import util

save_record_path = '/Users/kevin/PycharmProjects/reinforment-learning-work/work/record/'
save_model_path = '/Users/kevin/PycharmProjects/reinforment-learning-work/work/saved_model/'


class DQN():
    def __init__(self, env_name, env, model_name, config):
        self.env = env
        self.env_name = env_name
        self.network_type = config['network_type']

        if self.network_type is 'cnn':
            self.qnet = CNNValueNetwork(config)
            self.target_qnet = CNNValueNetwork(config)
        else:
            self.qnet = DNNValueNetwork(config)
            self.target_qnet = DNNValueNetwork(config)

        if model_name is not None:
            self.load(model_name)
            print('load success')
        else:
            self.target_qnet.copy_from(self.qnet)

        self.max_step = config['max_step']
        self.replay_buffer = deque(maxlen=config['replay_buffer'])
        self.buffer_size = config['replay_buffer']
        self.minibatch_size = config['minibatch_size']
        self.batch_size = config['batch_size']
        self.target_update = config['target_update']
        self.target_update_idx = 0
        self.rng = np.random.default_rng()
        self.warm_up_steps = config['warm_up_steps']
        self.learning_rate = config['lr']
        self.initial_weight_required = config['initial_weight_required']

        self.is_action_discrete = config['is_action_discrete']
        self.is_observation_space_image = config['is_observation_space_image']
        self.discount = config['discount']
        self.epsilon = config['greedy']
        self.epsilon_min = config['greedy_min']
        self.decay_speed = config['decay_speed']

        self.stack_frame_num = config['stack_frame_num']
        self.skip_frame_num = config['skip_frame_num']
        self.use_skip_frame = config['use_skip_frame']
        self.frames = deque(maxlen=self.skip_frame_num)
        self.preferable_action_probs = config['preferable_action_probs']
        self.actions = config['actions']

    def behaviour(self, state, steps, stochastic=True):
        if self.network_type is 'cnn':
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        else:
            state = torch.flatten(torch.tensor(state, dtype=torch.float))

        if stochastic and (random.random() <= self.epsilon or steps <= self.warm_up_steps):
            return np.random.choice(self.actions, size = 1, p=self.preferable_action_probs)[0]
        else:
            state = torch.tensor(state, dtype=torch.float)
            q = self.qnet.forward(state).detach()
            j = self.rng.permutation(self.env.action_space.n)
            return j[q[j].argmax().item()]

    def policy(self, state):
        if not torch.is_tensor(state):
            if self.network_type is 'cnn':
                state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            else:
                state = torch.flatten(torch.tensor(state, dtype=torch.float))

        return self.qnet(state).detach().argmax().item()

    def update(self):
        loss = 0
        if len(self.replay_buffer) >= self.minibatch_size:
            batch = self.rng.choice(len(self.replay_buffer), size=self.batch_size, replace=False)
            if self.network_type is 'cnn':
                if self.env_name.startswith('CarRacing'):
                    inputs = torch.zeros((self.batch_size, self.stack_frame_num, 84, 84))
                else:
                    inputs = torch.zeros((self.batch_size, self.stack_frame_num, 64, 80))
            else:
                if self.env_name.startswith('CarRacing'):
                    inputs = torch.zeros((self.batch_size, self.stack_frame_num * 84 * 84))
                else:
                    inputs = torch.zeros((self.batch_size, self.stack_frame_num * 64 * 80))

            targets = torch.zeros((self.batch_size, self.env.action_space.n))
            for n, index in enumerate(batch):
                state, action, reward, next_state, terminated = self.replay_buffer[index]
                state = torch.tensor(state, dtype=torch.float)
                if self.network_type is 'cnn':
                    inputs[n, :] = state
                else:
                    inputs[n, :] = torch.flatten(state)
                if self.network_type is 'cnn':
                    targets[n, :] = self.target_qnet(state.unsqueeze(0)).detach()
                else:
                    targets[n, :] = self.target_qnet(torch.flatten(state)).detach()
                if terminated:
                    targets[n, action] = reward
                else:
                    if self.network_type is 'cnn':
                        next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)
                    else:
                        next_state = torch.flatten(torch.tensor(next_state, dtype=torch.float))
                    targets[n, action] = reward + self.discount * self.target_qnet(next_state).detach().max()

            if self.network_type is 'cnn':
                loss = self.qnet.update(inputs, targets, True)
            else:
                loss = self.qnet.update(inputs, targets)

        self.target_update_idx += 1
        if self.target_update_idx % self.target_update == 0:
            self.target_qnet.copy_from(self.qnet)

        return loss

    def train(self, max_episodes, stop_criterion, criterion_episodes):
        now = time.time()
        txt = open(save_record_path + str(datetime.datetime.now()) + '_' + self.env_name +
                   '_greedy_' + str(self.epsilon).replace(".", "") +
                   '_greedy_min_' + str(self.epsilon_min).replace(".", "") +
                   '_decay_speed_' + str(self.decay_speed).replace(".", "") +
                   '_buffer_size_' + str(self.buffer_size) +
                   '_target_update_freq_' + str(self.target_update) +
                   '_use_skip_frame_' + str(self.use_skip_frame) +
                   '_lr_' + str(self.learning_rate) +
                   '_init_w_' + str(self.initial_weight_required) + '_dqn.txt', 'a')
        rewards = []
        num_steps = 0
        for episode in range(max_episodes):
            state, _ = self.env.reset()
            if self.env_name is 'CarRacing':
                state = util.box2d_preProcess(state)
            else:
                state = util.atari_preProcess(state)

            for i in range(self.stack_frame_num):
                self.frames.append(state)
            state = np.stack(self.frames)

            terminated = False
            truncated = False
            rewards.append(0)
            loss = 0
            actions = []
            skipped = 0
            last_action = None
            while not (terminated or truncated):
                if self.use_skip_frame:
                    if skipped == 0:
                        action = self.behaviour(state, num_steps, True)
                        last_action = action
                    else:
                        action = last_action
                else:
                    action = self.behaviour(state, num_steps, True)

                skipped = (skipped + 1) % self.skip_frame_num
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                if self.env_name is 'CarRacing':
                    next_state = util.box2d_preProcess(next_state)
                else:
                    next_state = util.atari_preProcess(next_state)

                self.frames.append(next_state)
                next_state = np.stack(self.frames)

                self.replay_buffer.append((state, action, reward, next_state, terminated))
                actions.append(action)

                loss += self.update()
                self.epsilon = max(self.epsilon - self.decay_speed, self.epsilon_min)

                state = next_state
                rewards[-1] += reward
                num_steps += 1

                if num_steps >= self.max_step:
                    break

            txt.write("Episode {} done, steps = {}, actions = {}, loss = {}, rewards = {}, consume time = {}s\n".format(
                episode + 1, num_steps, util.display_action_distribution(actions, len(self.actions)),
                loss, rewards[episode], time.time() - now))
            print(f'Episode {episode + 1} done, steps = {num_steps}, actions = {util.display_action_distribution(actions, len(self.actions))}, '
                  f'loss = {loss}, epsilon = {self.epsilon}, rewards = {rewards[episode]}')

            if episode >= criterion_episodes - 1 and episode % 50 == 0:
                if stop_criterion(rewards[-criterion_episodes:]):
                    self.save(save_model_path + str(datetime.datetime.now()) + '_' + self.env_name + '_greedy_' + str(self.epsilon).replace(".", "") +'_dqn')
                    print(f'\nStopping criterion satisfied after {episode} episodes')
                    break
                else:
                    self.save(save_model_path + str(datetime.datetime.now()) + '_' + self.env_name + '_greedy_' + str(self.epsilon).replace(".", "") +'_dqn_fail')

        plt.figure(dpi=100)
        plt.plot(range(1, len(rewards) + 1), rewards, label=f'Rewards')

        plt.xlabel('Episodes')
        plt.ylabel('Rewards per episode')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()

    def save(self, path):
        torch.save(self.qnet.state_dict(), path)

    def load(self, path):
        self.qnet.load_state_dict(torch.load(path))
        self.target_qnet.copy_from(self.qnet)


class REINFORCE():
    def __init__(self, env_name, env, model_name, config):
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
        self.network_type = config['network_type']
        self.learning_rate = config['lr']
        self.initial_weight_required = config['initial_weight_required']

        if self.network_type is 'cnn':
            self.policy_net = CNNPolicyNetwork(config)
        else:
            self.policy_net = DNNPolicyNetwork(config)

        if model_name is not None:
            self.load(model_name)

        self.stack_frame_num = config['stack_frame_num']
        self.frames = deque(maxlen=self.stack_frame_num)
        self.skip_frame_num = config['skip_frame_num']
        self.use_skip_frame = config['use_skip_frame']

        self.preferable_action_probs = config['preferable_action_probs']
        self.actions = config['actions']


    def policy(self, state, stochastic=True):
        if not torch.is_tensor(state):
            if self.network_type is 'cnn':
                state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            else:
                state = torch.flatten(torch.tensor(state, dtype=torch.float))

        action_probs = self.policy_net(state).detach()
        dist = torch.distributions.Categorical(logits=action_probs)

        if stochastic:
            return dist.sample().item()
        else:
            return dist.probs.argmax().item()

    def update(self, trajectory):
        states, actions, rewards = list(zip(*trajectory))
        returns = torch.zeros((len(trajectory),))
        returns[-1] = rewards[-1]

        for t in reversed(range(len(trajectory) - 1)):
            returns[t] = rewards[t] + self.discount * returns[t + 1]

        if self.network_type is 'cnn':
            states = torch.tensor(states, dtype=torch.float)
        else:
            states = torch.stack([torch.flatten(torch.tensor(state, dtype=torch.float))
                                  for state in states])

        actions = torch.tensor(actions)

        if self.network_type is 'cnn':
            return self.policy_net.update(states, actions, returns, True)
        else:
            return self.policy_net.update(states, actions, returns)

    def train(self, max_episodes, stop_criterion, criterion_episodes):
        txt = open(save_record_path + str(datetime.datetime.now()) + '_' + self.env_name +
                   '_lr_' + str(self.learning_rate) +
                   '_use_skip_frame_' + str(self.use_skip_frame) +
                   '_init_w_' + str(self.initial_weight_required) + '_rein.txt', 'a')
        now = time.time()
        num_steps = 0
        episode_rewards = []
        for episode in range(max_episodes):
            state, _ = self.env.reset()
            if self.env_name is 'CarRacing':
                state = util.box2d_preProcess(state)
            else:
                state = util.atari_preProcess(state)

            for i in range(self.stack_frame_num):
                self.frames.append(state)
            state = np.stack(self.frames)

            terminated = False
            truncated = False
            episode_rewards.append(0)
            trajectory = []
            loss = 0
            actions = []
            skipped = 0
            last_action = None

            while not (terminated or truncated):
                if self.use_skip_frame:
                    if skipped == 0:
                        action = self.policy(state)
                        last_action = action
                    else:
                        action = last_action
                else:
                    action = self.policy(state)

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                episode_rewards[-1] += reward

                if not self.env_name.startswith('CarRacing'):
                    trajectory.append((state, action, reward))
                    actions.append(action)
                else:
                    trajectory.append((state, action, reward))
                    actions.append(action)

                if self.env_name is 'CarRacing':
                    next_state = util.box2d_preProcess(next_state)
                else:
                    next_state = util.atari_preProcess(next_state)

                self.frames.append(next_state)
                next_state = np.stack(self.frames)

                state = next_state
                num_steps += 1

                if num_steps % 1000 == 0:
                    self.epsilon = max(self.epsilon - self.decay_speed, self.epsilon_min)

            if len(trajectory) > 0:
                loss = self.update(trajectory)

            txt.write("Episode {} done, steps = {}, actions = {}, loss = {}, rewards = {}, consume time = {}s\n".format(
                episode + 1, num_steps, util.display_action_distribution(actions, len(self.actions)),
                loss, episode_rewards[episode], time.time() - now))
            print(f'Episode {episode + 1} done, steps = {num_steps}, actions = {util.display_action_distribution(actions, len(self.actions))}, '
                  f'loss = {loss}, epsilon = {self.epsilon}, '
                  f'rewards = {episode_rewards[episode]}')

            if episode >= criterion_episodes - 1 and episode % 100 == 0:
                if stop_criterion(episode_rewards[-criterion_episodes:]):
                    self.save(save_model_path + str(datetime.datetime.now()) + '_' + self.env_name + '_lr_' + str(self.learning_rate).replace(".", "") +'_rein')
                    print(f'\nStopping criterion satisfied after {episode} episodes')
                    break
                else:
                    self.save(save_model_path + str(datetime.datetime.now()) + '_' + self.env_name + '_lr_' + str(self.learning_rate).replace(".", "") +'_rein_fail')

        # plot rewards received during training
        plt.figure(dpi=100)
        plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, label=f'Rewards')

        plt.xlabel('Episodes')
        plt.ylabel('Rewards per episode')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))

class ActorCritic():
    def __init__(self, env_name, env, model_name, policy_config, value_config):
        self.env = env
        self.env_name = env_name
        self.network_type = policy_config['network_type']

        if self.network_type is 'cnn':
            self.policy_net = CNNPolicyNetwork(policy_config)
            self.value_net = CNNValueNetwork(value_config)
        else:
            self.policy_net = DNNPolicyNetwork(policy_config)
            self.value_net = DNNValueNetwork(value_config)

        if model_name is not None:
            self.load(model_name)
            print('load success')

        self.is_action_discrete = policy_config['is_action_discrete']
        self.is_observation_space_image = policy_config['is_observation_space_image']
        self.discount = policy_config['discount']
        self.epsilon = policy_config['greedy']
        self.epsilon_min = policy_config['greedy_min']
        self.decay_speed = policy_config['decay_speed']
        self.warm_up_steps = policy_config['warm_up_steps']
        self.learning_rate = policy_config['lr']
        self.initial_weight_required = policy_config['initial_weight_required']

        self.stack_frame_num = policy_config['stack_frame_num']
        self.frames = deque(maxlen=self.stack_frame_num)
        self.skip_frame_num = policy_config['skip_frame_num']
        self.use_skip_frame = policy_config['use_skip_frame']

        self.preferable_action_probs = policy_config['preferable_action_probs']
        self.actions = policy_config['actions']

    def policy(self, state, stochastic=True):
        if not torch.is_tensor(state):
            if self.network_type is 'cnn':
                state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            else:
                state = torch.flatten(torch.tensor(state, dtype=torch.float))

        action_probs = self.policy_net(state).detach()
        dist = torch.distributions.Categorical(logits=action_probs)

        if stochastic:
            return dist.sample().item()
        else:
            return dist.probs.argmax().item()

    def update(self, state, action, reward, next_state, terminated):
        if not torch.is_tensor(state):
            if self.network_type is 'cnn':
                state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            else:
                state = torch.flatten(torch.tensor(state, dtype=torch.float))

        if not torch.is_tensor(next_state):
            if self.network_type is 'cnn':
                next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)
            else:
                next_state = torch.flatten(torch.tensor(next_state, dtype=torch.float))

        if terminated:
            target = reward
        else:
            target = reward + self.discount * self.value_net(next_state).detach()

        target = torch.tensor([target], dtype=torch.float)
        delta = target - self.value_net(state).detach()

        action = torch.tensor(action)

        if self.network_type is 'cnn':
            policy_loss = self.policy_net.update(state, action, delta, False)
            value_loss = self.value_net.update(state, target, False)
        else:
            policy_loss = self.policy_net.update(state, action, delta)
            value_loss = self.value_net.update(state, target)

        return policy_loss, value_loss


    def train(self, max_episodes, stop_criterion, criterion_episodes):
        txt = open(save_record_path + str(datetime.datetime.now()) + '_' + self.env_name +
                   '_lr_' + str(self.learning_rate) +
                   '_use_skip_frame_' + str(self.use_skip_frame) +
                   '_init_w_' + str(self.initial_weight_required) + '_ac.txt', 'a')
        now = time.time()
        num_steps = 0
        episode_rewards = []

        for episode in range(max_episodes):
            state, _ = self.env.reset()

            if self.env_name is 'CarRacing':
                state = util.box2d_preProcess(state)
            else:
                state = util.atari_preProcess(state)

            for i in range(self.stack_frame_num):
                self.frames.append(state)
            state = np.stack(self.frames)

            terminated = False
            truncated = False
            episode_rewards.append(0)

            policy_loss = 0
            value_loss = 0
            actions = []
            skipped = 0
            last_action = None

            while not (terminated or truncated):
                if self.use_skip_frame:
                    if skipped == 0:
                        action = self.policy(state)
                        last_action = action
                    else:
                        action = last_action
                else:
                    action = self.policy(state)

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                episode_rewards[-1] += reward

                if self.env_name is 'CarRacing':
                    next_state = util.box2d_preProcess(next_state)
                else:
                    next_state = util.atari_preProcess(next_state)

                self.frames.append(next_state)
                next_state = np.stack(self.frames)

                p_loss, v_loss = self.update(state, action, reward, next_state, terminated)

                if p_loss != 0.0:
                    policy_loss = p_loss

                if v_loss != 0.0:
                    value_loss = v_loss

                actions.append(action)

                num_steps += 1
                self.epsilon = max(self.epsilon - self.decay_speed, self.epsilon_min)

            txt.write("Episode {} done, steps = {}, actions = {}, p_loss = {}, v_loss = {}, rewards = {}, consume time = {}s\n".format(
                episode + 1, num_steps, util.display_action_distribution(actions, len(self.actions)), policy_loss, value_loss, episode_rewards[episode], time.time() - now))
            print(f'Episode {episode + 1} done, steps = {num_steps}, '
                  f'actions = {util.display_action_distribution(actions, len(self.actions))}, '
                  f'p_loss = {policy_loss}, v_loss = {value_loss}, '
                  f'epsilon = {self.epsilon}, rewards = {episode_rewards[episode]}')

            if episode >= criterion_episodes - 1 and episode % 100 == 0:
                if stop_criterion(episode_rewards[-criterion_episodes:]):
                    self.save(save_model_path + str(datetime.datetime.now()) + '_' + self.env_name + '_lr_' + str(self.learning_rate).replace(".",
                                                                                                                    "") + '_ac')
                    print(f'\nStopping criterion satisfied after {episode} episodes')
                    break
                else:
                    self.save(save_model_path + str(datetime.datetime.now()) + '_' + self.env_name + '_lr_' + str(self.learning_rate).replace(".",
                                                                                                                "") + '_ac_fail')

        plt.figure(dpi=100)
        plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, label=f'Rewards')

        plt.xlabel('Episodes')
        plt.ylabel('Rewards per episode')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()

    def save(self, path):
        torch.save({'policy': self.policy_net.state_dict(),
                    'value': self.value_net.state_dict()}, path)

    def load(self, path):
        networks = torch.load(path)
        self.policy_net.load_state_dict(networks['policy'])
        self.value_net.load_state_dict(networks['value'])

class SAC():
    def __init__(self, env_name, env, model_name, policy_config, value_config):
        self.env = env
        self.env_name = env_name
        self.network_type = policy_config['network_type']

        if self.network_type is 'cnn':
            self.actor_net = SACPolicyNetwork(policy_config)
            self.critic_net_1 = SACvalueNetwork(value_config)
            self.critic_net_2 = SACvalueNetwork(value_config)
            self.target_critic_net_1 = SACvalueNetwork(value_config)
            self.target_critic_net_2 = SACvalueNetwork(value_config)
        else:
            self.actor_net = SACDNNPolicyNetwork(policy_config)
            self.critic_net_1 = SACDNNValueNetwork(value_config)
            self.critic_net_2 = SACDNNValueNetwork(value_config)
            self.target_critic_net_1 = SACDNNValueNetwork(value_config)
            self.target_critic_net_2 = SACDNNValueNetwork(value_config)

        self.actor_net_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=policy_config['lr'])
        self.critic_net_1_optimizer = torch.optim.Adam(self.critic_net_1.parameters(), lr=value_config['lr'])
        self.critic_net_2_optimizer = torch.optim.Adam(self.critic_net_2.parameters(), lr=value_config['lr'])

        if model_name is not None:
            self.load(model_name)
            print('load success')

        self.is_action_discrete = policy_config['is_action_discrete']
        self.is_observation_space_image = policy_config['is_observation_space_image']
        self.discount = policy_config['discount']
        self.tau = policy_config['tau']
        self.epsilon = policy_config['greedy']
        self.epsilon_min = policy_config['greedy_min']
        self.decay_speed = policy_config['decay_speed']
        self.warm_up_steps = policy_config['warm_up_steps']
        self.learning_rate = policy_config['lr']
        self.initial_weight_required = policy_config['initial_weight_required']

        self.replay_buffer = deque(maxlen=policy_config['replay_buffer'])
        self.buffer_size = policy_config['replay_buffer']
        self.minibatch_size = policy_config['minibatch_size']
        self.batch_size = policy_config['batch_size']

        self.stack_frame_num = policy_config['stack_frame_num']
        self.frames = deque(maxlen=self.stack_frame_num)
        self.skip_frame_num = policy_config['skip_frame_num']
        self.use_skip_frame = policy_config['use_skip_frame']

        self.preferable_action_probs = policy_config['preferable_action_probs']
        self.actions = policy_config['actions']
        self.target_entropy = policy_config['target_entropy']

        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=policy_config['alpha_lr'])

    def sac_policy(self, state):
        if not torch.is_tensor(state):
            if self.network_type is 'cnn':
                state = torch.tensor(state, dtype=torch.float)
            else:
                state = torch.flatten(torch.tensor(state, dtype=torch.float))
        probs = self.actor_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def calculate_target_value(self, rewards, next_states, terminated):
        if self.network_type is 'cnn':
            next_probs = torch.stack([self.actor_net(next_state) for next_state in next_states])
        else:
            next_probs = self.actor_net(next_states)

        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1_value = torch.stack([self.target_critic_net_1(next_state) for next_state in next_states])
        q2_value = torch.stack([self.target_critic_net_2(next_state) for next_state in next_states])
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        td_target = rewards + self.discount * next_value * (1 - terminated)

        return td_target


    def update(self, transition_dict):
        if self.network_type is 'cnn':
            states = torch.tensor(transition_dict['states'], dtype=torch.float)
            actions = torch.tensor(transition_dict['actions']).view(-1, 1)
            rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1)
            next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float)
            dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1)
        else:
            states = torch.stack([torch.flatten(torch.tensor(state, dtype=torch.float)) for state in transition_dict['states']])
            actions = torch.tensor(transition_dict['actions']).view(-1, 1)
            rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1)
            next_states = torch.stack([torch.flatten(torch.tensor(state,  dtype=torch.float)) for state in transition_dict['next_states']])
            dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1)

        td_target = self.calculate_target_value(rewards, next_states, dones)

        if self.network_type is 'cnn':
            critic_1_q_values = torch.stack([self.critic_net_1(state) for state in states])
        else:
            critic_1_q_values = self.critic_net_1(states)

        critic_1_loss = torch.mean(F.mse_loss(critic_1_q_values, td_target.detach()))

        if self.network_type is 'cnn':
            critic_2_q_values = torch.stack([self.critic_net_2(state) for state in states])
        else:
            critic_2_q_values = self.critic_net_2(states)

        critic_2_loss = torch.mean(F.mse_loss(critic_2_q_values, td_target.detach()))

        self.critic_net_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_net_1_optimizer.step()
        self.critic_net_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_net_2_optimizer.step()

        if self.network_type is 'cnn':
            probs = torch.stack([self.actor_net(state) for state in states])
        else:
            probs = self.actor_net(states)

        log_probs = torch.log(probs + 1e-8)

        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)

        if self.network_type is 'cnn':
            q1_value = torch.stack([self.critic_net_1(state) for state in states])
            q2_value = torch.stack([self.critic_net_2(state) for state in states])
        else:
            q1_value = self.critic_net_1(states)
            q2_value = self.critic_net_2(states)

        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        self.actor_net_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_net_optimizer.step()

        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_net_1, self.target_critic_net_1)
        self.soft_update(self.critic_net_2, self.target_critic_net_2)

        return actor_loss, critic_1_loss + critic_2_loss

    def train(self, max_episodes, stop_criterion, criterion_episodes):
        txt = open(save_record_path + str(datetime.datetime.now()) + '_' + self.env_name +
                   '_lr_' + str(self.learning_rate) +
                   '_use_skip_frame_' + str(self.use_skip_frame) +
                   '_tau_' + str(self.tau) +
                   '_init_w_' + str(self.initial_weight_required) + '_sac.txt', 'a')
        now = time.time()
        num_steps = 0
        episode_rewards = []

        for episode in range(max_episodes):
            state, _ = self.env.reset()

            if self.env_name is 'CarRacing':
                state = util.box2d_preProcess(state)
            else:
                state = util.atari_preProcess(state)

            for i in range(self.stack_frame_num):
                self.frames.append(state)
            state = np.stack(self.frames)

            terminated = False
            truncated = False
            episode_rewards.append(0)

            policy_loss = 0
            value_loss = 0
            action_list = []
            skipped = 0
            last_action = None

            while not (terminated or truncated):
                if self.use_skip_frame:
                    if skipped == 0:
                        action = self.sac_policy(state)
                        last_action = action
                    else:
                        action = last_action
                else:
                    action = self.sac_policy(state)

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                episode_rewards[-1] += reward

                if self.env_name is 'CarRacing':
                    next_state = util.box2d_preProcess(next_state)
                else:
                    next_state = util.atari_preProcess(next_state)

                self.frames.append(next_state)
                next_state = np.stack(self.frames)

                self.replay_buffer.append((state, action, reward, next_state, terminated))

                if len(self.replay_buffer) >= self.minibatch_size:
                    transitions = random.sample(self.replay_buffer, self.batch_size)
                    states, actions, rewards, next_states, terminateds = zip(*transitions)
                    transition_dict = {'states': np.array(states), 'actions': actions,
                                       'next_states': np.array(next_states),
                                       'rewards': rewards, 'dones': terminateds}
                    p_loss, v_loss = self.update(transition_dict)

                    if p_loss != 0.0:
                        policy_loss = p_loss

                    if v_loss != 0.0:
                        value_loss = v_loss

                action_list.append(action)

                num_steps += 1

            txt.write("Episode {} done, steps = {}, actions = {}, p_loss = {}, v_loss = {}, rewards = {}, consume time = {}s\n".format(
                episode + 1, num_steps, util.display_action_distribution(action_list, len(self.actions)), policy_loss, value_loss, episode_rewards[episode], time.time() - now))
            print(f'Episode {episode + 1} done, steps = {num_steps}, '
                  f'actions = {util.display_action_distribution(action_list, len(self.actions))}, '
                  f'p_loss = {policy_loss}, v_loss = {value_loss}, '
                  f'epsilon = {self.epsilon}, rewards = {episode_rewards[episode]}')

            if episode >= criterion_episodes - 1 and episode % 100 == 0:
                if stop_criterion(episode_rewards[-criterion_episodes:]):
                    self.save(save_model_path + str(datetime.datetime.now()) + '_' + self.env_name + '_tau_' + str(self.tau).replace(".", "") + '_sac')
                    print(f'\nStopping criterion satisfied after {episode} episodes')
                    break
                else:
                    self.save(save_model_path + str(datetime.datetime.now()) + '_' + self.env_name + '_tau_' + str(self.tau).replace(".",
                                                                                                                "") + '_sac_fail')

        plt.figure(dpi=100)
        plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, label=f'Rewards')

        plt.xlabel('Episodes')
        plt.ylabel('Rewards per episode')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()

    def save(self, path):
        torch.save({'policy': self.actor_net.state_dict(),
                    'value_1': self.critic_net_1.state_dict(),
                    'value_2': self.critic_net_2.state_dict(),
                    'target_value_1': self.target_critic_net_1.state_dict(),
                    'target_value_2': self.target_critic_net_2.state_dict()}, path)

    def load(self, path):
        networks = torch.load(path)
        self.actor_net.load_state_dict(networks['policy'])
        self.critic_net_1.load_state_dict(networks['value_1'])
        self.critic_net_2.load_state_dict(networks['value_2'])
        self.target_critic_net_1.load_state_dict(networks['target_value_1'])
        self.target_critic_net_2.load_state_dict(networks['target_value_2'])

