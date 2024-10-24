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
from Network import CNNValueNetwork, CNNPolicyNetwork
import util

save_record_path = '/Users/kevin/PycharmProjects/reinforment-learning-work/work/record/'
save_model_path = '/Users/kevin/PycharmProjects/reinforment-learning-work/work/saved_model/'


class DQN():
    def __init__(self, env_name, env, model_name, config):
        self.env = env
        self.env_name = env_name
        self.qnet = CNNValueNetwork(config)
        self.target_qnet = CNNValueNetwork(config)

        if model_name is not None:
            self.load(model_name)
            print('load success')
        else:
            self.target_qnet.copy_from(self.qnet)

        self.max_step = config['max_step']
        self.replay_buffer = deque(maxlen=config['replay_buffer'])
        self.buffer_size = config['replay_buffer']
        self.minibatch_size = config['minibatch_size']
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
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)

        if stochastic and (random.random() <= self.epsilon or steps <= self.warm_up_steps):
            return np.random.choice(self.actions, size = 1, p=self.preferable_action_probs)[0]
        else:
            state = torch.tensor(state, dtype=torch.float)
            q = self.qnet.forward(state).detach()
            j = self.rng.permutation(self.env.action_space.n)
            return j[q[j].argmax().item()]

    def policy(self, state):
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)

        return self.qnet(state).detach().argmax().item()

    def update(self):
        loss = 0
        if len(self.replay_buffer) >= self.minibatch_size:
            batch = self.rng.choice(len(self.replay_buffer), size=self.minibatch_size, replace=False)
            if self.env_name.startswith('CarRacing'):
                inputs = torch.zeros((self.minibatch_size, self.stack_frame_num, 84, 84))
            else:
                inputs = torch.zeros((self.minibatch_size, self.stack_frame_num, 64, 80))

            targets = torch.zeros((self.minibatch_size, self.env.action_space.n))
            for n, index in enumerate(batch):
                state, action, reward, next_state, terminated = self.replay_buffer[index]
                state = torch.tensor(state, dtype=torch.float)
                inputs[n, :] = state
                targets[n, :] = self.target_qnet(state.unsqueeze(0)).detach()
                if terminated:
                    targets[n, action] = reward
                else:
                    next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)
                    targets[n, action] = reward + self.discount * self.target_qnet(next_state).detach().max()

            loss = self.qnet.update(inputs, targets, True)

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
                next_state = next_state
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
        self.policy_net = CNNPolicyNetwork(config)
        self.learning_rate = config['lr']
        self.initial_weight_required = config['initial_weight_required']

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
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)

        action_probs = self.policy_net(state).detach()
        dist = torch.distributions.Categorical(logits=action_probs)

        #if stochastic and (random.random() <= self.epsilon or steps <= self.warm_up_steps):
        #    return np.random.choice(self.actions, size = 1, p=self.preferable_action_probs)[0]
        if stochastic:
            return dist.sample().item()
        else:
            return dist.probs.argmax().item()

    def update(self, trajectory):
        states, actions, rewards = list(zip(*trajectory))
        returns = torch.zeros((len(trajectory),))
        returns[-1] = rewards[-1]
        '''
        G = 0
        loss = 0
        for i in reversed(range(len(rewards))):  
            reward = rewards[i]
            state = states[i]
            action = torch.tensor(actions[i]).view(-1, 1)
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            G = self.discount * G + reward
            loss = -log_prob * G  
            loss.backward()  
        self.policy_net.optimizer.step()  

        return loss
        '''
        for t in reversed(range(len(trajectory) - 1)):
            returns[t] = rewards[t] + self.discount * returns[t + 1]

        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions)

        return self.policy_net.update(states, actions, returns, True)

    def train(self, max_episodes, stop_criterion, criterion_episodes):
        txt = open(save_record_path + str(datetime.datetime.now()) + '_' + self.env_name +
                   '_greedy_' + str(self.epsilon).replace(".", "") +
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
                    self.save(save_model_path + str(datetime.datetime.now()) + '_' + self.env_name + '_greedy_' + str(self.epsilon).replace(".", "") +'_rein')
                    print(f'\nStopping criterion satisfied after {episode} episodes')
                    break
                else:
                    self.save(save_model_path + str(datetime.datetime.now()) + '_' + self.env_name + '_greedy_' + str(self.epsilon).replace(".", "") +'_rein_fail')

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
        self.policy_net = CNNPolicyNetwork(policy_config)
        self.value_net = CNNValueNetwork(value_config)

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
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)

        action_probs = self.policy_net(state).detach()
        dist = torch.distributions.Categorical(logits=action_probs)

        if stochastic:
            return dist.sample().item()
        else:
            return dist.probs.argmax().item()

    def update(self, state, action, reward, next_state, terminated):
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)

        if not torch.is_tensor(next_state):
            next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)

        if terminated:
            target = reward
        else:
            target = reward + self.discount * self.value_net(next_state).detach()

        target = torch.tensor([target], dtype=torch.float)
        delta = target - self.value_net(state).detach()

        action = torch.tensor(action)

        policy_loss = self.policy_net.update(state, action, delta, False)
        value_loss = self.value_net.update(state, target, False)
        return policy_loss, value_loss


    def train(self, max_episodes, stop_criterion, criterion_episodes):
        txt = open(save_record_path + str(datetime.datetime.now()) + '_' + self.env_name +
                   '_greedy_' + str(self.epsilon).replace(".", "") +
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

            txt.write("Episode {} done, steps = {}, actions = {}, p_loss = {}, v_loss = {}, rewards = {}. consume time = {}s\n".format(
                episode + 1, num_steps, util.display_action_distribution(actions, len(self.actions)), policy_loss, value_loss, episode_rewards[episode], time.time() - now))
            print(f'Episode {episode + 1} done, steps = {num_steps}, '
                  f'actions = {util.display_action_distribution(actions, len(self.actions))}, '
                  f'p_loss = {policy_loss}, v_loss = {value_loss}, '
                  f'epsilon = {self.epsilon}, rewards = {episode_rewards[episode]}')

            if episode >= criterion_episodes - 1 and episode % 100 == 0:
                if stop_criterion(episode_rewards[-criterion_episodes:]):
                    self.save(save_model_path + str(datetime.datetime.now()) + '_' + self.env_name + '_greedy_' + str(self.epsilon).replace(".",
                                                                                                                    "") + '_ac')
                    print(f'\nStopping criterion satisfied after {episode} episodes')
                    break
                else:
                    self.save(save_model_path + str(datetime.datetime.now()) + '_' + self.env_name + '_greedy_' + str(self.epsilon).replace(".",
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

