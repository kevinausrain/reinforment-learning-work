import numpy as np
import torch

import gymnasium as gym
import ale_py
from gymnasium import spaces
import moviepy.editor as mpy
from Agent import REINFORCE, ActorCritic, DQN, SAC
from collections import deque
import util
import matplotlib.pyplot as plt


rng = np.random.default_rng()
gym.register_envs(ale_py)
env = gym.make("ALE/Pong-v5")

q_net_config = {
    "conv1": [4, 32, 8, 4, 0],
    "conv2": [32, 64, 4, 2, 0],
    "conv3": [64, 64, 3, 1, 0],
    "fc1": [3072, 512],
    "fc2": [512, 6],
    "dnn_fc1": [20480, 1536],
    "dnn_fc2": [1536, 128],
    "dnn_fc3": [128, 6],
    "lr": 0.001,
    "alpha_lr": 0.01,
    "tau": 0.005,
    "discount": 0.8,
    "greedy": 0.95,
    "greedy_min": 0.05,
    "is_action_discrete": True,
    "is_observation_space_image": True,
    "stack_frame_num": 4,
    "skip_frame_num": 4,
    "replay_buffer": 100000,
    "max_step": 5000000,
    "minibatch_size": 1000,
    "batch_size": 64,
    "target_update": 10000,
    "actions": [0, 1, 2, 3, 4, 5],
    "preferable_action_probs": [0.16, 0.16, 0.17, 0.17, 0.17, 0.17],
    "warm_up_steps": 5000,
    "decay_speed": 0.000001,
    "env_name": 'Pong',
    "initial_weight_required": False,
    "use_skip_frame": False,
    "type": 'value',
    "network_type": 'dnn'
}

policy_config = {
    "conv1": [4, 32, 8, 4, 0],
    "conv2": [32, 64, 4, 2, 0],
    "conv3": [64, 64, 3, 1, 0],
    "fc1": [3072, 512],
    "fc2": [512, 6],
    "dnn_fc1": [20480, 200],
    "dnn_fc2": [200, 6],
    #"dnn_fc3": [128, 6],
    "lr": 0.0001,
    "alpha_lr": 0.01,
    "tau": 0.005,
    "discount": 0.9,
    "greedy": 0.95,
    "greedy_min": 0.05,
    "is_action_discrete": True,
    "is_observation_space_image": True,
    "stack_frame_num": 4,
    "skip_frame_num": 4,
    "replay_buffer": 100000,
    "max_step": 5000000,
    "minibatch_size": 1000,
    "batch_size": 64,
    "target_update": 10,
    "preferable_action_probs": [0.16, 0.16, 0.17, 0.17, 0.17, 0.17],
    "actions": [0, 1, 2, 3, 4, 5],
    "warm_up_steps": 500,
    "decay_speed": 0.000001,
    "env_name": 'Pong',
    "use_skip_frame": True,
    "initial_weight_required": True,
    "target_entropy": -1,
    "network_type": 'dnn',
    "type": "policy",
    "normalize_prob_required": False
}

value_config = {
    "conv1": [4, 32, 8, 4, 0],
    "conv2": [32, 64, 4, 2, 0],
    "conv3": [64, 64, 3, 1, 0],
    "fc1": [3072, 512],
    "fc2": [512, 1],
    "dnn_fc1": [20480, 200],
    "dnn_fc2": [200, 1],
    "dnn_fc3": [128, 1],
    "lr": 0.01,
    "discount": 0.95,
    "greedy": 0.05,
    "is_action_discrete": True,
    "is_observation_space_image": True,
    "permute": [2, 0, 1],
    "stack_frame_num": 4,
    "replay_buffer": 100000,
    "minibatch_size": 64,
    "target_update": 10,
    "env_name": 'Pong',
    "initial_weight_required": True,
    "type": 'value'
}

def pong_solve_by_reinforce(train_required, network_type, model_name):
    max_episodes = 10000
    max_steps = 10000
    criterion_episodes = 200
    stack_frame_num = policy_config['stack_frame_num']
    frames = deque(maxlen=stack_frame_num)

    if network_type is not None:
        policy_config['network_type'] = network_type

    agent = REINFORCE('Pong', env, model_name, config=policy_config)

    if train_required:
        agent.train(max_episodes, lambda x: min(x) >= 600, criterion_episodes)

    state, _ = env.reset()
    state = util.atari_preProcess(state)

    for i in range(stack_frame_num):
        frames.append(state)
    state = np.stack(frames)

    terminated = False
    truncated = False
    steps = 0
    total_reward = 0

    while not (terminated or truncated or steps > max_steps):
        action = agent.policy(state, stochastic=False)
        state, reward, terminated, truncated, info = env.step(action)
        state = util.atari_preProcess(state)
        frames.append(state)
        state = np.stack(frames)

        total_reward += reward
        steps += 1

    print(f'Reward: {total_reward}')



def pong_solve_by_actor_critic(train_required, network_type, model_name):
    max_episodes = 10000
    max_steps = 10000
    criterion_episodes = 100
    stack_frame_num = policy_config['stack_frame_num']
    frames = deque(maxlen=stack_frame_num)

    if network_type is not None:
        policy_config['network_type'] = network_type

    agent = ActorCritic('Pong', env, model_name, policy_config=policy_config, value_config=value_config)

    if train_required:
        agent.train(max_episodes, lambda x: min(x) >= 600, criterion_episodes)

    state, _ = env.reset()
    state = util.atari_preProcess(state)

    for i in range(stack_frame_num):
        frames.append(state)
    state = np.stack(frames)

    terminated = False
    truncated = False
    steps = 0
    total_reward = 0

    while not (terminated or truncated or steps > max_steps):
        action = agent.policy(state, stochastic=False)
        state, reward, terminated, truncated, info = env.step(action)
        state = util.atari_preProcess(state)
        frames.append(state)
        state = np.stack(frames)

        total_reward += reward
        steps += 1

    print(f'Reward: {total_reward}')



def pong_solve_by_dqn(train_required, network_type, model_name, greedy, decay_speed, preferable_action_probs):
    max_episodes = 10000
    max_steps = 10000
    criterion_episodes = 200
    stack_frame_num = q_net_config['stack_frame_num']
    frames = deque(maxlen=stack_frame_num)

    if network_type is not None:
        q_net_config['network_type'] = network_type

    if greedy is not None:
        q_net_config['greedy'] = greedy

    if decay_speed is not None:
        q_net_config['decay_speed'] = decay_speed

    if preferable_action_probs is not None:
        q_net_config['preferable_action_probs'] = preferable_action_probs

    agent = DQN('Pong', env, model_name, config=q_net_config)

    if train_required:
        agent.train(max_episodes, lambda x: min(x) >= 600, criterion_episodes)

    state, _ = env.reset()
    state = util.atari_preProcess(state)

    for i in range(stack_frame_num):
        frames.append(state)
    state = np.stack(frames)

    terminated = False
    truncated = False
    steps = 0
    total_reward = 0

    while not (terminated or truncated or steps > max_steps):
        action = agent.policy(state)
        state, reward, terminated, truncated, info = env.step(action)
        state = util.atari_preProcess(state)
        frames.append(state)
        state = np.stack(frames)
        total_reward += reward
        steps += 1

    print(f'Reward: {total_reward}')

def pong_solve_by_sac(train_required, network_type, model_name):
    max_episodes = 10000
    max_steps = 10000
    criterion_episodes = 200
    stack_frame_num = policy_config['stack_frame_num']
    frames = deque(maxlen=stack_frame_num)

    if network_type is not None:
        policy_config['network_type'] = network_type

    agent = SAC('Pong', env, model_name, policy_config=policy_config, value_config=value_config)

    if train_required:
        agent.train(max_episodes, lambda x: min(x) >= 15, criterion_episodes)

    state, _ = env.reset()
    state = util.atari_preProcess(state)

    for i in range(stack_frame_num):
        frames.append(state)
    state = np.stack(frames)

    terminated = False
    truncated = False
    steps = 0
    total_reward = 0

    while not (terminated or truncated or steps > max_steps):
        action = agent.sac_policy(state)
        state, reward, terminated, truncated, info = env.step(action)
        state = util.atari_preProcess(state)
        frames.append(state)
        state = np.stack(frames)

        total_reward += reward
        steps += 1

    print(f'Reward: {total_reward}')

# train from start
#pong_solve_by_dqn(True, 'dnn', None, 1.0, 1e-6, None)
#pong_solve_by_dqn(True, 'cnn', None, 1.0, 1e-6, None)

# train from start
#pong_solve_by_reinforce(True, 'dnn', None)
#pong_solve_by_reinforce(True, 'cnn', None)

# train from start
#pong_solve_by_actor_critic(True, 'dnn', None)
#pong_solve_by_actor_critic(True, 'cnn', None)

# train from start
#pong_solve_by_sac(True, 'dnn', None)
#pong_solve_by_sac(True, 'cnn', None)

#env.close()
'''
fig, axes = plt.subplots(1, 4, figsize=(10, 10))
states = []
env.reset()
a_state = None

for i in range(5):
    state, reward, terminated, truncated, info = env.step(0)
    a_state = state

states.append(util.atari_preProcess(a_state))
for i in range(3):
    state, reward, terminated, truncated, info = env.step(4)
    states.append(util.atari_preProcess(state))

for i in range(4):
    axes[i].imshow(states[i], cmap='gray')
    axes[i].axis('off')
plt.show()
'''