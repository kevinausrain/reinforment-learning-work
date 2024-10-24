import numpy as np
import torch

import gymnasium as gym
from gymnasium import spaces
import moviepy.editor as mpy
from Agent import REINFORCE, ActorCritic, DQN, SAC
from collections import deque
import util

rng = np.random.default_rng()
env = gym.make("CarRacing-v2", continuous=False)

q_net_config = {
    "conv1": [4, 16, 8, 4, 0],
    "conv2": [16, 32, 4, 2, 0],
    "fc1": [32 * 9 * 9, 256],
    "fc2": [256, 5],
    "dnn_fc1": [28224, 1536],
    "dnn_fc2": [1536, 128],
    "dnn_fc3": [128, 5],
    "lr": 0.00025,
    "alpha_lr": 0.01,
    "tau": 0.005,
    "discount": 0.99,
    "greedy": 1.0,
    "greedy_min": 0.1,
    "max_step": 5000000,
    "is_action_discrete": True,
    "is_observation_space_image": True,
    "stack_frame_num": 4,
    "skip_frame_num": 4,
    "replay_buffer": 100000,
    "minibatch_size": 1000,
    "batch_size": 64,
    "target_update": 10000,
    "actions": [0, 1, 2, 3, 4],
    "preferable_action_probs": [0.2, 0.2, 0.2, 0.2, 0.2],
    "warm_up_steps": 5000,
    "decay_speed": 0.000001,
    "env_name": 'CarRacing',
    "initial_weight_required": False,
    "use_skip_frame": True,
    "type": 'value',
    "network_type": 'dnn'
}

policy_config = {
    "conv1": [4, 16, 8, 4, 0],
    "conv2": [16, 32, 4, 2, 0],
    "fc1": [32 * 9 * 9, 256],
    "fc2": [256, 5],
    "dnn_fc1": [28224, 1536],
    "dnn_fc2": [1536, 128],
    "dnn_fc3": [128, 5],
    "lr": 0.00025,
    "alpha_lr": 0.01,
    "tau": 0.005,
    "discount": 0.99,
    "greedy": 1.0,
    "greedy_min": 0.1,
    "max_step": 5000000,
    "is_action_discrete": True,
    "is_observation_space_image": True,
    "stack_frame_num": 4,
    "skip_frame_num": 4,
    "replay_buffer": 100000,
    "minibatch_size": 1000,
    "batch_size": 64,
    "target_update": 10000,
    "actions": [0, 1, 2, 3, 4],
    "preferable_action_probs": [0.2, 0.2, 0.2, 0.2, 0.2],
    "warm_up_steps": 5000,
    "decay_speed": 0.000001,
    "env_name": 'CarRacing',
    "initial_weight_required": False,
    "use_skip_frame": True,
    "type": 'policy',
    "network_type": 'dnn',
    "target_entropy": -1,
    "normalize_prob_required": False
}
value_config = {
    "conv1": [4, 16, 8, 4, 0],
    "conv2": [16, 32, 4, 2, 0],
    "fc1": [32 * 9 * 9, 256],
    "fc2": [256, 1],
    "dnn_fc1": [28224, 1536],
    "dnn_fc2": [1536, 128],
    "dnn_fc3": [128, 1],
    "lr": 0.01,
    "discount": 0.99,
    "greedy": 0.05,
    "is_action_discrete": True,
    "is_observation_space_image": True,
    "stack_frame_num": 4,
    "replay_buffer": 500000,
    "minibatch_size": 1000,
    "batch_size": 64,
    "target_update": 10,
    "env_name": 'CarRacing',
    "initial_weight_required": False,
    "type": 'value'
}


def car_racing_solve_by_reinforce(train_required, network_type, model_name):
    max_episodes = 10000
    max_steps = 10000
    criterion_episodes = 200
    stack_frame_num = policy_config['stack_frame_num']
    frames = deque(maxlen=stack_frame_num)

    if network_type is not None:
        policy_config['network_type'] = network_type

    agent = REINFORCE('CarRacing', env, model_name, config=policy_config)

    if train_required:
        agent.train(max_episodes, lambda x: min(x) >= 600, criterion_episodes)

    state, _ = env.reset()
    state = util.box2d_preProcess(state)

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
        state = util.box2d_preProcess(state)
        frames.append(state)
        state = np.stack(frames)

        total_reward += reward
        steps += 1

    print(f'Reward: {total_reward}')



def car_racing_solve_by_actor_critic(train_required, network_type, model_name):
    max_episodes = 10000
    max_steps = 10000
    criterion_episodes = 100
    stack_frame_num = policy_config['stack_frame_num']
    frames = deque(maxlen=stack_frame_num)

    if network_type is not None:
        policy_config['network_type'] = network_type

    agent = ActorCritic('CarRacing', env, model_name, policy_config=policy_config, value_config=value_config)

    if train_required:
        agent.train(max_episodes, lambda x: min(x) >= 600, criterion_episodes)

    state, _ = env.reset()
    state = util.box2d_preProcess(state)

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
        state = util.box2d_preProcess(state)
        frames.append(state)
        state = np.stack(frames)

        total_reward += reward
        steps += 1

    print(f'Reward: {total_reward}')



def car_racing_solve_by_dqn(train_required, network_type, model_name, greedy, decay_speed, preferable_action_probs):
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

    agent = DQN('CarRacing', env, model_name, config=q_net_config)

    if train_required:
        agent.train(max_episodes, lambda x: min(x) >= 600, criterion_episodes)

    state, _ = env.reset()
    state = util.box2d_preProcess(state)

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
        state = util.box2d_preProcess(state)
        frames.append(state)
        state = np.stack(frames)
        total_reward += reward
        steps += 1

    print(f'Reward: {total_reward}')

def car_racing_solve_by_sac(train_required, network_type, model_name):
    max_episodes = 10000
    max_steps = 10000
    criterion_episodes = 200
    stack_frame_num = policy_config['stack_frame_num']
    frames = deque(maxlen=stack_frame_num)

    if network_type is not None:
        policy_config['network_type'] = network_type

    agent = SAC('CarRacing', env, model_name, policy_config=policy_config, value_config=value_config)

    if train_required:
        agent.train(max_episodes, lambda x: min(x) >= 15, criterion_episodes)

    state, _ = env.reset()
    state = util.box2d_preProcess(state)

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
        state = util.box2d_preProcess(state)
        frames.append(state)
        state = np.stack(frames)

        total_reward += reward
        steps += 1

    print(f'Reward: {total_reward}')

# train from start
#car_racing_solve_by_dqn(True,'dnn', None, 1.0, 1e-6, [0.2, 0.2, 0.2, 0.2, 0.2])
#car_racing_solve_by_dqn(True,'cnn', None, 1.0, 1e-6, [0.2, 0.2, 0.2, 0.2, 0.2])

# train from start
#car_racing_solve_by_reinforce(True, 'dnn', None)
#car_racing_solve_by_reinforce(True, 'cnn', None)

# train from start
#car_racing_solve_by_actor_critic(True, 'dnn', None)
car_racing_solve_by_actor_critic(True, 'cnn', None)

# train from start
#car_racing_solve_by_sac(True, 'dnn', None)
#car_racing_solve_by_sac(True, 'cnn', None)

env.close()
