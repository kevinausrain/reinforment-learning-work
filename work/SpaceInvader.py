import numpy as np
import torch

import gymnasium as gym
import ale_py
from gymnasium import spaces
import moviepy.editor as mpy
from Agent import REINFORCE, ActorCritic, DQN
from collections import deque
import util


rng = np.random.default_rng()
gym.register_envs(ale_py)
env = gym.make("ALE/SpaceInvaders-v5")

policy_config = {
    "conv1": [4, 32, 8, 4, 0],
    "conv2": [32, 64, 4, 2, 0],
    "conv3": [64, 64, 3, 2, 0],
    "fc1": [384, 128],
    "fc2": [128, 6],
    "lr": 0.001,
    "discount": 0.99,
    "greedy": 1.0,
    "greedy_min": 0.1,
    "max_step": 2000000,
    "is_action_discrete": True,
    "is_observation_space_image": True,
    "permute": [2, 0, 1],
    "stack_frame_num": 4,
    "skip_frame_num": 4,
    "replay_buffer": 10000,
    "minibatch_size": 32,
    "target_update": 20,
    "preferable_action_probs": [0.16, 0.16, 0.17, 0.17, 0.17, 0.17],
    "actions": [0, 1, 2, 3, 4, 5],
    "warm_up_steps": 5000,
    "decay_speed": 0.05,
    "env_name": 'SpaceInvader',
    "use_skip_frame": True,
    "initial_weight_required": True
}

value_config = {
    "conv1": [4, 32, 8, 4, 0],
    "conv2": [32, 64, 4, 2, 0],
    "conv3": [64, 64, 3, 2, 0],
    "fc1": [384, 128],
    "fc2": [128, 1],
    "lr": 0.001,
    "discount": 0.99,
    "greedy": 0.1,
    "is_action_discrete": True,
    "is_observation_space_image": True,
    "permute": [2, 0, 1],
    "stack_frame_num": 4,
    "replay_buffer": 10000,
    "minibatch_size": 32,
    "env_name": 'SpaceInvader',
    "initial_weight_required": True
}


def solve_by_reinforce(train_required, model_name, greedy):
    max_episodes = 10000
    max_steps = 10000
    criterion_episodes = 200
    stack_frame_num = policy_config['stack_frame_num']
    frames = deque(maxlen=stack_frame_num)

    if greedy is not None:
        policy_config['greedy'] = greedy

    agent = REINFORCE('SpaceInvader', env, model_name, config=policy_config)

    if train_required:
        agent.train(max_episodes, lambda x: min(x) >= 1200, criterion_episodes)

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


def solve_by_actor_critic(train_required, model_name, greedy):
    max_episodes = 10000
    max_steps = 10000
    criterion_episodes = 100
    stack_frame_num = policy_config['stack_frame_num']
    frames = deque(maxlen=stack_frame_num)

    if greedy is not None:
        policy_config['greedy'] = greedy

    agent = ActorCritic('SpaceInvader', env, model_name, policy_config=policy_config, value_config=value_config)

    if train_required:
        agent.train(max_episodes, lambda x: min(x) >= 1200, criterion_episodes)

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


def solve_by_dqn(train_required, model_name, greedy, decay_speed, preferable_action_probs):
    max_episodes = 10000
    max_steps = 10000
    criterion_episodes = 200
    stack_frame_num = policy_config['stack_frame_num']
    frames = deque(maxlen=stack_frame_num)

    # pre load model and set greedy
    if greedy is not None:
        policy_config['greedy'] = greedy

    if decay_speed is not None:
        policy_config['decay_speed'] = decay_speed

    if preferable_action_probs is not None:
        policy_config['preferable_action_probs'] = preferable_action_probs

    agent = DQN('SpaceInvader', env, model_name, config=policy_config)

    if train_required:
        agent.train(max_episodes, lambda x: min(x) >= 1200, criterion_episodes)

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



# train from start
solve_by_dqn(True,None, 1.0, 1e-6, None)
# train from halfway trained model
#solve_by_dqn(True,'2024-10-23 11:37:20.482783_CarRacing_greedy_005_dqn_fail', 0.05, None, [0.1, 0.6, 0.1, 0.1, 0.1])
# evaluate
#solve_by_dqn(False, '2024-10-23 11:37:20.482783_CarRacing_greedy_005_dqn_fail', None, None, None)

# train from start
solve_by_reinforce(True, None, None)
# evaluate
#solve_by_reinforce(False, 'model', None)

# train from start
solve_by_actor_critic(True, None, None)
# evaluate
#solve_by_actor_critic(False, 'model', None)
env.close()