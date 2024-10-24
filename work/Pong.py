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
env = gym.make("ALE/Pong-v5")

policy_config = {
    "conv1": [4, 32, 8, 4, 0],
    "conv2": [32, 64, 4, 2, 0],
    "conv3": [64, 64, 3, 1, 0],
    "fc1": [1536, 512],
    "fc2": [512, 6],
    "lr": 0.001,
    "discount": 0.99,
    "greedy": 0.95,
    "greedy_min": 0.05,
    "is_action_discrete": True,
    "is_observation_space_image": True,
    "permute": [2, 0, 1],
    "stack_frame_num": 4,
    "replay_buffer": 50000,
    "max_step": 5000000,
    "minibatch_size": 32,
    "target_update": 10,
    "preferable_action_probs": [0.16, 0.16, 0.17, 0.17, 0.17, 0.17],
    "actions": [0, 1, 2, 3, 4, 5],
    "warm_up_steps": 500,
    "decay_speed": 0.001,
    "env_name": 'Pong'
}

value_config = {
    "conv1": [4, 32, 8, 4, 0],
    "conv2": [32, 64, 4, 2, 0],
    "conv3": [64, 64, 3, 1, 0],
    "fc1": [1536, 512],
    "fc2": [512, 1],
    "lr": 0.001,
    "discount": 0.95,
    "greedy": 0.05,
    "is_action_discrete": True,
    "is_observation_space_image": True,
    "permute": [2, 0, 1],
    "stack_frame_num": 4,
    "replay_buffer": 10000,
    "minibatch_size": 32,
    "target_update": 10,
    "env_name": 'Pong'
}

def solve_by_reinforce(train_required, model_name, greedy):
    max_episodes = 20000
    max_steps = 20000
    criterion_episodes = 200
    stack_frame_num = policy_config['stack_frame_num']
    frames = deque(maxlen=stack_frame_num)

    if greedy is not None:
        policy_config['greedy'] = greedy

    agent = REINFORCE('Pong-84x84', env, model_name, config=policy_config)

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
        action = agent.policy(state, stochastic=False)
        if action != 0:
            action = action + 1
        state, reward, terminated, truncated, info = env.step(action)
        state = util.atari_preProcess(state)
        frames.append(state)
        state = np.stack(frames)

        total_reward += reward
        steps += 1

    print(f'Reward: {total_reward}')


def solve_by_actor_critic(train_required, model_name, greedy):
    max_episodes = 5000
    max_steps = 2000
    criterion_episodes = 300
    stack_frame_num = policy_config['stack_frame_num']
    frames = deque(maxlen=stack_frame_num)

    if greedy is not None:
        policy_config['greedy'] = greedy

    agent = ActorCritic('Pong-84x84', env, model_name, policy_config=policy_config, value_config=value_config)

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
        action = agent.policy(state, stochastic=False)
        state, reward, terminated, truncated, info = env.step(action)
        state = util.atari_preProcess(state)
        frames.append(state)
        state = np.stack(frames)

        total_reward += reward
        steps += 1

    print(f'Reward: {total_reward}')


def solve_by_dqn(train_required, model_name, greedy, decay_speed, preferable_action_probs):
    max_episodes = 2000
    max_steps = 2000
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

    agent = DQN('Pong-84x84', env, model_name, config=policy_config)

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
