import numpy as np
import torch

import gymnasium as gym
import ale_py
from gymnasium import spaces
import moviepy.editor as mpy
from Agent import REINFORCE, ActorCritic, DQN
from collections import deque

rng = np.random.default_rng()
gym.register_envs(ale_py)
env = gym.make("ALE/Pong-v5")
env = gym.wrappers.ResizeObservation(env, (84, 84))
env = gym.wrappers.GrayscaleObservation(env)

policy_config = {
    "conv1": [4, 32, 8, 4, 0],
    "conv2": [32, 64, 4, 2, 0],
    "conv3": [64, 64, 3, 1, 0],
    "fc1": [3136, 512],
    "fc2": [512, 3],
    "lr": 0.001,
    "discount": 0.99,
    "greedy": 0.9,
    "greedy_min": 0.0,
    "is_action_discrete": True,
    "is_observation_space_image": True,
    "permute": [2, 0, 1],
    "stack_num": 4,
    "replay_buffer": 10000,
    "minibatch_size": 32,
    "target_update": 20,
    "preferable_action_probs": [0.2, 0.4, 0.4],
    "actions": [0, 1, 2],
    "warm_up_steps": 500,
    "decay_speed": 0.1,
}

value_config = {
    "conv1": [4, 32, 8, 4, 0],
    "conv2": [32, 64, 4, 2, 0],
    "conv3": [64, 64, 3, 1, 0],
    "fc1": [3136, 512],
    "fc2": [512, 1],
    "lr": 0.001,
    "discount": 0.99,
    "greedy": 0.1,
    "is_action_discrete": True,
    "is_observation_space_image": True,
    "permute": [2, 0, 1],
    "stack_num": 4,
    "replay_buffer": 10000,
    "minibatch_size": 32,
    "target_update": 20
}

def solve_by_reinforce():
    max_episodes = 2000
    max_steps = 2000
    criterion_episodes = 200
    stack_num = policy_config['stack_num']
    frames = deque(maxlen=stack_num)

    policy_config['decay_speed'] = 0.02
    agent = REINFORCE('Pong-84x84', env, config=policy_config)
    agent.train(max_episodes, lambda x: min(x) >= 15, criterion_episodes)

    state, _ = env.reset()
    #state = util.atari_v2_image_preprocess(state)
    state = torch.tensor(state, dtype=torch.float)

    for i in range(stack_num):
        frames.append(state)
    state = torch.tensor(np.stack(frames))

    terminated = False
    truncated = False
    steps = 0
    total_reward = 0

    while not (terminated or truncated or steps > max_steps):
        action = agent.policy(state, stochastic=False)
        if action != 0:
            action = action + 1
        state, reward, terminated, truncated, info = env.step(action)
        #state = util.atari_v2_image_preprocess(state)
        state = torch.tensor(state, dtype=torch.float)
        frames.append(state)
        state = torch.tensor(np.stack(frames))

        total_reward += reward
        steps += 1

    print(f'Reward: {total_reward}')

    frames = env.render()
    env.close()
    clip = mpy.ImageSequenceClip(frames, fps=50)
    clip.ipython_display(rd_kwargs=dict(logger=None))


def solve_by_actor_critic():
    max_episodes = 2000
    max_steps = 2000
    criterion_episodes = 100
    stack_num = policy_config['stack_num']
    frames = deque(maxlen=stack_num)

    policy_config['decay_speed'] = 0.02
    agent = ActorCritic('Pong-84x84', env, policy_config=policy_config, value_config=value_config)
    agent.train(max_episodes, lambda x: min(x) >= 15, criterion_episodes)

    state, _ = env.reset()
    #state = util.atari_v2_image_preprocess(state)
    state = torch.tensor(state, dtype=torch.float)

    for i in range(stack_num):
        frames.append(state)
    state = torch.tensor(np.stack(frames))

    terminated = False
    truncated = False
    steps = 0
    total_reward = 0

    while not (terminated or truncated or steps > max_steps):
        action = agent.policy(state, stochastic=False)
        state, reward, terminated, truncated, info = env.step(action)
        #state = util.atari_v2_image_preprocess(state)
        state = torch.tensor(state, dtype=torch.float)
        frames.append(state)
        state = torch.tensor(np.stack(frames))

        total_reward += reward
        steps += 1

    print(f'Reward: {total_reward}')

    frames = env.render()
    env.close()

    # create and play video clip using the frames and given fps
    clip = mpy.ImageSequenceClip(frames, fps=50)
    clip.ipython_display(rd_kwargs=dict(logger=None))


def solve_by_dqn():
    max_episodes = 2000
    max_steps = 2000
    criterion_episodes = 200
    stack_num = policy_config['stack_num']
    frames = deque(maxlen=stack_num)

    policy_config['decay_speed'] = 0.05
    agent = DQN('Pong-84x84', env, config=policy_config)
    agent.train(max_episodes, lambda x: min(x) >= 15, criterion_episodes)

    state, _ = env.reset()
    #state = util.atari_v2_image_preprocess(state)
    state = torch.tensor(state, dtype=torch.float)

    for i in range(stack_num):
        frames.append(state)
    state = torch.tensor(np.stack(frames))

    terminated = False
    truncated = False
    steps = 0
    total_reward = 0

    while not (terminated or truncated or steps > max_steps):
        action = agent.policy(state)
        state, reward, terminated, truncated, info = env.step(action)
        #state = util.atari_v2_image_preprocess(state)
        state = torch.tensor(state, dtype=torch.float)
        frames.append(state)
        state = torch.tensor(np.stack(frames))

        total_reward += reward
        steps += 1

    print(f'Reward: {total_reward}')

    frames = env.render()
    env.close()


solve_by_dqn()
solve_by_reinforce()
#solve_by_actor_critic()
