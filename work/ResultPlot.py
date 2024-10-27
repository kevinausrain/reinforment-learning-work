import matplotlib.pyplot as plt


car_action_dict = {
    0: 'do nothing',
    1: 'turn left',
    2: 'turn right',
    3: 'gas',
    4: 'brake'
}

atari_action_dict = {
    0: 'do nothing',
    1: 'fire',
    2: 'move right',
    3: 'move left',
    4: 'right fire',
    5: 'left fire'
}

def display_result(file, title, action_nums):
    txt = open(file, 'r')
    lines = txt.readlines()
    steps = []
    actions = dict()
    rewards = []

    if action_nums is not None:
        for i in range(action_nums):
            actions[i] = list()

    for line in lines:
        values = line.split(",")
        for value in values:
            if value.__contains__('step'):
                steps.append(int(value.split("=")[1].replace(" ", "")))
            if value.__contains__('reward'):
                rewards.append(float(value.split("=")[1].replace(" ", "")))
            if value.__contains__('action'):
                action_distribution = value.split("[")[1].split("|")
                for action in action_distribution:
                    if action.__contains__("/"):
                        params = action.split(":")
                        actions[int(params[0])].append(int(params[1].split("/")[0]))


    plt.figure(figsize=(10, 6))
    plt.plot(steps, rewards, label='reward')
    plt.xlabel('steps')

    if action_nums is not None:
        for key in actions.keys():
            plt.plot(steps, actions[key], label='action:{}'.format(key))

    if action_nums is None:
        plt.ylabel('rewards')
    else:
        plt.ylabel('rewards and actions')

    plt.title(title)
    plt.legend()
    plt.show()

def display_action_reward(file, title):
    txt = open(file, 'r')
    lines = txt.readlines()
    steps = []
    actions = []
    rewards = []
    penalty_actions = []

    for line in lines:
        if float(line.split(',')[2]) == -1.0:
            penalty_actions.append(int(line.split(',')[1]))

        steps.append(int(line.split(',')[0]))
        actions.append(int(line.split(',')[1]))
        rewards.append(float(line.split(',')[2]))

    print(penalty_actions)

    plt.figure(figsize=(30, 6))
    plt.plot(steps, rewards)
    plt.plot(steps, actions)
    plt.xlabel('steps')
    plt.ylabel('rewards')

    plt.title(title)
    plt.legend()
    plt.show()


#display_action_reward('/Users/kevin/PycharmProjects/reinforment-learning-work/work/record/2024-10-26 03:00:04.409273_Pong_rein_reward.txt',
#               'PONG reward for each step')

display_result(
    '/Users/kevin/PycharmProjects/reinforment-learning-work/work/record/2024-10-24 18:13:18.416525_SpaceInvader_greedy_10_greedy_min_01_decay_speed_1e-06_buffer_size_10000_target_update_freq_20_use_skip_frame_True_lr_0.001_init_w_True_dqn.txt',
               'SPACE INVADER DQN lr = 0.001, gamma = 0.99, epsilon = 1.0 min = 0.1, decay = 1e-6, '
               'target update = 10000, init weight = False', 6)

display_result(
    '/Users/kevin/PycharmProjects/reinforment-learning-work/work/record/2024-10-27 04:59:46.612269_SpaceInvader_greedy_09_greedy_min_005_decay_speed_1e-06_buffer_size_100000_target_update_freq_10_use_skip_frame_False_lr_0.001_init_w_False_dqn.txt',
               'SPACE INVADER DQN lr = 0.001, gamma = 0.95, epsilon = 0.9 min = 0.1, decay = 1e-6, '
               'target update = 20, init weight = True', 6)

display_result('/Users/kevin/PycharmProjects/reinforment-learning-work/work/record/2024-10-26 18:15:30.277496_CarRacing_greedy_03_greedy_min_01_decay_speed_1e-06_buffer_size_100000_target_update_freq_20_use_skip_frame_True_lr_0.0005_init_w_False_dqn.txt',
               'CARRACING DQN lr = 0.0005, gamma = 0.9, epsilon = 0.3 min = 0.1, decay = 1e-6, '
               'target update = 20, init weight = False', 5)

'''
display_result('/Users/kevin/PycharmProjects/reinforment-learning-work/work/record/2024-10-23 02:53:25.472859_Pong-84x84_greedy_095_rein.txt',
               'REINFORCE PONG lr = 0.001, init weight = True', 6)


display_result(
    '/Users/kevin/PycharmProjects/reinforment-learning-work/work/record/2024-10-23 23:24:02.761045_CarRacing_greedy_10_greedy_min_01_decay_speed_1e-06_buffer_size_100000_target_update_freq_10000_use_skip_frame_False_lr_0.00025_init_w_False_dqn.txt',
               'DQN CARRACING lr = 0.00025, epsilon = 1.0 min = 0.1, decay = 1e-6, '
               'target update = 10000, init weight = False', 5)


display_result(
    '/Users/kevin/PycharmProjects/reinforment-learning-work/work/record/2024-10-24 18:13:18.416525_SpaceInvader_greedy_10_greedy_min_01_decay_speed_1e-06_buffer_size_10000_target_update_freq_20_use_skip_frame_True_lr_0.001_init_w_True_dqn.txt',
               'DQN CARRACING lr = 0.00025, epsilon = 1.0 min = 0.1, decay = 1e-6, '
               'target update = 10000, init weight = False', 6)
'''