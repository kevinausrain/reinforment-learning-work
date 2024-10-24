import matplotlib.pyplot as plt

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

display_result('/Users/kevin/PycharmProjects/reinforment-learning-work/work/record/2024-10-21 18:03:33.519521_Pong-84x84_greedy_095_dqn.txt',
               'DQN PONG lr = 0.001, epsilon = 0.95 min = 0.05, decay = 1e-6, '
               'target update = 5, init weight = False', None)

display_result('/Users/kevin/PycharmProjects/reinforment-learning-work/work/record/2024-10-23 02:53:25.472859_Pong-84x84_greedy_095_rein.txt',
               'REINFORCE PONG lr = 0.001, init weight = True', 6)


display_result('/Users/kevin/PycharmProjects/reinforment-learning-work/work/record/2024-10-23 23:24:02.761045_CarRacing_greedy_10_greedy_min_01_decay_speed_1e-06_buffer_size_100000_target_update_freq_10000_lr_0.00025_init_w_False_dqn.txt',
               'DQN CARRACING lr = 0.00025, epsilon = 1.0 min = 0.1, decay = 1e-6, '
               'target update = 10000, init weight = False', 5)
