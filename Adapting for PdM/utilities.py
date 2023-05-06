# Setting background color
# ax.set_facecolor('#EFEFEF')

import numpy as np
import matplotlib.pyplot as plt
import datetime

def write_test_results(results, results_file):
    from csv import writer
    with open(results_file, 'a', newline='') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(results)
        f_object.close()

    print(f'- Test results written to file: {results_file}')
    
def test_script(df, algo, episodes, env, agent, test_cases, data_file, results_file):
    dt = datetime.datetime.now()
    dt_d = dt.strftime('%m-%b-%Y')
    dt_t = dt.strftime('%H:%M:%S')

    cumm_error_0 = 0.0
    cumm_error_1 = 0.0
    n_0 = 0
    n_1 = 0
    for idx in test_cases:
        state = env._get_observation(idx)
        action_pred, next_state = agent.predict(state)
        action_actual = int(df['ACTION_CODE'][idx])
        e = int(action_pred - action_actual)
        if action_actual:
            cumm_error_1 += abs(e)
            n_1 += 1
            # print(f'    {idx:4d}: VB (mm): {state[1]*17.855:6.3f} \t Action predicted: {action_pred} \t actual: {action_actual} \t error: {e}')
        else:
            cumm_error_0 += abs(e)
            n_0 += 1
            # print(f' ** {idx:4d}: VB (mm): {state[1]*17.855:6.3f} \t Action predicted: {action_pred} \t actual: {action_actual} \t error: {e}')

    if n_0 == 0: n_0 = 1
    if n_1 == 0: n_1 = 1

    print(f'Algo.\tNormal\tError\tReplace\tError\tOverall')
    print(f'\tcases\t\tcases\t\terror')
    print(50*'-')
    print(f'{algo}\t{n_0}\t{cumm_error_0/n_0:5.3f}\t{n_1}\t{cumm_error_1/n_1:5.3f}\t{(cumm_error_0 + cumm_error_1)/(n_0+n_1):5.3f}')
    print(50*'-')
    
    # File format
    # Date	Time	Enviroment	Data_file	Test_set	Algo.	Episodes	Normal_cases	Normal_Error\
    # Replacement_cases	Replacement_Error	Overall_Error	Parameter	Value
    # 5/7/2023	51:35.2	Simple ME	Tool_Wear_VB.csv	Sampled from training	A2C	300	37	54.1%	63	54.0%	54.0%		
    results = [dt_d, dt_t, 'Simple', data_file, 'Sampled from training', algo, episodes, n_0, cumm_error_0/n_0, n_1, cumm_error_1/n_1, (cumm_error_0 + cumm_error_1)/(n_0+n_1)]

    return results


# def test_script(df, algo_SB, env_RF, agent_RF, env_SB, agent_SB, test_cases, results_file):
#     cumm_error_RF_0 = 0.0
#     cumm_error_RF_1 = 0.0
#     n_RF_0 = 0
#     n_RF_1 = 0
#     for idx in test_cases:
#         state = env_RF._get_observation(idx)
#         action_pred = agent_RF.act(state)
#         action_actual = int(df['ACTION_CODE'][idx])
#         e = int(action_pred - action_actual)
#         if idx>77:
#             cumm_error_RF_0 += abs(e)
#             n_RF_0 += 1
#             # print(f' ** {idx:4d}: VB (mm): {state[1]*17.855:6.3f} \t Action predicted: {action_pred} \t actual: {action_actual} \t error: {e}')
#         else:
#             cumm_error_RF_1 += abs(e)
#             n_RF_1 += 1
#             # print(f'    {idx:4d}: VB (mm): {state[1]*17.855:6.3f} \t Action predicted: {action_pred} \t actual: {action_actual} \t error: {e}')

#     if n_RF_0 == 0: n_RF_0 = 1
#     if n_RF_1 == 0: n_RF_1 = 1
#     # print(50*'-', f'\n Test-1: REINFORCE algorithm (300 episodes)')
#     # print(50*'-')
#     # print(f' Avg. Error for NORMAL {n_RF_0} cases: \t {cumm_error_RF_0*100.00/n_RF_0:6.2f} %')
#     # print(f' Avg. Error for REPLACEMENT {n_RF_1} cases: \t {cumm_error_RF_1*100.00/n_RF_1:6.2f} %')
#     # print(f' Overall avg. error {n_RF_0+n_RF_1} cases: \t\t {(cumm_error_RF_0 + cumm_error_RF_1)*100.00/(n_RF_0+n_RF_1):6.2f} %')

#     ## Stable Baselines Test
#     cumm_error_SB_0 = 0.0
#     cumm_error_SB_1 = 0.0
#     n_SB_0 = 0
#     n_SB_1 = 0
#     for idx in test_cases:
#         state = env_SB._get_observation(idx)
#         action_pred, next_state = agent_SB.predict(state)
#         action_actual = int(df['ACTION_CODE'][idx])
#         e = int(action_pred - action_actual)
#         if idx>77:
#             cumm_error_SB_0 += abs(e)
#             n_SB_0 += 1
#             # print(f' ** {idx:4d}: VB (mm): {state[1]*17.855:6.3f} \t Action predicted: {action_pred} \t actual: {action_actual} \t error: {e}')
#         else:
#             cumm_error_SB_1 += abs(e)
#             n_SB_1 += 1
#             # print(f'    {idx:4d}: VB (mm): {state[1]*17.855:6.3f} \t Action predicted: {action_pred} \t actual: {action_actual} \t error: {e}')

#     if n_SB_0 == 0: n_SB_0 = 1
#     if n_SB_1 == 0: n_SB_1 = 1
#     # print(50*'-', f'\n Test-2: Stable Baselines algorithm (3k episodes)')
#     # print(50*'-')
#     # print(f' Avg. Error for NORMAL {n_SB_0} cases: \t {cumm_error_SB_0*100.00/n_SB_0:6.2f} %')
#     # print(f' Avg. Error for REPLACEMENT {n_SB_1} cases: \t {cumm_error_SB_1*100.00/n_SB_1:6.2f} %')
#     # print(f' Overall avg. error {n_SB_0+n_SB_1} cases: \t\t {(cumm_error_SB_0 + cumm_error_SB_1)*100.00/(n_SB_0+n_SB_1):6.2f} %')
#     print(50*'-')
#     print(f'Algo.\tNormal\tError\tReplace\tError\tOverall')
#     print(f'\tcases\t\tcases\t\terror')
#     print(50*'-')
#     print(f'RF\t{n_RF_0}\t{cumm_error_RF_0/n_RF_0:5.3f}\t{n_RF_1}\t{cumm_error_RF_1/n_RF_1:5.3f}\t{(cumm_error_RF_0 + cumm_error_RF_1)/(n_RF_0+n_RF_1):5.3f}')
#     print(f'{algo_SB}\t{n_SB_0}\t{cumm_error_SB_0/n_SB_0:5.3f}\t{n_SB_1}\t{cumm_error_SB_1/n_SB_1:5.3f}\t{(cumm_error_SB_0 + cumm_error_SB_1)/(n_SB_0+n_SB_1):5.3f}')
#     print(50*'-')
    
#     # File format
#     # Date	Time	Enviroment	Data_file	Test_set	Algo.	Episodes	Normal_cases	Normal_Error\
#     # Replacement_cases	Replacement_Error	Overall_Error	Parameter	Value
#     # 5/7/2023	51:35.2	Simple ME	Tool_Wear_VB.csv	Sampled from training	A2C	300	37	54.1%	63	54.0%	54.0%		

#     rf_results = ['RF', n_RF_0, cumm_error_RF_0/n_RF_0, n_RF_1, cumm_error_RF_1/n_RF_1, (cumm_error_RF_0 + cumm_error_RF_1)/(n_RF_0+n_RF_1)]
#     sb_results = [algo_SB, n_SB_0, cumm_error_SB_0/n_SB_0, n_SB_1, cumm_error_SB_1/n_SB_1, (cumm_error_SB_0 + cumm_error_SB_1)/(n_SB_0+n_SB_1)]
#     results = [dt.date(), dt.time(), 'Simple', data_file, 'Sampled from training', *rf_results, *sb_results]

#     from csv import writer
#     with open(results_file, 'a', newline='') as f_object:
#         writer_object = writer(f_object)
#         writer_object.writerow(results)
#         f_object.close()

#     print(f'{dt.date()}-{dt.time()} - Test results written to file: {results_file}')
    
# Print learning curve

def plot_learning_curve(x, rewards_history, loss_history, moving_avg_n, filename):
    fig = plt.figure()
    plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111, label='1')
    ax2 = fig.add_subplot(111, label='2', frame_on=False)
    
    ax.plot(x, rewards_history, color='C0')
    ax.set_xlabel('Training steps', color='C0')
    ax.set_ylabel('Rewards', color='C0')
    ax.tick_params(axis='x', color='C0')
    ax.tick_params(axis='y', color='C0')
    
    N = len(rewards_history)
    moving_avg = np.empty(N)
    for t in range(N):
        moving_avg[t] = np.mean(loss_history[max(0, t-moving_avg_n):(t+1)])
        
    ax2.scatter(x, moving_avg, color='C1', s=1)
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Loss', color='C1')
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors='C1')
    
    plt.savefig(filename)
        
def single_axes_plot(x, y, title='', x_label='', y_label='', xticks=0, threshold=0.0, filename='plot.png'):
    # Plot y
    fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi= 80)
    ax.plot(x, y, color='tab:blue', linewidth=2)

    # Decorations
    ax.set_xlabel(x_label, fontsize=16)
    ax.tick_params(axis='x', rotation=0, labelsize=10)
    ax.set_ylabel(y_label, color='tab:blue', fontsize=16)
    ax.tick_params(axis='y', rotation=0, labelcolor='tab:blue')
    ax.grid(alpha=.4)
    if threshold > 0.0:
        ax.axhline(y = threshold, color = 'r', linestyle = '--',  linewidth=1.0)
        ax.grid(alpha=.4)
    ax.set_xticks(np.arange(0, len(x), xticks))
    ax.set_xticklabels(x[::xticks], rotation=90, fontdict={'fontsize':10})
    ax.set_title(title, fontsize=18)
    fig.tight_layout()
    plt.savefig(filename)
    plt.show()

def two_variable_plot(x, y1, y2, title='', x_label='', y1_label='', y2_label='', xticks=0, filename='plot.png'):
    # Plot Line1 (Left Y Axis)
    fig, ax = plt.subplots(1,1,figsize=(10, 4), dpi= 80)
    ax.plot(x, y1, color='tab:green', alpha=0.7, linewidth=0.5)
    ax.plot(x, y2, color='tab:blue', linewidth=2.0)

    # Decorations
    ax.set_xlabel(x_label, fontsize=16)
    ax.tick_params(axis='x', rotation=0, labelsize=10)
    ax.set_ylabel(y2_label, color='tab:blue', fontsize=16)
    ax.tick_params(axis='y', rotation=0, labelcolor='tab:blue')
    ax.grid(alpha=.4)
    ax.set_xticks(np.arange(0, len(x), xticks))
    ax.set_xticklabels(x[::xticks], rotation=90, fontdict={'fontsize':10})
    ax.set_title(title, fontsize=18)
    ax.legend(['Rewards', 'Moving avg.'])
    
    fig.tight_layout()
    plt.savefig(filename)
    plt.show()  

def two_axes_plot(x, y1, y2, title='', x_label='', y1_label='', y2_label='', xticks=0,threshold=0.0):
    # Plot Line1 (Left Y Axis)
    fig, ax1 = plt.subplots(1,1,figsize=(10, 4), dpi= 80)
    ax1.plot(x, y1, color='tab:orange', linewidth=2)

    # Plot Line2 (Right Y Axis)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(x, y2, color='tab:blue', alpha=0.5)

    # Decorations
    # ax1 (left Y axis)
    ax1.set_xlabel(x_label, fontsize=16)
    ax1.tick_params(axis='x', rotation=0, labelsize=10)
    ax1.set_ylabel(y1_label, color='tab:red', fontsize=16)
    ax1.tick_params(axis='y', rotation=0, labelcolor='tab:red')
    ax1.grid(alpha=.4)
    if threshold > 0.0:
        ax1.axhline(y = threshold, color = 'r', linestyle = '--',  linewidth=1.0)
        ax1.grid(alpha=.4)

    # ax2 (right Y axis)
    ax2.set_ylabel(y2_label, color='tab:blue', fontsize=16)
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.set_xticks(np.arange(0, len(x), xticks))
    ax2.set_xticklabels(x[::xticks], rotation=90, fontdict={'fontsize':10})
    ax2.set_title(title, fontsize=18)
    fig.tight_layout()
    plt.show()
    
def plot_error_bounds(x, y):
    import seaborn as sns
    sns.set()

    # Compute standard error
    sem = np.std(y, ddof=1) / np.sqrt(np.size(y))
    sd = np.std(y)

    plt.figure(figsize=(9, 4))
    center_line = plt.plot(x, y, 'b-')
    fill = plt.fill_between(x, y-sd, y+sd, color='b', alpha=0.2)
    plt.margins(x=0)
    plt.legend(['Rewards'])
    plt.show()