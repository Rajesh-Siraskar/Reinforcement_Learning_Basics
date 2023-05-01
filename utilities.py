import numpy as np
import matplotlib.pyplot as plt

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
    