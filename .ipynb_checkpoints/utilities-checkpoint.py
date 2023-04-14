import numpy as np
import matplotlib.pyplot as plt

# Print learning curve

def plot_learning_curve(x, rewards, epsilons, moving_avg_n, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, label='1')
    ax2 = fig.add_subplot(111, label='2', frame_on=False)
    
    ax.plot(x, epsilons, color='C0')
    ax.set_xlabel('Training steps', color='C0')
    ax.set_ylabel('Epsilon', color='C0')
    ax.tick_params(axis='x', color='C0')
    ax.tick_params(axis='y', color='C0')
    
    N = len(rewards)
    moving_avg = np.empty(N)
    for t in range(N):
        moving_avg[t] = np.mean(rewards[max(0, t-moving_avg_n):(t+1)])
        
    ax2.scatter(x, moving_avg, color='C1', s=1)
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Reward', color='C1')
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors='C1')
    
    plt.savefig(filename)
        
