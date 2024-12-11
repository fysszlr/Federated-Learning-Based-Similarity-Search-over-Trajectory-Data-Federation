import matplotlib.pyplot as plt

plt.style.use("ggplot")

rate = [5, 10, 20, 50]

downsampling = {}
downsampling['ours'] = [0.1514, 0.2914, 0.41255, 0.50758]
downsampling['fedavg'] = [0.1522, 0.2858, 0.40585, 0.49162]
downsampling['single'] = [0.1158, 0.2383, 0.3378, 0.3809]

fig, ax = plt.subplots(figsize=(6, 5))

ax.plot(rate, downsampling['ours'], lw=2, color='r', marker='o', markersize=18, markerfacecolor='none',
        markeredgewidth=2, label='Ours', linestyle="-.")
ax.plot(rate, downsampling['fedavg'], lw=2, color='g', marker='s', markersize=18, markerfacecolor='none',
        markeredgewidth=2, label='FedAvg', linestyle="-.")
ax.plot(rate, downsampling['single'], lw=2, color='y', marker='D', markersize=18, markerfacecolor='none',
        markeredgewidth=2, label='Local', linestyle="-")

ax.set_ylabel('Hit Ratio HR@k', fontsize=20)
ax.set_xlabel('k', fontsize=20)
ax.tick_params(axis='y', labelcolor='black')
plt.yticks(fontsize=20)
ax.grid(True)

ax.set_xticks(rate)
ax.set_xticklabels(["5", "10", "20", "50"], fontsize=20)

legend = ax.legend(loc='best', fontsize=15)
legend.get_frame().set_alpha(0)

plt.tight_layout()

plt.savefig('./knn_query.png')
plt.show()
