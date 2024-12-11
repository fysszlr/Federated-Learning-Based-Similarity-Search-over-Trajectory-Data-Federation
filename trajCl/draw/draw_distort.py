import matplotlib.pyplot as plt

plt.style.use("ggplot")

rate = [1, 2, 3]

downsampling = {}
downsampling['ours'] = [3.42, 4.685, 3.454]
downsampling['fedavg'] = [3.538, 5.657, 4.3]
downsampling['single'] = [7.246, 7.818, 10.124]
downsampling['heuristic'] = [737.074, 757.018, 737.103]

fig, ax = plt.subplots(figsize=(6, 5))

ax.plot(rate, downsampling['ours'], lw=2, color='r', marker='o', markersize=18, markerfacecolor='none',
        markeredgewidth=2, label='Ours', linestyle="-.")
ax.plot(rate, downsampling['fedavg'], lw=2, color='g', marker='s', markersize=18, markerfacecolor='none',
        markeredgewidth=2, label='FedAvg', linestyle="-.")
ax.plot(rate, downsampling['single'], lw=2, color='y', marker='D', markersize=18, markerfacecolor='none',
        markeredgewidth=2, label='Local', linestyle="-")
ax.plot(rate, downsampling['heuristic'], lw=2, color='b', marker='^', markersize=18, markerfacecolor='none',
        markeredgewidth=2, label='Lcss', linestyle="-.")

ax.set_ylabel('Mean Rank', fontsize=20)
ax.set_xlabel('Distorting Rate', fontsize=20)
ax.tick_params(axis='y', labelcolor='black')
ax.set_yscale('log', base=2)
plt.yticks(fontsize=20)
ax.grid(True)

ax.set_xticks(rate)
ax.set_xticklabels(["0.1", "0.2", "0.3"], fontsize=20)

legend = ax.legend(loc='best', fontsize=15)
legend.get_frame().set_alpha(0)

plt.tight_layout()

plt.savefig('./distort.png')
plt.show()
