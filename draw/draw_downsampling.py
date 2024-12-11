import matplotlib.pyplot as plt

plt.style.use("ggplot")

rate = [1, 2, 3, 4, 5]

downsampling = {}
downsampling['ours'] = [4.246, 7.328, 15.846, 34.14, 67.911]
downsampling['fedavg'] = [4.486, 7.48, 16.725, 46.781, 73.648]
downsampling['single'] = [7.328, 29.647, 43.832, 106.189, 148.673]
downsampling['heuristic'] = [727.189, 747.616, 827.632, 1067.717, 1277.494]

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
ax.set_xlabel('Downsampling Rate', fontsize=20)
ax.tick_params(axis='y', labelcolor='black')
ax.set_yscale('log', base=2)
plt.yticks(fontsize=20)
ax.grid(True)

ax.set_xticks(rate)
ax.set_xticklabels(["0.1", "0.2", "0.3", "0.4", "0.5"], fontsize=20)

legend = ax.legend(loc='best', fontsize=15)
legend.get_frame().set_alpha(0)

plt.tight_layout()

plt.savefig('./downsampling.png')
plt.show()
