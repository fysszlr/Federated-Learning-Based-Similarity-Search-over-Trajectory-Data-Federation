import pickle
import numpy as np
import matplotlib.pyplot as plt

number_k = 40

traj = []
with open(f'../data/traj0drop_ratio0.pkl', 'rb') as file:
    traj = pickle.load(file)

with open('../data/traj0_origin_length', 'r') as file:
    traj_length = np.fromstring(file.read(), dtype=float, sep='\n')

with open(f'../data/total_knn_tagsNumber_k{number_k}drop_ratio0.pkl', 'rb') as file:
    total_knn_tags = pickle.load(file)

drop_knn_tags = [[] for i in range(7)]
for i in range(2, 3):
    drop_ratio = i / 10
    with open(f'../data/total_knn_tagsNumber_k{number_k}drop_ratio0.{i}.pkl', 'rb') as file:
        drop_knn_tags[i] = pickle.load(file)

# traj = []
# with open(f'../data/traj0distort_ratio0.pkl', 'rb') as file:
#     traj = pickle.load(file)
#
# with open('../data/traj0_origin_length', 'r') as file:
#     traj_length = np.fromstring(file.read(), dtype=float, sep='\n')
#
# with open(f'../data/total_knn_tagsNumber_k{number_k}distort_ratio0.pkl', 'rb') as file:
#     total_knn_tags = pickle.load(file)
#
# drop_knn_tags = [[] for i in range(7)]
# for i in range(2, 3):
#     distort_ratio = i / 10
#     with open(f'../data/total_knn_tagsNumber_k{number_k}distort_ratio0.{i}.pkl', 'rb') as file:
#         drop_knn_tags[i] = pickle.load(file)

print(len(total_knn_tags))

feature_trajs = []
x = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
x_pos = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
x_length = ['20-29', '30-39', '40-49', '50+']
marker_size = 15
font_size = 30
legend_size = 30
figure_width = 30
figure_height = 18
bar_width = 0.15
rate = np.zeros(5)
for i in range(2, 3):
    rate_pos = np.zeros(10)
    rate_length = np.zeros((6, 10))
    for j in range(1000):
        rate_now = 0
        feature_trajs.append(traj[j])
        for item in drop_knn_tags[i][j]:
            flag = 0
            for point in total_knn_tags[j]:
                if point == item:
                    rate_now += 1
                    flag = 1
                    break

            # if flag==1 and i == 2:
            #     print(drop_knn_tags[i][j], total_knn_tags[j])
            rate[i - 2] += flag
        rate_now = int(rate_now * 10 / number_k)
        rate_pos[rate_now] += 1
        length = traj_length[j]
        for idx, tmp in enumerate(range(50, 10, -10)):
            if length >= tmp:
                if length < 20:
                    print(length)
                rate_length[3 - idx][rate_now] += 1
                break
    # plt.close("all")
    # x_tmp = np.arange(len(x_pos))
    # w = bar_width
    # fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    # ax.bar(x_tmp - 1.5 * w, rate_length[0], color='#E24A33', width=w, label=x_length[0], edgecolor='k',
    #        linewidth=1)
    # ax.bar(x_tmp - 0.5 * w, rate_length[1], color='b', width=w, label=x_length[1], edgecolor='k', linewidth=1)
    # ax.bar(x_tmp + 0.5 * w, rate_length[2], color='#8EBA42', width=w, label=x_length[2], edgecolor='k', linewidth=1)
    # ax.bar(x_tmp + 1.5 * w, rate_length[3], color='brown', width=w, label=x_length[3], edgecolor='k',
    #        linewidth=1)
    #
    # # ax.set_yscale('log')
    # ax.set_ylabel('Num', fontsize=font_size)
    # ax.set_xlabel('acc', fontsize=font_size)
    # plt.xticks(np.arange(len(x_pos)), labels=x_pos, fontsize=font_size)
    # plt.xticks(fontsize=font_size)
    # plt.yticks(fontsize=font_size)
    # plt.title(f'k={number_k} drop_ratio={i / 10}',fontsize=font_size)
    # plt.grid()
    # plt.legend(ncol=2, loc='best', fontsize=legend_size)
    # plt.tight_layout()
    # plt.show()
    rate[i - 2] /= (1000 * number_k)
print(rate)
# plt.cla()
# plt.plot(x, rate)
# plt.show()

# with open('/home/zlr/t2vec/experimentData/feature_trajs','w') as file:
#     for line in feature_trajs:
#         for point in line:
#             file.write(str(point) + ' ')
#         file.write('\n')