import pickle
import numpy as np
import random
import pandas as pd


number_k = 40
writer = pd.ExcelWriter('data.xlsx', engine='xlsxwriter')
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

print(len(total_knn_tags))
x = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
x_pos = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
x_length = ['20-29', '30-39', '40-49', '50+']
rate = np.zeros(5)
rate_record = np.zeros(1000)

for i in range(2, 3):
    rate_pos = np.zeros(10)
    rate_length = np.zeros((6, 10))
    for j in range(1000):
        rate_now = 0
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
        rate_record[j] = rate_now / number_k
        rate_now = int(rate_now * 10 / number_k)
        rate_pos[rate_now] += 1
        length = traj_length[j]
        for idx, tmp in enumerate(range(50, 10, -10)):
            if length >= tmp:
                if length < 20:
                    print(length)
                rate_length[3 - idx][rate_now] += 1
                break
    rate[i - 2] /= (1000 * number_k)
    data = {
        '输入':traj,
        '长度':traj_length,
        'K':[number_k for _ in range(1000)],
        '抽样率':[i/10 for _ in range(1000)],
        'acc': rate_record,
        '输出':drop_knn_tags[i]
    }
    df = pd.DataFrame(data)
    df.to_excel(writer,sheet_name=f'drop',index=False)





with open(f'../data/traj0distort_ratio0.pkl', 'rb') as file:
    traj = pickle.load(file)

with open('../data/traj0_origin_length', 'r') as file:
    traj_length = np.fromstring(file.read(), dtype=float, sep='\n')

with open(f'../data/total_knn_tagsNumber_k{number_k}distort_ratio0.pkl', 'rb') as file:
    total_knn_tags = pickle.load(file)

drop_knn_tags = [[] for i in range(7)]
for i in range(2, 3):
    drop_ratio = i / 10
    with open(f'../data/total_knn_tagsNumber_k{number_k}distort_ratio0.{i}.pkl', 'rb') as file:
        drop_knn_tags[i] = pickle.load(file)

print(len(total_knn_tags))
x = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
x_pos = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
x_length = ['20-29', '30-39', '40-49', '50+']
rate = np.zeros(5)
rate_record = np.zeros(1000)

for i in range(2, 3):
    rate_pos = np.zeros(10)
    rate_length = np.zeros((6, 10))
    for j in range(1000):
        rate_now = 0
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
        rate_record[j] = rate_now / number_k
        rate_now = int(rate_now * 10 / number_k)
        rate_pos[rate_now] += 1
        length = traj_length[j]
        for idx, tmp in enumerate(range(50, 10, -10)):
            if length >= tmp:
                if length < 20:
                    print(length)
                rate_length[3 - idx][rate_now] += 1
                break
    rate[i - 2] /= (1000 * number_k)
    data = {
        '输入':traj,
        '长度':traj_length,
        'K':[number_k for _ in range(1000)],
        '抽样率':[i/10 for _ in range(1000)],
        'acc': rate_record,
        '输出':drop_knn_tags[i]
    }
    df = pd.DataFrame(data)
    df.to_excel(writer,sheet_name=f'distort',index=False)

writer.close()
print(rate)