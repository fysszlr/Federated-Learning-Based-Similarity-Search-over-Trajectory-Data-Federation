import copy
import pickle
import numpy as np
import random

def drop_items_with_ratio(input_list, drop_ratio):
    # 计算要删除的项的数量
    num_items_to_drop = int(len(input_list) * drop_ratio)

    # 随机选择要删除的项的索引
    items_to_drop_indices = random.sample(range(len(input_list)), num_items_to_drop)

    # 删除选定的项
    modified_list = [item for index, item in enumerate(input_list) if index not in items_to_drop_indices]

    return modified_list

def distort(t, ratio):
    new_t = copy.deepcopy(t)
    for i, it in enumerate(new_t):
        if random.random() <= ratio:
            noise = random.gauss(0,1)
            noise = 1000 * noise
            new_t[i] = abs(int(it + noise))
    return new_t

for id in range(6):
    traj_drop = []
    traj_distort = []
    with open(f'../data/traj{id}drop_ratio0.pkl','rb') as file:
        traj0 = pickle.load(file)
        for t in traj0:
            t_drop = drop_items_with_ratio(t, 0.2)
            t_distort = distort(t,0.2)
            print(t)
            print(t_distort)
            traj_drop.append(t_drop)
            traj_distort.append(t_distort)


    with open(f'../data/traj{id}drop_ratio0.2.pkl','wb') as file:
        pickle.dump(traj_drop,file)
    with open(f'../data/traj{id}distort_ratio0.2.pkl','wb')as file:
        pickle.dump(traj_distort,file)