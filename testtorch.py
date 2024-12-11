import pandas as pd
from tqdm import tqdm
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np

def find_most_dense_interval(data, window_size):
    # 先对数据进行排序
    sorted_data = np.sort(data)

    # 初始化变量来存储最密集区间的起点、终点以及区间内的数据个数
    max_count = 0
    best_interval_start = None
    best_interval_end = None

    # 滑动窗口遍历数据
    for i in range(len(sorted_data) - window_size + 1):
        interval_start = sorted_data[i]
        interval_end = sorted_data[i + window_size - 1]
        count = window_size

        if count > max_count:
            max_count = count
            best_interval_start = interval_start
            best_interval_end = interval_end

    return best_interval_start, best_interval_end, max_count

# 示例数据
data = np.array([1, 3, 5, 6, 7, 8, 9, 15, 18, 20])
window_size = 5

# 获取最密集区间
interval_start, interval_end, count = find_most_dense_interval(data, window_size)
print(f"最密集的区间是：[{interval_start}, {interval_end}]，包含的数据个数为：{count}")

def main():
    base_dir = f"/data/yanglinghua/trajCl/" + f"data/beijing/jksx/"
    csv_list = os.listdir(base_dir)
    print(csv_list)

    csv_name = csv_list[0]
    df = pd.read_csv(base_dir + csv_name, header=None)
    print(df.loc[0])
    # print(df.loc[2])
    # print(df.loc[3])

    x_min = 1e9 + 7
    x_max = 0
    y_min = 1e9 + 7
    y_max = 0
    csv_list = csv_list[11:12]
    xs = []
    ys = []
    for csv_name in tqdm(csv_list, desc='start solving', unit='files'):
        df = pd.read_csv(base_dir + csv_name, header=None)
        # print(df.info())
        for i in tqdm(range(len(df))):
            # if df.loc[i, 4] < 115.6:
            #     print('no')
            # # if df.loc[i, 4] > 117.3:
            # #     print('no')
            # # if df.loc[i, 4] < 115.5:
            # #     print('no')
            # # if df.loc[i, 4] < 115.5:
            # #     print('no')
            # x_min = min(x_min, df.loc[i, 4])
            # x_max = max(x_max, df.loc[i, 4])
            # y_min = min(y_min, df.loc[i, 5])
            # y_max = max(y_max, df.loc[i, 5])
            x, y = df.loc[i, 4], df.loc[i, 5]
            xs.append(x)
            ys.append(y)
    print(find_most_dense_interval(xs, 9000))
    print(find_most_dense_interval(ys, 9000))

    #         plt.scatter(x, y)
    # plt.show()
    print(x_min, x_max, y_min, y_max)

if __name__ == "__main__":
    main()
