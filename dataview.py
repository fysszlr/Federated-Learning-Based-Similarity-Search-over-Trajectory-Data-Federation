import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

data = pd.read_csv('/home/zlr/t2vec/data/porto.csv',nrows=800000)
rows, columns = data.shape
print(f"Number of rows and columns: {rows}, {columns}")
print(data.columns.values)
print(data.loc[100:105, 'POLYLINE'])
all_coordinates = []
for coords in tqdm(data['POLYLINE'], desc='Loading Coordinates', unit='rows'):
    all_coordinates.append(json.loads(coords))
# all_coordinates = np.array(all_coordinates)
threshold_length = 30
length = [len(subline) for subline in all_coordinates]
# all_coordinates = all_coordinates[length >= threshold_length]
tmp_coordinates = []
for i,subline in enumerate(all_coordinates):
    if length[i] >= threshold_length:
        tmp_coordinates.append(all_coordinates[i])
all_coordinates=tmp_coordinates
print(f'Number of vaild rows: {len(all_coordinates)}')
flattened_coordinates = [coord for line in all_coordinates for coord in line]
flattened_coordinates = np.array(flattened_coordinates)
# threshold_density = 5
# density = np.zeros(flattened_coordinates.shape[0])
# for i, point in tqdm(enumerate(flattened_coordinates), desc='Loading Points', unit='points'):
#     density[i] = np.sum(np.linalg.norm(flattened_coordinates - point, axis=1) < 0.1)
# filtered_coordinates = flattened_coordinates[density >= threshold_density]
plt.scatter(flattened_coordinates[:, 0], flattened_coordinates[:, 1], s=0.01)
# x_tick = np.arange(-9.3, -7.5, 0.2)
# y_tick = np.arange(38.5, 42.5, 0.25)
# plt.xticks(x_tick)
# plt.yticks(y_tick)
plt.show()

