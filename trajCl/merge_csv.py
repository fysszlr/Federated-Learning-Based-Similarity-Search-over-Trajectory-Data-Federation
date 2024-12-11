# import pandas as pd
#
# companys = ['jksx', 'qh', 'jyj', 'zhtc', 'ytax']
# file_paths = [f"/data/yanglinghua/trajCl/" + f"data/beijing/{name}.csv" for name in companys]
#
# dfs = []
#
# for i, file_path in enumerate(file_paths):
#     # 读取CSV文件
#     df = pd.read_csv(file_path)
#
#     df['dataset_id'] = companys[i]
#
#     dfs.append(df)
#
# # 将所有DataFrame合并为一个
# combined_df = pd.concat(dfs, ignore_index=True)
#
# # 打乱数据
# shuffled_df = combined_df.sample(frac=1).reset_index(drop=True)
#
# shuffled_df.to_csv(f"/data/yanglinghua/trajCl/" + f"data/beijing/beijing.csv", index=False)
#
# print(shuffled_df.head())


import pandas as pd

# 读取CSV文件
df = pd.read_csv(f"/data/yanglinghua/trajCl/data/beijing/ytax.csv")

# 设置要检查的列名和过滤条件
min_lon = 115.3676
min_lat = 38.454923
max_lon = 116.73552
max_lat = 40.14955

print(df.info())

new_csv = []
import ast
from tqdm import tqdm
for index, row in tqdm(df.iterrows()):
    flag = False
    for it in ast.literal_eval(row['POLYLINE']):
        if it[0] < min_lon or it[0] > max_lon:
            flag = True
            break
        if it[1] < min_lat or it[1] > max_lat:
            flag = True
            break
    if not flag:
        new_csv.append(index)

print(len(new_csv))
new_csv = df.loc[new_csv]

new_csv.to_csv(f"/data/yanglinghua/trajCl/data/beijing/beijing2.csv", index=False)
print(new_csv.head())