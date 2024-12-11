from config import *
import pickle
import argparse
import torch
import pandas as pd
from utils.data_loader import read_traj_dataset

def parse_args():
    # dont set default value here! -- it will incorrectly overwrite the values in config.py.
    # config.py is the correct place for default values.

    parser = argparse.ArgumentParser(description="TrajCL/train.py")
    parser.add_argument('--dumpfile_uniqueid', type=str, help='see config.py')
    parser.add_argument('--seed', type=int, help='')
    parser.add_argument('--dataset', type=str, help='')
    parser.add_argument('--moon_loss_weight', type=float, help='')

    args = parser.parse_args()
    return dict(filter(lambda kv: kv[1] is not None, vars(args).items()))

Config.update(parse_args())

train_dataset, _, test_dataset = read_traj_dataset(Config.dataset_file)
print(train_dataset.data['merc_seq'])
print(train_dataset.data['merc_seq'].values)

# with open(Config.dataset_cell_file, 'rb') as f:
#     print(pickle.load(f))

# with open(Config.dataset_embs_file, 'rb')as f:
#     tensor = pickle.load(f)
#     print(tensor[0])
#     print(tensor.shape)

# with open(Config.dataset_file,  'rb') as f:
#     df = pd.read_pickle(f)
#     print(df)

# with open(Config.dataset_file + '_newsimi_raw.pkl', 'rb') as f:
#     print(pickle.load(f))