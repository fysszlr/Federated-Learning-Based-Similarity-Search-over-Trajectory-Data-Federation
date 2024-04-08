import numpy as np
import pandas as pd
import os
import sys
import time, datetime
import csv
import pickle

try:
    grids_hash2idx = pickle.load(open("grids_hash2idx", "rb"))
    grids_idx2hash = pickle.load(open("grids_idx2hash", "rb"))
except:
    grid_name = "hexagon_grid_table.csv"
    grids_hash2idx = dict()
    grids_idx2hash = []
    f = open(grid_name, "r")
    line = f.readline()
    while line:
        hash_code = line.split(',')[0]
        grids_hash2idx[hash_code] = len(grids_idx2hash)
        grids_idx2hash.append(hash_code)
        line = f.readline()
    f.close()
    pickle.dump(grids_hash2idx, open("grids_hash2idx", "wb"))
    pickle.dump(grids_idx2hash, open("grids_idx2hash", "wb"))
