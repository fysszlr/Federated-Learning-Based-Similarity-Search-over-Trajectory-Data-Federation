import copy
import random

import numpy as np
from rtree import index
import sys
import pandas
from hex_config import *
from math import *
import pickle

data_name = "porto.csv"
grid_name = "hexagon_grid_table.csv"


def pnpoly(testx, testy, boundary):
    nvert = boundary.shape[0]
    c = 0
    i = 0
    j = nvert - 1
    vertx = boundary[:, 0]
    verty = boundary[:, 1]
    while i < nvert:
        if (((verty[i] > testy) != (verty[j] > testy)) and
                (testx < (vertx[j] - vertx[i]) * (testy - verty[i]) / (verty[j] - verty[i]) + vertx[i])):
            c = 1 ^ c
        j = i
        i = i + 1
    return c


def judge_area(lng, lat, boundary, fuzzy=False):
    boundary = np.array(boundary)
    [lng_max, lat_max] = np.amax(boundary, axis=0)
    [lng_min, lat_min] = np.amin(boundary, axis=0)
    if lng < lng_min or lng > lng_max or lat < lat_min or lat > lat_max:
        return False
    if fuzzy:
        return True
    else:
        c = pnpoly(lng, lat, boundary)
        if c == 1:
            return True
        else:
            return False


def discrete_location(lng, lat, grids):
    ids = list(hot_tree_index.nearest((lng, lat, lng, lat), 5))[:5]
    for one_id in ids:
        if judge_area(lng, lat, grids[hot_cell2grid_idx[one_id]][1]):
            return one_id

    return ids[0]


def load_grids():
    grids = []
    f = open(grid_name, "r")
    line = f.readline()
    while line:
        grid = [len(grids)]
        points = line.split(',')[1:]
        grid.append([[float(points[i]), float(points[i + 1])] for i in range(0, len(points), 2)])
        grids.append(grid)
        line = f.readline()
    f.close()
    print("load finished, %d grids found" % len(grids))
    return grids


def in_region(lon, lat):
    return minlon <= lon <= maxlon and minlat <= lat <= maxlat


def lonlat2meters(lon, lat):
    semimajoraxis = 6378137.0
    east = lon * 0.017453292519943295
    north = lat * 0.017453292519943295
    t = sin(north)
    return semimajoraxis * east, 3189068.5 * log((1 + t) / (1 - t))


def meters2lonlat(x, y):
    semimajoraxis = 6378137.0
    lon = x / semimajoraxis / 0.017453292519943295
    t = exp(y / 3189068.5)
    lat = asin((t - 1) / (t + 1)) / 0.017453292519943295
    return lon, lat


def downsampling(trip, dropping_rate):
    num_items_to_drop = int(len(trip) * dropping_rate)
    items_to_drop_indices = random.sample(range(len(trip)), num_items_to_drop)
    modified_list = [item for index, item in enumerate(trip) if index not in items_to_drop_indices]
    return modified_list


def distort(t, ratio):
    new_t = copy.deepcopy(t)
    for i in range(len(t)):
        if random.random() <= ratio:
            x, y = lonlat2meters(t[i][0], t[i][1])
            xnoise, ynoise = random.gauss(0,1), random.gauss(0,1)
            new_t[i][0], new_t[i][1] = meters2lonlat(x + xnoise * 50, y + ynoise * 50)
    return new_t

def downsampling_distort(trip):
    noise_trips = []
    dropping_rates = [0, 0.2, 0.4, 0.5, 0.6]
    distorting_rates = [0, 0.2, 0.4, 0.6]
    for dropping_rate in dropping_rates:
        noise_trip1 = downsampling(trip, dropping_rate)
        for distorting_rate in distorting_rates:
            noise_trip2 = distort(noise_trip1, distorting_rate)
            noise_trips.append(noise_trip2)
    return noise_trips

if __name__ == "__main__":
    train_length = int(sys.argv[1])
    val_length = int(sys.argv[2])
    # 加载热区对应表
    with open('hot_cell2grid_idx', 'rb') as file:
        hot_cell2grid_idx = pickle.load(file)
    print(max(hot_cell2grid_idx))
    # 加载六边形
    grids = load_grids()
    # 加载索引树
    hot_tree_index = index.Index('hot_tree')
    print(hot_tree_index)
    f = open(data_name, 'r')
    df = pandas.read_csv(f, nrows=train_length + val_length)
    polylines = list(df.to_dict()['POLYLINE'].values())
    polylines = [eval(line) for line in polylines]

    trainsrc = open(f"{data_path}/train.src", "w")
    traintrg = open(f"{data_path}/train.trg", "w")
    trainmta = open(f"{data_path}/train.mta", "w")

    validsrc = open(f"{data_path}/val.src", "w")
    validtrg = open(f"{data_path}/val.trg", "w")
    validmta = open(f"{data_path}/val.mta", "w")

    for idx, polyline in enumerate(polylines):
        if idx < train_length:
            src, mta, trg = trainsrc, trainmta, traintrg
        else:
            src, mta, trg = validsrc, validmta, validtrg
        if not (min_length <= len(polyline) <= max_length):
            continue
        # src
        noise_trips = downsampling_distort(polyline)
        # mta
        min_x, min_y = 1e9, 1e9
        max_x, max_y = -1e9, -1e9
        for (x, y) in polyline:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        centre_x = min_x + (max_x - min_x) / 2
        centre_y = min_y + (max_y - min_y) / 2
        offset_x, offset_y = lonlat2meters(centre_x, centre_y)
        offset_x = (offset_x - meter_minlon) / a
        offset_y = (offset_y - meter_minlat) / (a * sqrt3 / 2)
        # trg
        seq = [0 for _ in range(len(polyline))]
        for i, (x, y) in enumerate(polyline):
            elem = discrete_location(x, y, grids)
            seq[i] = elem
        #write
        for noise_trip in noise_trips:
            noise_seq = [0 for _ in range(len(noise_trip))]
            for i, (x, y) in enumerate(noise_trip):
                elem = discrete_location(x, y, grids)
                noise_seq[i] = elem
            for elem in noise_seq:
                src.write(str(elem) + ' ')
            src.write('\n')
            mta.write(str(offset_x) + " " + str(offset_y) + "\n")
            for elem in seq:
                trg.write(str(elem) + ' ')
            trg.write('\n')
        #out
        if idx % 5000 == 0:
            print(f"scaned {idx} trajs")
