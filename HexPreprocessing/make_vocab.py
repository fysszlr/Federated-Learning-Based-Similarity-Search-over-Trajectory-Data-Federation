import numpy as np
from rtree import index
import sys
import pandas
from hex_config import *

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
    ids = list(tree_index.nearest((lng, lat, lng, lat), 5))[:5]
    for one_id in ids:
        if judge_area(lng, lat, grids[one_id][1]):
            return one_id

    return -1


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


class Hex(object):
    def __init__(self, no, points):
        self.no = no
        self.points = points


if __name__ == "__main__":
    # 加载六边形
    grids = load_grids()
    # 加载索引树
    tree_index = index.Index('rtree')
    print(tree_index)
    f = open(data_name, 'r')
    df = pandas.read_csv(f)
    polylines = list(df.to_dict()['POLYLINE'].values())
    polylines = [eval(line) for line in polylines]
    not_in_region = 0
    not_in_grid = 0
    hot_cell_count = [0 for _ in range(200000)]
    for idx, polyline in enumerate(polylines):
        flag: bool = True
        for x, y in polyline:
            if not in_region(x, y):
                not_in_region += 1
                continue
            elems = discrete_location(x, y, grids)
            if elems == -1:
                not_in_grid += 1
            else:
                hot_cell_count[elems] += 1
        if idx % 5000 == 0:
            print(f"scaned {idx} trajs")

    print("not in grid: ", not_in_grid)
    print("not in region: ", not_in_region)

    hot_tree_index = index.Index('hot_tree')
    hot_cell_sorted = []
    for i in range(200000):
        if hot_cell_count[i] < min_freq:
            continue
        hot_cell_sorted.append((hot_cell_count[i], i))

    hot_cell_sorted = sorted(hot_cell_sorted, reverse=True)
    print(hot_cell_sorted)

    hot_cell2grid_idx = []
    for i, (count, grid) in enumerate(hot_cell_sorted):
        idx, points = grids[i]
        hexa = Hex(idx, points)
        bottom_point = points[0]
        hot_tree_index.insert(len(hot_cell2grid_idx),
                              (bottom_point[0], bottom_point[1], bottom_point[0], bottom_point[1]), obj=hexa)
        hot_cell2grid_idx.append(grid)

    import pickle

    with open('hot_cell_sorted', 'wb') as f:
        pickle.dump(hot_cell_sorted, f)
    with open('hot_cell2grid_idx', 'wb') as f:
        pickle.dump(hot_cell2grid_idx, f)
