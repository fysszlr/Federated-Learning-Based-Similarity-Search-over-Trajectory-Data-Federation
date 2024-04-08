from math import *
import numpy as np
from hex_config import *


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


centres = []
print(lonlat2meters(minlon, minlat))
print(lonlat2meters(maxlon, maxlat))
meter_minlon, meter_minlat = lonlat2meters(minlon, minlat)
meter_maxlon, meter_maxlat = lonlat2meters(maxlon, maxlat)
meter_minlon -= a * 2
meter_minlat -= a * 2
meter_maxlon += a * 2
meter_maxlat += a * 2

# meter_lons = np.arange(meter_minlon, meter_maxlon, a * 2)
# meter_lats = np.arange(meter_minlat, meter_maxlat, a * sqrt3)
# for meter_lon in meter_lons:
#     for meter_lat in meter_lats:
#         centres.append((meter_lon, meter_lat))
#         centres.append((meter_lon + a, meter_lat + a * sqrt3 / 2))
idx_lon = 0
while idx_lon * a + meter_minlon <= meter_maxlon:
    idx_lat = 0
    while idx_lat * a * sqrt3 /2 + meter_minlat <= meter_maxlat:
        if (idx_lat + idx_lon) % 2 == 0:
            centres.append((idx_lon, idx_lat))
        idx_lat += 1
    idx_lon += 1

x_num = [-1 / 2, 1 / 2, 1, 1 / 2, -1 / 2, -1]
y_num = [-1, -1, 0, 1, 1, 0]

grid_data = []
for idx, (x, y) in enumerate(centres):
    line = [0 for _ in range(13)]
    line[0] = idx
    for i in range(6):
        x_now = (x + x_num[i]) * a
        y_now = (y + y_num[i]) * a * sqrt3 / 2
        lon_now, lat_now = meters2lonlat(x_now + meter_minlon, y_now + meter_minlat)
        line[i * 2 + 1] = lon_now
        line[i * 2 + 2] = lat_now
    grid_data.append(line)

print(len(grid_data))

import csv

csv_file = 'hexagon_grid_table.csv'

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    for row in grid_data:
        writer.writerow(row)

# 197945
