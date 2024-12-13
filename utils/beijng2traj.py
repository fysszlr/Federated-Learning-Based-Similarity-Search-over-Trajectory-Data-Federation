import pandas as pd
from tqdm import tqdm
from datetime import datetime
import os


def main(name, files):
    base_dir = f"/data/yanglinghua/trajCl/" + f"data/beijing/{name}/"
    csv_list = os.listdir(base_dir)
    print(csv_list)

    time_format = "%Y-%m-%d %H:%M:%S"

    data = {}

    keys = [
        'TRIP_ID',
        'CALL_TYPE',
        'ORIGIN_CALL',
        'ORIGIN_STAND',
        'TAXI_ID',
        'TIMESTAMP',
        'DAY_TYPE',
        'MISSING_DATA'
    ]

    for key in keys:
        data[key] = []
    data['POLYLINE'] = []

    def add_poly_line(trip):
        for key in keys:
            data[key].append([])
        data['POLYLINE'].append(trip)

    csv_list = csv_list[5:5 + int(files)]
    for csv_name in tqdm(csv_list, desc='start solving', unit='files'):
        df = pd.read_csv(base_dir + csv_name, header=None)

        # print(df.info())
        # print(df.shape)
        # print(df.head(1).to_string())

        # for i in range(1, len(df)):
        #     first = df.loc[0]
        #     last = df.loc[i - 1]
        #     line = df.loc[i]
        #     now = datetime.strptime(line[3], time_format)
        #     last_time = datetime.strptime(last[3], time_format)
        #     first_time = datetime.strptime(first[3], time_format)
        #     delta = now - last_time
        #     total = now - first_time
        #     # print(delta.total_seconds())
        #     print(total.total_seconds())

        # times = df.loc[:, 3]
        # times = [datetime.strptime(time, time_format) for time in times]
        # print(min(times))
        # print(max(times))

        used = [False for _ in range(len(df))]
        for i in tqdm(range(len(df)), desc='reading file', unit='logs'):
            if used[i]:
                continue
            used[i] = False
            # line = df.loc[i]
            id = df.loc[i, 1]
            time = datetime.strptime(df.loc[i, 3], time_format)
            # start_time = time
            trip = []
            trip.append([df.loc[i, 4], df.loc[i, 5]])
            for j in range(i + 1, len(df)):
                if used[j]:
                    continue
                # now = df.loc[j]
                id_now = df.loc[j, 1]
                time_now = datetime.strptime(df.loc[j, 3], time_format)
                time_difference = time_now - time
                # total_time = time_now - start_time
                if time_difference.total_seconds() > 360:
                    break
                if id != id_now:
                    continue

                trip.append([df.loc[j, 4], df.loc[j, 5]])
                used[j] = True
                time = time_now
                # print(time_difference.total_seconds())
            if len(trip) >= 5:
                add_poly_line(trip)

    new_df = pd.DataFrame(data)

    new_df.to_csv(base_dir + f'../{name}.csv', index=False)


if __name__ == "__main__":
    import sys

    main(sys.argv[1], sys.argv[2])
