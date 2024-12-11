import sys

sys.path.append('..')
import numpy as np
import random
import math

from config import Config
from utils import tool_funcs
from utils.rdp import rdp
from utils.cellspace import CellSpace
from utils.tool_funcs import truncated_rand


def straight(src):
    return src


def simplify(src):
    # src: [[lon, lat], [lon, lat], ...]
    return rdp(src, epsilon=Config.traj_simp_dist)


def shift(src):
    return [[p[0] + truncated_rand(), p[1] + truncated_rand()] for p in src]


def mask(src):
    l = len(src)
    arr = np.array(src)
    mask_idx = np.random.choice(l, int(l * Config.traj_mask_ratio), replace=False)
    return np.delete(arr, mask_idx, 0).tolist()


def subset(src):
    l = len(src)
    max_start_idx = l - int(l * Config.traj_subset_ratio)
    start_idx = random.randint(0, max_start_idx)
    end_idx = start_idx + int(l * Config.traj_subset_ratio)
    return src[start_idx: end_idx]


def translate(src):
    rand = np.random.uniform(0, 2 * np.pi)
    dist = np.random.uniform(0, Config.cell_size)
    x = float(np.cos(rand) * dist)
    y = float(np.sin(rand) * dist)
    return [[p[0] + x, p[1] + y] for p in src]


def time_shift(src):
    return [[src[idx][0] * 3 / 4 + src[idx + 1][0] * 1 / 4, src[idx][1] * 3 / 4 + src[idx + 1][1] * 1 / 4] for idx in
            range(len(src) - 1)]


def scaling(src):
    l = len(src)
    x = src[0][0] * 1 / 2 + src[l - 1][0] * 1 / 2
    y = src[0][1] * 1 / 2 + src[l - 1][1] * 1 / 2
    scaling_rate = np.random.uniform(0.7, 1)
    return [[p[0] * scaling_rate + x * (1 - scaling_rate), p[1] * scaling_rate + y * (1 - scaling_rate)] for p in src]


def down_sampling(src):
    def get_angle(v1, v2):
        v1 = np.array(v1)
        v2 = np.array(v2)
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return np.pi
        cos_theta = dot_product / (norm_v1 * norm_v2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle = np.arccos(cos_theta)
        return angle

    l = len(src)
    new_src = []
    for i in range(l):
        if i != 0 and i != l - 1:
            v1 = [src[i][0] - src[i - 1][0], src[i][1] - src[i - 1][1]]
            v2 = [src[i][0] - src[i + 1][0], src[i][1] - src[i + 1][1]]
            if -1 / 2 * np.pi <= get_angle(v1, v2) <= 1 / 2 * np.pi:
                continue
        if np.random.rand() < 0.7:
            new_src.append(src[i])
    return new_src


def splicing(src):
    l = len(src)
    new_src_head = []
    for i in range(int(l * 0.1)):
        if not new_src_head:
            new_src_head.append([src[0][0] + truncated_rand(), src[0][1] + truncated_rand()])
        else:
            new_src_head.append([new_src_head[i - 1][0] + truncated_rand(), new_src_head[i - 1][1] + truncated_rand()])
    new_src_head.reverse()
    new_src_tail = []
    for i in range(int(l * 0.1)):
        if not new_src_tail:
            new_src_tail.append([src[l - 1][0] + truncated_rand(), src[l - 1][1] + truncated_rand()])
        else:
            new_src_tail.append([new_src_tail[i - 1][0] + truncated_rand(), new_src_tail[i - 1][1] + truncated_rand()])
    return new_src_head + src + new_src_tail


def get_aug_fn(name: str):
    return {'straight': straight, 'simplify': simplify, 'shift': shift,
            'mask': mask, 'subset': subset,
            'translate': translate, 'time_shift': time_shift,
            'scaling': scaling, 'down_sampling': down_sampling,
            'splicing': splicing}.get(name, None)


# pair-wise conversion -- structural features and spatial feasures
def merc2cell2(src, cs: CellSpace):
    # convert and remove consecutive duplicates
    tgt = [(cs.get_cellid_by_point(*p), p) for p in src]
    tgt = [v for i, v in enumerate(tgt) if i == 0 or v[0] != tgt[i - 1][0]]
    tgt, tgt_p = zip(*tgt)
    return tgt, tgt_p


def generate_spatial_features(src, cs: CellSpace):
    # src = [length, 2]
    tgt = []
    lens = []
    for p1, p2 in tool_funcs.pairwise(src):
        lens.append(tool_funcs.l2_distance(p1[0], p1[1], p2[0], p2[1]))

    for i in range(1, len(src) - 1):
        dist = (lens[i - 1] + lens[i]) / 2
        dist = dist / (Config.trajcl_local_mask_sidelen / 1.414)  # float_ceil(sqrt(2))

        radian = math.pi - math.atan2(src[i - 1][0] - src[i][0], src[i - 1][1] - src[i][1]) \
                 + math.atan2(src[i + 1][0] - src[i][0], src[i + 1][1] - src[i][1])
        radian = 1 - abs(radian) / math.pi

        x = (src[i][0] - cs.x_min) / (cs.x_max - cs.x_min)
        y = (src[i][1] - cs.y_min) / (cs.y_max - cs.y_min)
        tgt.append([x, y, dist, radian])

    x = (src[0][0] - cs.x_min) / (cs.x_max - cs.x_min)
    y = (src[0][1] - cs.y_min) / (cs.y_max - cs.y_min)
    tgt.insert(0, [x, y, 0.0, 0.0])

    x = (src[-1][0] - cs.x_min) / (cs.x_max - cs.x_min)
    y = (src[-1][1] - cs.y_min) / (cs.y_max - cs.y_min)
    tgt.append([x, y, 0.0, 0.0])
    # tgt = [length, 4]
    return tgt


def traj_len(src):
    length = 0.0
    for p1, p2 in tool_funcs.pairwise(src):
        length += tool_funcs.l2_distance(p1[0], p1[1], p2[0], p2[1])
    return length
