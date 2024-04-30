import itertools
import random
import numpy as np
from numba import jit

# https://danceswithcode.net/engineeringnotes/rotations_in_3d/rotations_in_3d_part1.html
@jit(nopython=True)
def xyz_matrix(x, y, z):
    c1 = np.cos(x * np.pi / 180)
    c2 = np.cos(y * np.pi / 180)
    c3 = np.cos(z * np.pi / 180)
    s1 = np.sin(x * np.pi / 180)
    s2 = np.sin(y * np.pi / 180)
    s3 = np.sin(z * np.pi / 180)
    return np.array([
        [c2 * c3, s1 * s2 * c3 - c1 * s3, c1 * s2 * c3 + s1 * s3],
        [c2 * s3, s1 * s2 * s3 + c1 * c3, c1 * s2 * s3 - s1 * c3],
        [-s2, s1 * c2, c1 * c2]
    ])

@jit(nopython=True)
def mat_to_euler(mat):
    w = np.arctan2(mat[1,0], mat[0,0])
    v = -np.arcsin(mat[2,0])
    u = np.arctan2(mat[2,1], mat[2,2])
    return [u * 180 / np.pi, v * 180 / np.pi, w * 180 / np.pi]

def normalize_angle(rot):
    return [r % 360. for r in rot]

def constraints_penalty(x, constraints):
    penalty = 0.
    for c in constraints:
        cfun = c['fun']
        args = c['args']
        if c['type'] == 'ineq':
            if cfun(x, *args) < 0:
                # print("negative ineq")
                # print(c, cfun(x, *args) ** 2)
                penalty += cfun(x, *args) ** 2
        elif c['type'] == 'eq':
            penalty += cfun(x, *args) ** 2
            # centity1, centity1fid, centity2, centity2fid, _ = args[0]
            # print(centity1, centity1fid, centity2, centity2fid, cfun(x, *args) ** 2)
            # this could be exponential, so that the constraint can be satisfied
    return penalty

def reservoir_sample(it, k):
    if k < 0:
        raise ValueError("sample size must be positive")
    sample = list(itertools.islice(it, k)) # fill the reservoir
    random.shuffle(sample) # if #items < k then return all items in random order.
    for i, item in enumerate(it, start=k+1):
        j = random.randrange(i) # random [0..i)
        if j < k:
            sample[j] = item # replace item with gradually decreasing probability
    return sample

def generate_initial_guesses(initial_guess_params, k=None, random=False):
    if random:
        if k is None:
            k = 16
        return [[p.random_value() for p in initial_guess_params] for _ in range(k)]
    else:
        param_quantized_values_list = [[]] * len(initial_guess_params)
        for p in initial_guess_params:
            param_quantized_values_list[p.index] = p.quantized_values()
        if k is None:
            return itertools.product(*param_quantized_values_list)
        return reservoir_sample(itertools.product(*param_quantized_values_list), k)

@jit(nopython=True)
def rotation_matrix(a, b):
    """Returns the rotation matrix that rotates vector a to vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s == 0:
        return np.eye(3)
    v = v / s
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rot_part = np.eye(3) + vx + vx.dot(vx) * (1 - c) / s ** 2
    homogeneous_mat = np.eye(4)
    homogeneous_mat[:3, :3] = rot_part
    return homogeneous_mat
