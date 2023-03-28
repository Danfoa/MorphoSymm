#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 25/1/22
# @Author  : Daniel Ordonez 
# @email   : daniels.ordonez@gmail.com
import pathlib
import numpy as np
import scipy.sparse
import torch
from pytransform3d import transformations as tr, rotations as rt
from scipy.sparse import issparse


def check_if_resume_experiment(ckpt_call):
    ckpt_path = pathlib.Path(ckpt_call.dirpath).joinpath(ckpt_call.CHECKPOINT_NAME_LAST + ckpt_call.FILE_EXTENSION)
    best_path = pathlib.Path(ckpt_call.dirpath).joinpath(ckpt_call.filename + ckpt_call.FILE_EXTENSION)

    terminated = False
    if best_path.exists() and not ckpt_path.exists():
        terminated = True
    elif ckpt_path.exists() and best_path.exists():
        terminated = False

    return terminated, ckpt_path, best_path


def dense(x):
    if issparse(x):
        return x.todense()
    return x


def coo2torch_coo(M: scipy.sparse.coo_matrix):
    density = M.getnnz() / np.prod(M.shape)
    memory = np.prod(M.shape) * 32
    if memory > 1e9:
        idx = np.vstack((M.row, M.col))
        return torch.sparse_coo_tensor(idx, M.data, size=M.shape, dtype=torch.float32).coalesce()
    else:
        return torch.tensor(np.asarray(M.todense(), dtype=np.float32))


def pprint_dict(d: dict):
    str = []
    d_sorted = dict(sorted(d.items()))
    for k, v in d_sorted.items():
        str.append(f"{k}={v}")
    return "-".join(str)


def cm2inch(cm):
    return cm / 2.54


def configure_bullet_simulation(gui=True, debug=False):
    import pybullet_data
    from pybullet import GUI, DIRECT, COV_ENABLE_GUI, COV_ENABLE_SEGMENTATION_MARK_PREVIEW, \
        COV_ENABLE_DEPTH_BUFFER_PREVIEW, \
        COV_ENABLE_MOUSE_PICKING
    from pybullet_utils import bullet_client

    BACKGROUND_COLOR = '--background_color_red=%.2f --background_color_green=%.2f --background_color_blue=%.2f' % \
                       (1.0, 1.0, 1.0)

    if gui:
        pb = bullet_client.BulletClient(connection_mode=GUI, options=BACKGROUND_COLOR)
    else:
        pb = bullet_client.BulletClient(connection_mode=DIRECT)
    pb.configureDebugVisualizer(COV_ENABLE_GUI, debug)
    pb.configureDebugVisualizer(COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
    pb.configureDebugVisualizer(COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    pb.configureDebugVisualizer(COV_ENABLE_MOUSE_PICKING, 0)

    pb.resetSimulation()
    pb.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    # Load floor
    # floor_id = pb.loadURDF("plane.urdf", basePosition=[0, 0, 0.0], useFixedBase=1)
    return pb


def symbolic_matrix(base_name, rows, cols):
    from sympy import Symbol

    SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    w = np.empty((rows, cols), dtype=object)
    for r in range(rows):
        for c in range(cols):
            var_name = "%s%d,%d" % (base_name, r + 1, c + 1)
            w[r, c] = Symbol(var_name.translate(SUB))
    return w


def permutation_matrix(oneline_notation):
    d = len(oneline_notation)
    assert d == len(np.unique(oneline_notation)), np.unique(oneline_notation, return_counts=True)
    P[range(d), np.abs(oneline_notation)] = 1
    return P


def is_canonical_permutation(ph):
    return np.allclose(ph @ ph, np.eye(ph.shape[0]))


def append_dictionaries(dict1, dict2, recursive=True):
    result = {}
    for k in set(dict1) | set(dict2):
        item1, item2 = dict1.get(k, 0), dict2.get(k, 0)
        if isinstance(item1, list) and (isinstance(item2, int) or isinstance(item2, float)):
            result[k] = item1 + [item2]
        elif isinstance(item1, int) or isinstance(item1, float):
            result[k] = [item1, item2]
        elif isinstance(item1, torch.Tensor) and isinstance(item2, torch.Tensor):
            result[k] = torch.cat((item1, item2))
        elif isinstance(item1, dict) and isinstance(item2, dict) and recursive:
            result[k] = append_dictionaries(item1, item2)
    return result


import unicodedata
import re


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


if __name__ == "__main__":
    a = (2, 3, 0, 1)
    P = permutation_matrix(a)
    assert is_canonical_permutation(P)


def reflection_matrix(plane_norm_vector):
    aa = np.expand_dims(plane_norm_vector, -1) if plane_norm_vector.shape == (3,) else plane_norm_vector
    d = aa.shape[0]
    return np.eye(d) - 2 * ((aa @ aa.T) / (aa.T @ aa))


def homogenousMatrix(R: np.ndarray, T=np.zeros(3)):
    X = np.zeros((4, 4), dtype=R.dtype)
    X[3, 3] = 1
    X[:, 3] = T
    X[:3, :3] = R
    return X


def reflection_transformation(vnorm, point_in_plane):
    """Generates the Homogenous trasformation matrix of a reflection"""
    if vnorm.ndim == 1:
        vnorm = np.expand_dims(vnorm, axis=1)
    KA = reflection_matrix(vnorm)
    # The plane position is defined as a function of a point in the plane.
    tKA = np.squeeze(-2 * vnorm * (-point_in_plane.dot(vnorm)))
    TK = tr.transform_from(R=KA, p=tKA)
    return TK


def matrix_to_quat_xyzw(R):
    assert R.shape == (3, 3)
    return rt.quaternion_xyzw_from_wxyz(rt.quaternion_from_matrix(R))


def quat_xyzw_to_SO3(q):
    assert q.shape == (4,)
    return rt.matrix_from_quaternion(rt.quaternion_wxyz_from_xyzw(q))


def SE3_2_gen_coordinates(X):
    assert X.shape == (4, 4)
    pos = X[:3, 3]
    quat = matrix_to_quat_xyzw(X[:3, :3])
    return np.concatenate((pos, quat))
