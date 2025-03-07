#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 25/1/22
# @Author  : Daniel Ordonez
# @email   : daniels.ordonez@gmail.com
import pathlib

import numpy as np
import scipy.sparse
import torch
from scipy.spatial.transform import Rotation


def check_if_resume_experiment(ckpt_call):
    """Check if the experiment is a resumed experiment."""
    ckpt_path = pathlib.Path(ckpt_call.dirpath).joinpath(ckpt_call.CHECKPOINT_NAME_LAST + ckpt_call.FILE_EXTENSION)
    best_path = pathlib.Path(ckpt_call.dirpath).joinpath(ckpt_call.filename + ckpt_call.FILE_EXTENSION)

    terminated = False
    if best_path.exists() and not ckpt_path.exists():
        terminated = True
    elif ckpt_path.exists() and best_path.exists():
        terminated = False

    return terminated, ckpt_path, best_path


def coo2torch_coo(M: scipy.sparse.coo_matrix):
    """Convert sparse scipy coo matrix to pytorch sparse coo matrix."""
    M.getnnz() / np.prod(M.shape)
    memory = np.prod(M.shape) * 32
    if memory > 1e9:
        idx = np.vstack((M.row, M.col))
        return torch.sparse_coo_tensor(idx, M.data, size=M.shape, dtype=torch.float32).coalesce()
    else:
        return torch.tensor(np.asarray(M.todense(), dtype=np.float32))


def permutation_matrix(oneline_notation):
    """Generate a permutation matrix from its oneline notation."""
    d = len(oneline_notation)
    assert d == np.unique(oneline_notation).size, "oneline_notation must describe a non-defective permutation"
    P = np.zeros((d, d), dtype=int)
    P[range(d), np.abs(oneline_notation)] = 1
    return P


def gen_permutation_matrix(oneline_notation, reflections):
    """Generate a permutation matrix from its oneline notation and a list of reflections."""
    P = permutation_matrix(oneline_notation)
    ref = np.asarray(reflections).reshape(-1, 1)
    genP = ref * P
    return genP


def append_dictionaries(dict1, dict2, recursive=True):
    """Append two dictionaries. If the same key is present in both dictionaries, the values are appended."""
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


def slugify(value, allow_unicode=False):
    """Taken from github.com/django/django/blob/master/django/utils/text.py.

    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    import re
    import unicodedata

    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


# def reflection_matrix(plane_norm_vector):
#     aa = np.expand_dims(plane_norm_vector, -1) if plane_norm_vector.shape == (3,) else plane_norm_vector
#     d = aa.shape[0]
#     return np.eye(d) - 2 * ((aa @ aa.T) / (aa.T @ aa))


# def homogenousMatrix(R: np.ndarray, T=np.zeros(3)):
#     X = np.zeros((4, 4), dtype=R.dtype)
#     X[3, 3] = 1
#     X[:, 3] = T
#     X[:3, :3] = R
#     return X

# def reflection_transformation(vnorm, point_in_plane):
#     """Generates the Homogenous trasformation matrix of a reflection."""
#     if vnorm.ndim == 1:
#         vnorm = np.expand_dims(vnorm, axis=1)
#     KA = reflection_matrix(vnorm)
#     # The plane position is defined as a function of a point in the plane.
#     tKA = np.squeeze(-2 * vnorm * (-point_in_plane.dot(vnorm)))
#     TK = tr.transform_from(R=KA, p=tKA)
#     return TK


def matrix_to_quat_xyzw(R):
    """SO(3) rotation to xyzw quaternion representation."""
    return Rotation.from_matrix(R).as_quat(scalar_first=False)


def quat_xyzw_to_SO3(q):
    """Xyzw quaternion representation to SO(3) representation."""
    return Rotation.from_quat(q, scalar_first=False).as_matrix()


def SE3_2_gen_coordinates(X):
    """Convert a homogenous matrix in SE(3) in R^{4x4} to vect-quaternion representation (R^{7}).

    In pinocchio and physics simulations representing position and orientation as a 3D position vector and a 3D
    quaternion (xyzw convention used in pybullet).
    """
    assert X.shape == (4, 4)
    pos = X[:3, 3]
    quat = matrix_to_quat_xyzw(X[:3, :3])
    return np.concatenate((pos, quat))


def matrix_from_two_vectors(a, b):
    """Compute rotation matrix from two vectors.

    We assume that the two given vectors form a plane so that we can compute
    a third, orthogonal vector with the cross product.

    The x-axis will point in the same direction as a, the y-axis corresponds
    to the normalized vector rejection of b on a, and the z-axis is the
    cross product of the other basis vectors.

    Parameters
    ----------
    a : array-like, shape (3,)
        First vector, must not be 0

    b : array-like, shape (3,)
        Second vector, must not be 0 or parallel to a

    Returns:
    -------
    R : array, shape (3, 3)
        Rotation matrix

    Raises:
    ------
    ValueError
        If vectors are parallel or one of them is the zero vector
    """
    if np.linalg.norm(a) == 0:
        raise ValueError("a must not be the zero vector.")
    if np.linalg.norm(b) == 0:
        raise ValueError("b must not be the zero vector.")

    c = np.cross(a, b)
    if np.linalg.norm(c) == 0:
        raise ValueError("a and b must not be parallel.")

    a = a / np.linalg.norm(a)

    b_on_a_projection = vector_projection(b, a)
    b_on_a_rejection = b - b_on_a_projection
    b = b_on_a_rejection / np.linalg.norm(b_on_a_projection)

    c = c / np.linalg.norm(c)

    return np.column_stack((a, b, c))


def vector_projection(a, b):
    """Orthogonal projection of vector a on vector b.

    Parameters
    ----------
    a : array-like, shape (3,)
        Vector a that will be projected on vector b

    b : array-like, shape (3,)
        Vector b on which vector a will be projected

    Returns:
    -------
    a_on_b : array, shape (3,)
        Vector a
    """
    b_norm_squared = np.dot(b, b)
    if b_norm_squared == 0.0:
        return np.zeros(3)
    return np.dot(a, b) * b / b_norm_squared
