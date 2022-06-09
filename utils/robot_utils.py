#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/3/22
# @Author  : Daniel Ordonez 
# @email   : daniels.ordonez@gmail.com
import numpy as np
import scipy.linalg

from groups.SymmetricGroups import C2, Klein4
from robots.atlas.AtlasBullet import AtlasBullet
from robots.bolt.BoltBullet import BoltBullet
from robots.solo.Solo12Bullet import Solo12Bullet
from utils.utils import reflex_matrix


def get_robot_params(robot_name):
    """
    Utility function to select which symmetry groups are used to augment the data and which are used to constrain the
    function approximator, or augment the data during training. This allow us to compare performance when we
    exploit the entire data symmetry groups or only subgroups.
    :param robot_name:
    :return: robot, Gin_data, Gout_data, Gin_model, Gout_model
    """
    Gin_data, Gout_data = None, None     # True data symmetry groups
    Gin_model, Gout_model = None, None   # Function approximator selected symmetry groups

    if 'bolt' in robot_name.lower():
        robot = BoltBullet()
        perm_q = robot.mirror_joint_idx
        perm_q = np.concatenate((perm_q, np.array(perm_q) + len(perm_q))).tolist()
        refx_q = robot.mirror_joint_signs
        refx_q = np.concatenate((refx_q, refx_q)).tolist()
        h_in = C2.oneline2matrix(oneline_notation=perm_q, reflexions=refx_q)
        Gin_data = C2(h_in)
        h_out = C2.oneline2matrix(oneline_notation=[0, 1, 2, 3, 4, 5], reflexions=[1, -1, 1, -1, 1, -1])
        Gout_data = C2(h_out)
        Gin_model, Gout_model = Gin_data, Gout_data  # No subgroup of C2 exists
    elif 'atlas' in robot_name.lower():
        robot = AtlasBullet()
        perm_q = robot.mirror_joint_idx
        perm_q = np.concatenate((perm_q, np.array(perm_q) + len(perm_q))).tolist()
        refx_q = robot.mirror_joint_signs
        refx_q = np.concatenate((refx_q, refx_q)).tolist()
        h_in = C2.oneline2matrix(oneline_notation=perm_q, reflexions=refx_q)
        Gin_data = C2(h_in)
        h_out = C2.oneline2matrix(oneline_notation=[0, 1, 2, 3, 4, 5], reflexions=[1, -1, 1, -1, 1, -1])
        Gout_data = C2(h_out)
        Gin_model, Gout_model = Gin_data, Gout_data  # No subgroup of C2 exists
    elif 'solo' in robot_name.lower():
        robot = Solo12Bullet()
        #                  Sagittal Symmetry                      Transversal symmetry
        perm_q = [[3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8],   [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5]]
        refx_q = [[-1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1], [1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1]]
        perm_q = [p + (np.array(p) + len(p)).tolist() for p in perm_q]
        refx_q = [e + e for e in refx_q]

        g_sagittal = C2.oneline2matrix(oneline_notation=perm_q[0], reflexions=refx_q[0])
        g_transversal = C2.oneline2matrix(oneline_notation=perm_q[1], reflexions=refx_q[1])
        g_sagittal_out = C2.oneline2matrix(oneline_notation=[0, 1, 2, 3, 4, 5],    reflexions=[1, -1, 1, -1,  1, -1])
        g_transversal_out = C2.oneline2matrix(oneline_notation=[0, 1, 2, 3, 4, 5], reflexions=[-1, 1, 1,  1, -1, -1])

        # Original structure symmetry groups
        Gin_data = Klein4(generators=[g_sagittal, g_transversal])
        Gout_data = Klein4(generators=[g_sagittal_out, g_transversal_out])

        if 'c2' in robot_name.lower():   # Data has still Klein4, Model "knows" only C2 symmetry
            Gin_model = C2(g_sagittal)
            Gout_model = C2(g_sagittal_out)
        else:
            Gin_model, Gout_model = Gin_data, Gout_data
    elif "mit" in robot_name.lower() or "cheetah" in robot_name.lower():
        # Joint Space
        #        ____RF___|___LF____|___RH______|____LH____|
        # q    = [ 0, 1, 2,  3, 4, 5,  6,  7,  8, 9, 10, 11]
        perm_q = [ 3, 4, 5,  0, 1, 2,  9, 10, 11,  6, 7,  8]
        refx_q = [-1, 1, 1, -1, 1, 1, -1,  1,  1, -1, 1,  1]
        perm_q = np.concatenate((perm_q, np.array(perm_q) + len(perm_q))).tolist()
        refx_q = np.concatenate((refx_q, refx_q)).tolist()
        # IMU acceleration and angular velocity
        na = np.array([0, 1, 0])   # Normal vector to reflection/symmetry plane.
        Rr = reflex_matrix(na)     # Reflection matrix
        refx_a_IMU = np.squeeze(Rr @ np.ones((3, 1))).astype(np.int).tolist()
        refx_w_IMU = np.squeeze((-Rr) @ np.ones((3, 1))).astype(np.int).tolist()
        perm_a_IMU, perm_w_IMU  = [24, 25, 26], [27, 28, 29]
        # Foot relative positions and velocities
        #            ____RF___|___LF____|___RH______|____LH____|
        # pf=        [0, 1, 2,  3, 4, 5,  6,  7,  8, 9, 10, 11]
        perm_foots = [3, 4, 5,  0, 1, 2,  9, 10, 11, 6,  7,  8]
        refx_foots = scipy.linalg.block_diag(*[Rr] * 4) @ np.ones((12, 1))  # Hips and IMU frames xz planes are coplanar
        refx_foots = np.squeeze(refx_foots).tolist()
        perm_foots = np.concatenate((perm_foots, np.array(perm_foots) + len(perm_foots))).astype(np.int)
        refx_foots = np.concatenate((refx_foots, refx_foots)).astype(np.int)

        perm = perm_q + perm_a_IMU + perm_w_IMU
        perm += (perm_foots + len(perm)).tolist()
        refx = refx_q + refx_a_IMU + refx_w_IMU + refx_foots.tolist()

        h_in = C2.oneline2matrix(oneline_notation=perm, reflexions=refx)
        Gin_data = C2(h_in)
        # One hot encoding of 16 contact states.
        h_out = C2.oneline2matrix(oneline_notation=[0, 2, 1, 3, 8, 10, 9, 11, 4, 6, 5, 7, 12, 14, 13, 15])
        Gout_data = C2(h_out)

        Gin_model, Gout_model = Gin_data, Gout_data  # No subgroup of C2 exists
    else:
        raise NotImplementedError()

    return robot, Gin_data, Gout_data, Gin_model, Gout_model