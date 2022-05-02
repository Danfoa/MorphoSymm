#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/3/22
# @Author  : Daniel Ordonez 
# @email   : daniels.ordonez@gmail.com
import numpy as np

from groups.SymmetricGroups import C2, Klein4
from robots.atlas.AtlasBullet import AtlasBullet
from robots.bolt.BoltBullet import BoltBullet
from robots.solo.Solo12Bullet import Solo12Bullet


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
        pq = robot.mirror_joint_idx
        pq = np.concatenate((pq, np.array(pq) + len(pq))).tolist()
        rq = robot.mirror_joint_signs
        rq = np.concatenate((rq, rq)).tolist()
        h_in = C2.oneline2matrix(oneline_notation=pq, reflexions=rq)
        Gin_data = C2(h_in)
        h_out = C2.oneline2matrix(oneline_notation=[0, 1, 2, 3, 4, 5], reflexions=[1, -1, 1, -1, 1, -1])
        Gout_data = C2(h_out)
        Gin_model, Gout_model = Gin_data, Gout_data  # No subgroup of C2 exists
    elif 'atlas' in robot_name.lower():
        robot = AtlasBullet()
        pq = robot.mirror_joint_idx
        pq = np.concatenate((pq, np.array(pq) + len(pq))).tolist()
        rq = robot.mirror_joint_signs
        rq = np.concatenate((rq, rq)).tolist()
        h_in = C2.oneline2matrix(oneline_notation=pq, reflexions=rq)
        Gin_data = C2(h_in)
        h_out = C2.oneline2matrix(oneline_notation=[0, 1, 2, 3, 4, 5], reflexions=[1, -1, 1, -1, 1, -1])
        Gout_data = C2(h_out)
        Gin_model, Gout_model = Gin_data, Gout_data  # No subgroup of C2 exists
    elif 'solo' in robot_name.lower():
        robot = Solo12Bullet()
        #                  Sagittal Symmetry                      Transversal symmetry
        pq = [[3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8],   [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5]]
        rq = [[-1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1], [1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1]]
        pq = [p + (np.array(p) + len(p)).tolist() for p in pq]
        rq = [e + e for e in rq]

        g_sagittal = C2.oneline2matrix(oneline_notation=pq[0], reflexions=rq[0])
        g_transversal = C2.oneline2matrix(oneline_notation=pq[1], reflexions=rq[1])
        g_sagittal_out = C2.oneline2matrix(oneline_notation=[0, 1, 2, 3, 4, 5], reflexions=[1, -1, 1, -1, 1, -1])
        g_transversal_out = C2.oneline2matrix(oneline_notation=[0, 1, 2, 3, 4, 5], reflexions=[-1, 1, 1, 1, -1, -1])

        # Original structure symmetry groups
        Gin_data = Klein4(generators=[g_sagittal, g_transversal])
        Gout_data = Klein4(generators=[g_sagittal_out, g_transversal_out])

        if 'c2' in robot_name.lower():   # Data has still Klein4, Model "knows" only C2 symmetry
            Gin_model = C2(g_sagittal)
            Gout_model = C2(g_sagittal_out)
        else:
            Gin_model, Gout_model = Gin_data, Gout_data
    else:
        raise NotImplementedError()

    return robot, Gin_data, Gout_data, Gin_model, Gout_model