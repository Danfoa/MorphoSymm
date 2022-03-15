#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/3/22
# @Author  : Daniel Ordonez 
# @email   : daniels.ordonez@gmail.com
import numpy as np

from groups.SymmetricGroups import C2, Klein4
from robots.bolt.BoltBullet import BoltBullet
from robots.solo.Solo12Bullet import Solo12Bullet


def get_robot_params(robot_name):
    if 'bolt' in robot_name.lower():
        robot = BoltBullet()
        pq = [3, 4, 5, 0, 1, 2]
        pq.extend((np.array(pq) + len(pq)).tolist())
        rq = [-1, 1, 1, -1, 1, 1]
        rq.extend(rq)
        h_in = C2.oneline2matrix(oneline_notation=pq, reflexions=rq)
        Gin = C2(h_in)
        h_out = C2.oneline2matrix(oneline_notation=[0, 1, 2, 3, 4, 5], reflexions=[1, -1, 1, -1, 1, -1])
        Gout = C2(h_out)
        G = C2
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

        if 'c2' in robot_name.lower():   # Use only sagittal symmetry for Solo.
            Gin = C2(g_sagittal)
            Gout = C2(g_sagittal_out)
            G = C2
        else:   # Use Sagittal and Transversal Symmetries.
            Gin = Klein4(generators=[g_sagittal, g_transversal])
            Gout = Klein4(generators=[g_sagittal_out, g_transversal_out])
            G = Klein4
    else:
        raise NotImplementedError()

    return robot, Gin, Gout, G