#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/3/22
# @Author  : Daniel Ordonez 
# @email   : daniels.ordonez@gmail.com
import copy
import pathlib

import numpy as np
import scipy.linalg
from scipy import sparse

from pytransform3d import rotations as rt
from pytransform3d import transformations as tr

from groups.SparseRepresentation import SparseRep, SparseRepE3
from groups.SymmetryGroups import C2, Klein4, Sym
from utils.utils import reflection_matrix

def get_robot_params(robot_name):
    """
    Utility function to select which symmetry groups are used to augment the data and which are used to constrain the
    function approximator, or augment the data during training. This allow us to compare performance when we
    exploit the entire data symmetry groups or only subgroups.
    :param robot_name:
    :return: robot, Gin_data, Gout_data, Gin_model, Gout_model
    """
    rep_E3, rep_qj = None, None                # Morphological Symmetry Representations
    rep_data_in, rep_data_out = None, None     # Input x and output y group representations
    rep_model_in, rep_model_out = None, None   # Input x and output y group representations assumed by the model
    robot = None

    try:
        from hydra.utils import get_original_cwd
        resources = pathlib.Path(get_original_cwd()) / 'robots'
    except:
        resources = pathlib.Path("robots/")

    if 'bolt' in robot_name.lower():
        from robots.bolt.BoltBullet import BoltBullet
        robot = BoltBullet(resources=resources / 'bolt/bolt_description')
        n_sagittal = np.array([0, 1, 0])  # Normal vector to reflection/symmetry plane w.r.t base frame
        R_s = reflection_matrix(n_sagittal)
        R_s = sparse.coo_matrix(R_s)
        G_E3 = C2(generator=R_s)
        rep_E3 = SparseRepE3(G_E3)

        # Configure qj (joint-space) group actions
        #        |___ R __|___ L ___|
        perm_q = [3,  4, 5,  0, 1, 2]
        refx_q = [-1, 1, 1, -1, 1, 1]
        g_qj_sagittal = Sym.oneline2matrix(oneline_notation=perm_q, reflexions=refx_q)
        G_qj = C2(g_qj_sagittal)
        rep_qj = SparseRep(G_qj)

        # Compute representation for a vector x=[qj, dqj].T (joint position and joint velocity) for CoM estimation
        rep_x = rep_qj + rep_qj
        # Compute representation for y=[l, k], the CoM linear `l` and angular `k` momentum.
        rep_l = SparseRepE3(G_E3, pseudovector=False)
        rep_k = SparseRepE3(G_E3, pseudovector=True)
        rep_y = rep_l + rep_k

        rep_model_in, rep_model_out = rep_x, rep_y
        rep_data_in, rep_data_out = rep_model_in, rep_model_out  # C2 has only the trivial group as subgroup

    elif 'atlas' in robot_name.lower():
        from robots.atlas.AtlasBullet import AtlasBullet
        robot = AtlasBullet(resources=resources / 'atlas/atlas_description')
        # Configure E3 group actions, due to sagittal symmety
        n_sagittal = np.array([0, 1, 0])  # Normal vector to reflection/symmetry plane w.r.t base frame
        R_s = reflection_matrix(n_sagittal)
        R_s = sparse.coo_matrix(R_s)
        G_E3 = C2(generator=R_s)
        rep_E3 = SparseRepE3(G_E3)

        # Configure qj (joint-space) group actions
        perm_q = robot.mirror_joint_idx
        refx_q = robot.mirror_joint_signs
        g_qj_sagittal = Sym.oneline2matrix(oneline_notation=perm_q, reflexions=refx_q)
        G_qj = C2(g_qj_sagittal)
        rep_qj = SparseRep(G_qj)

        # Compute representation for a vector x=[qj, dqj].T (joint position and joint velocity) for CoM estimation
        rep_x = rep_qj + rep_qj
        # Compute representation for y=[l, k], the CoM linear `l` and angular `k` momentum.
        rep_l = SparseRepE3(G_E3, pseudovector=False)
        rep_k = SparseRepE3(G_E3, pseudovector=True)
        rep_y = rep_l + rep_k

        rep_model_in, rep_model_out = rep_x, rep_y
        rep_data_in, rep_data_out = rep_model_in, rep_model_out  # C2 has only the trivial group as subgroup

    elif 'solo' in robot_name.lower():
        from robots.solo.Solo12Bullet import Solo12Bullet
        robot = Solo12Bullet(resources=resources / 'solo/solo_description')
        # Configure E3 group actions
        n_sagittal = np.array([0, 1, 0])  # Normal vector to reflection/symmetry plane.
        n_transversal = np.array([1, 0, 0])  # Normal vector to reflection/symmetry plane.
        R_s = reflection_matrix(n_sagittal)
        R_t = reflection_matrix(n_transversal)
        R_s, R_t = sparse.coo_matrix(R_s), sparse.coo_matrix(R_t)

        K4_E3 = Klein4(generators=[R_s, R_t])
        rep_E3 = SparseRepE3(K4_E3)
        # Configure qj (joint-space) group actions
        #                  Sagittal Symmetry                      Transversal symmetry
        perm_q = [[3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5]]
        # Reflections are determined by joint frame predefined orientation.
        refx_q = [[-1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1], [1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1]]

        # Ground truth generalized coordinates Symmetry Group.
        g_sagittal = Sym.oneline2matrix(oneline_notation=perm_q[0], reflexions=refx_q[0])
        g_transversal = Sym.oneline2matrix(oneline_notation=perm_q[1], reflexions=refx_q[1])
        K4_qj = Klein4(generators=[g_sagittal, g_transversal])
        rep_qj = SparseRep(K4_qj)

        rep_data_in = 2 * SparseRep(K4_qj)
        rep_data_out = SparseRepE3(K4_E3) + SparseRepE3(K4_E3, pseudovector=True)

        # Configure symmetries the model will see, in case the subgroup C2 is used
        if 'c2' in robot_name.lower():
            C2_E3 = C2(generator=R_s)
            C2_qj = C2(generator=g_sagittal)
            G_E3, G_qj = C2_E3, C2_qj
        else:  # Klein four-group
            G_E3, G_qj = K4_E3, K4_qj

        # Compute representation for a vector x=[qj, dqj].T (joint position and joint velocity) for CoM estimation
        req_qj = SparseRep(G_qj)
        rep_x = req_qj + req_qj

        # Compute representation for y=[l, k], the CoM linear `l` and angular `k` momentum.
        rep_l = SparseRepE3(G_E3, pseudovector=False)
        rep_k = SparseRepE3(G_E3, pseudovector=True)
        rep_y = rep_l + rep_k

        rep_model_in, rep_model_out = rep_x, rep_y

    elif "mit" in robot_name.lower() or "cheetah" in robot_name.lower():
        # Configure E3 group actions
        n_sagittal = np.array([0, 1, 0])  # Normal vector to reflection/symmetry plane w.r.t base frame
        R_s = reflection_matrix(n_sagittal)
        R_s = sparse.coo_matrix(R_s)
        G_E3 = C2(generator=R_s)
        rep_E3 = SparseRepE3(G_E3)
        # Configure qj (joint-space) group actions
        #        ____RF___|___LF____|___RH______|____LH____|
        # q    = [ 0, 1, 2,  3, 4, 5,  6,  7,  8, 9, 10, 11]
        perm_q = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        # Reflections are determined by joint frame predefined orientation, hips need to be reflected.
        refx_q = [-1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1]
        g_qj_sagittal = Sym.oneline2matrix(oneline_notation=perm_q, reflexions=refx_q)
        G_qj = C2(g_qj_sagittal)
        rep_qj = SparseRep(G_qj)

        # Configure input x and output y representations
        # x = [qj, dqj, a, w, pf, vf] - a: linear IMU acc, w: angular IMU velocity, pf:feet positions, vf:feet velocity
        rep_a = SparseRepE3(G_E3, pseudovector=False)
        rep_w = SparseRepE3(G_E3, pseudovector=True)

        # Configure pf, vf ∈ R^12  representations composed of reflections and permutations
        n_legs = 4
        rep_legs_reflected = n_legs * SparseRepE3(G_E3)    # Same representation as the hips ref frames are collinear with base.
        G_legs_reflected = rep_legs_reflected.G
        g_q_perm = Sym.oneline2matrix(oneline_notation=perm_q)  # Permutation swapping legs.
        G_pf = C2(generator=g_q_perm @ G_legs_reflected.discrete_generators[0])
        rep_pf = SparseRep(G_pf)
        rep_vf = SparseRep(G_pf)

        # x   = [ qj   ,  dqj   ,   a   ,   w   ,   pf   ,   vf  ]
        rep_x = rep_qj + rep_qj + rep_a + rep_w + rep_pf + rep_vf
        # y : 16 dimensional contact state with following symmetry. See paper abstract.
        g_y = C2.oneline2matrix(oneline_notation=[0, 2, 1, 3, 8, 10, 9, 11, 4, 6, 5, 7, 12, 14, 13, 15])
        G_y = C2(g_y)
        rep_y = SparseRep(G_y)

        rep_data_in, rep_data_out = rep_x, rep_y
        rep_model_in, rep_model_out = rep_x, rep_y
    else:
        raise NotImplementedError()

    return robot, rep_data_in, rep_data_out, rep_model_in, rep_model_out, rep_E3, rep_qj
