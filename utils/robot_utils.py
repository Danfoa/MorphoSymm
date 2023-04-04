#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/3/22
# @Author  : Daniel Ordonez 
# @email   : daniels.ordonez@gmail.com

import importlib

import numpy as np
from scipy import sparse

from groups.SparseRepresentation import SparseRep, SparseRepE3
from groups.SymmetryGroups import Sym
from robots.PinBulletWrapper import PinBulletWrapper
from .pybullet_visual_utils import setup_debug_sliders, listen_update_robot_sliders
from .utils import reflection_matrix, configure_bullet_simulation


def class_from_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c

def load_robot_and_symmetries(robot_cfg, debug=False):
    """
    Utility function to get the symmetry group and representations of a robotic system defined in config.
    :param robot_cfg: Dictionary holding the configuration parameters of the robot. Check `cfg/robot/`
    :param debug: whether to load the robot into an interactive simulation session to visually debug URDF
    :return:
        robot: `PinBulletWrapper` instance with the robot loaded in pinocchio and ready to be loaded in pyBullet
        rep_Ed: Representation of the symmetry group of the robot on the Euclidean space of `d` dimensions.
        rep_QJ: Representation of the symmetry group of the robot on the Joint Space of the robot.
    """
    robot_name = str.lower(robot_cfg.name)
    robot = PinBulletWrapper(robot_name=robot_name, init_q=robot_cfg.init_q, hip_height=robot_cfg.hip_height,
                             endeff_names=robot_cfg.endeff_names)

    if debug:
        pb = configure_bullet_simulation(gui=True, debug=debug)
        robot.configure_bullet_simulation(pb, world=None)
        setup_debug_sliders(pb, robot)
        listen_update_robot_sliders(pb, robot)

    # Permutations and reflection defining each ρ_Qjs(g) ∈ G.
    perm_q_js = robot_cfg.perm_qj
    refx_q_js = robot_cfg.refx_qj
    # Get the group class to validate group axioms.
    G_class = class_from_name('groups.SymmetryGroups', robot_cfg.G if robot_cfg.gens_ids is None else robot_cfg.G_sub)

    # Configure QJ representations and group, using the group generators alone.
    generators_QJ = []
    for gen_id, (p, r) in enumerate(zip(perm_q_js, refx_q_js)):
        if robot_cfg.gens_ids is not None and gen_id not in robot_cfg.gens_ids:
            continue
        # `g` here is the actual representation matrix R^{nj x nj} describing the permutations and reflections of QJ
        g = Sym.oneline2matrix(oneline_notation=p, reflexions=r)
        generators_QJ.append(g)
    G_QJ = G_class(generators=generators_QJ)

    # For now since we are using for the moment only reflection actions, we define the reflection from the normal vector
    # of the plane of reflection. # TODO: enable non abelian representations also in config (i.e, also rotations/translations)
    n_reflex = robot_cfg.n_reflex
    generators_E3 = []
    for gen_id, n in enumerate(n_reflex):
        if robot_cfg.gens_ids is not None and gen_id not in robot_cfg.gens_ids:
            continue
        # Use the vector normal to the reflection plane to build the actual 3x3 reflection matrix R
        n_vect = np.array(n)  # Normal vector to reflection/symmetry plane w.r.t base frame
        R = reflection_matrix(n_vect)
        R = sparse.coo_matrix(R)
        generators_E3.append(R)
    # TODO: E3 representations should consider translations also. Must change (not so relevant for now)
    G_E3 = G_class(generators=generators_E3)

    # Set representations
    rep_Ed = SparseRepE3(G_E3)
    rep_QJ = SparseRep(G_QJ)

    return robot, rep_Ed, rep_QJ
