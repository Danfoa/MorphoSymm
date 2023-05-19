#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/3/22
# @Author  : Daniel Ordonez 
# @email   : daniels.ordonez@gmail.com

import importlib
import re

import escnn
import numpy as np
from omegaconf import DictConfig
from scipy import sparse

# from groups.SparseRepresentation import SparseRep, SparseRepE3
# from groups.SymmetryGroups import C2, Klein4, Sym
from robots.PinBulletWrapper import PinBulletWrapper
from . import algebra_utils
from .pybullet_visual_utils import setup_debug_sliders, listen_update_robot_sliders
from .algebra_utils import reflection_matrix, configure_bullet_simulation


def class_from_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c


def get_escnn_group(cfg: DictConfig):
    group_label = cfg.group_label

    label_pattern = r'([A-Za-z]+)(\d+)'
    match = re.match(label_pattern, group_label)
    if match:
        group_class = match.group(1)
        order = int(match.group(2))
    else:
        raise AttributeError(f'Group label {group_label} is not a known group label (Dn: Dihedral, Cn: Cyclic) order n')

    group_axis = np.array([0, 0, 1])
    subgroup_id = np.zeros_like(group_axis, dtype=np.bool).astype(object)
    if group_class.lower() == 'd':  # Dihedral
        # Define the symmetry space using presets from ESCNN
        # subgroup_id[group_axis == 1] = order
        symmetry_space = escnn.gspaces.dihedralOnR3(n=order//2, axis=0.0)
    elif group_class.lower() == 'c':  # Cyclic
        assert order >= 2, f'Order of cyclic group must be greater than 2, got {order}'
        subgroup_id[group_axis == 1] = order
        symmetry_space = escnn.gspaces.GSpace3D(tuple(subgroup_id))
    elif group_class.lower() == "k":  # Klein four group
        is_planar_subgroup = True
        symmetry_space = escnn.gspaces.GSpace3D(sg_id=(is_planar_subgroup, False, 2))

    else:
        raise NotImplementedError(f"We have to implement the case of {group_label}")

    if cfg.subgroup_id is not None:
        pass
    return symmetry_space


def load_robot_and_symmetries(robot_cfg, debug=False) -> [PinBulletWrapper, escnn.gspaces.GSpace3D]:
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

    symmetry_space = get_escnn_group(robot_cfg)

    G = symmetry_space.fibergroup

    # Transformation required to obtain ρ_Q_js(g) ∈ G from the regular fields / permutation rep.
    num_symmetric_dof = robot.nj - robot_cfg.unique_bodies
    assert num_symmetric_dof % G.order() == 0, f"Number of symmetric DoF ({num_symmetric_dof}) must be a multiple of " \
                                               f"the group order ({G.order()})"
    norbits = num_symmetric_dof // G.order()

    # rep_reg = G.regular_representation
    # p = [np.asarray(rep_reg(g)) for g in G.elements]
    # pp = [np.asarray((G.irrep(1) + G.irrep(1))(g)) for g in G.elements]
    rep_QJ_reg = escnn.group.directsum([G.regular_representation] * (norbits))
    if robot_cfg.unique_bodies > 0:
        # In the case there are 'nu' unique DoF unafected by the representation rep_QJ we add 'nu' trivial reps
        # at the back of the space of the regular representation.
        rep_QJ_unique = escnn.group.directsum([G.trivial_representation] * (robot_cfg.unique_bodies))
        rep_QJ_reg = rep_QJ_reg + rep_QJ_unique

    # Define the coordinate change matrix P, which goes from the canonical regular rep (with trivial reps at the end)
    # to the joint-space basis defined by Pinocchio and the URDF. TODO: Add example of this to documentation.
    P = algebra_utils.permutation_matrix(robot_cfg.perm_qj)
    P = np.array(robot_cfg.refx_qj, dtype=np.int)[None, :] * P
    rep_QJ = escnn.group.change_basis(rep_QJ_reg, P, name="QJ",
                                      supported_nonlinearities=rep_QJ_reg.supported_nonlinearities)
    G.representations.update(QJ=rep_QJ)  # Save rep in group representations for future use.

    x = np.random.rand(robot.nj)

    for i, g in enumerate(G.elements):
        if i == 0: continue
        gg = rep_QJ(g)
        # gx = (a_reg @ x[:, None]).flatten()
        print(f"ρ_QJ({g}) = {rep_QJ(g)}")
    # Configure E3 representations and group
    if isinstance(G, escnn.group.CyclicGroup):
        rep_E3 = G.irrep(0) + G.irrep(1) + G.trivial_representation
    elif isinstance(G, escnn.group.DihedralGroup):
        rep_E3 = G.irrep(1) + G.irrep(0) + G.trivial_representation
    elif isinstance(G, escnn.group.DirectProductGroup):
        rep_E3 = G.representations['square'] + G.trivial_representation
    else:
        raise NotImplementedError(f"Group {G} not implemented yet.")

    rep_E3.name = "E3"
    G.representations.update(E3=rep_E3)  # Save E3 rep in group representations for future use.

    return robot, symmetry_space
