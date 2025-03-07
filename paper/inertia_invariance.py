#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19/4/22
# @Author  : Daniel Ordonez
# @email   : daniels.ordonez@gmail.com

import matplotlib.pyplot as plt
import numpy as np
from pytransform3d import rotations as rt
from pytransform3d import transformations as tr
from pytransform3d.plot_utils import plot_box, plot_sphere, plot_vector, remove_frame
from pytransform3d.transform_manager import TransformManager


def plot_object(A2B, ax=None, k=np.eye(4), **kwargs):
    h, w, d = 0.1, 0.3, 0.5
    t_sphere = np.array([h / 2, w / 2, d / 2, 1])
    plot_box(A2B=A2B, size=[h, w, d], wireframe=False, ax=ax, **kwargs)
    plot_box(A2B=A2B, size=[h, w, d], wireframe=True, ax=ax, **kwargs)
    plot_sphere(p=A2B @ k @ t_sphere, radius=h / 2, wireframe=False, n_steps=20, ax=ax, **kwargs)

    return ax


def plot_lef_handed_axis(A2B, **kwargs):
    p = A2B[:3, 3]
    x, y, z = A2B[:3, 0], A2B[:3, 1], A2B[:3, 2]
    ax = kwargs.pop("ax", None)
    ax = plot_vector(start=p, direction=x, ax=ax, color="r", **kwargs)
    ax = plot_vector(start=p, direction=y, ax=ax, color="g", **kwargs)
    ax = plot_vector(start=p, direction=z, ax=ax, color="b", **kwargs)
    return ax


def reflex_matrix(a):
    assert a.shape[1] == 1
    d = a.shape[0]
    return np.eye(d) - 2 * ((a @ a.T) / (a.T @ a))


def reflection_transformation(vnorm, point_in_plane):
    """Generates the Homogenous trasformation matrix of a reflection"""
    KA = reflex_matrix(vnorm)
    # The plane position is defined as a function of a point in the plane.
    tKA = np.squeeze(-2 * vnorm * (-point_in_plane.dot(vnorm)))
    TK = tr.transform_from(R=KA, p=tKA)
    return TK


if __name__ == "__main__":
    random_state = np.random.RandomState(42)
    mirror_c = "r"
    c = "b"
    tm = TransformManager()
    fig = plt.figure(figsize=(15, 10))

    # Define the plane orientation
    RA_w = rt.matrix_from_compact_axis_angle(rt.random_compact_axis_angle())
    # The normal plane vector is the X axis of the plane rotation matrix
    vA_w = RA_w[:, 0:1]
    pA_w = np.array([0.3, -0.5, 0.3])
    # The Homogenous transformation matrix of the location of the plane is defined as
    TA_w = tr.transform_from(RA_w, pA_w)
    # The Homogenous transformation matrix of reflection w.r.t to the plane is defined as
    TKA_w = reflection_transformation(vA_w, pA_w)
    # TKA_w = tr.transform_from(KA_w, tKA_w)
    tm.add_transform(r"A", "w", TA_w)
    # Plot mirror plane
    ax = plot_box(A2B=TA_w, size=[0.001, 2, 2], wireframe=False, alpha=0.1, color="c")

    # Define Original body position t, and orientation R_o2w
    t_o = np.array([-0.0, 0.1, 0.3])
    R_o2w = rt.active_matrix_from_intrinsic_euler_xyz(random_state.randn(3))
    R_w2o = np.linalg.inv(R_o2w)
    TKA_o = reflection_transformation(R_w2o @ vA_w, R_w2o @ pA_w)
    # The object is assumed to be symmetrical w.r.t the YZ plane
    Kosymm_o = reflex_matrix(np.array([[1], [0], [0]]))
    To_w = tr.transform_from(R_o2w, t_o)
    tm.add_transform(r"o", "w", To_w)
    # Plot body
    plot_object(To_w, alpha=0.2, color="b", ax=ax)

    # Get the configuration of the reflected body, this result in a improper configuration
    Tobar_w = TKA_w @ To_w  # Configuration of the reflected body in world coordinates.
    # R_go2w = Tobar_w[:3,:3] @ Kosymm_o  # By using the body symmetry plane we obtain the proper rotation.
    T_o2w = To_w
    T_w2o = tr.invert_transform(T_o2w)
    T_obar2o = TKA_o
    R_go2o = T_obar2o[:3, :3] @ Kosymm_o
    R_go2w = R_o2w @ R_go2o

    Tgo2_w = tr.transform_from(R=R_go2o, p=Tobar_w[:3, 3])
    plot_lef_handed_axis(Tobar_w, ax=ax, s=0.3, lw=0.005)
    tm.add_transform("go", "w", Tgo2_w)
    plot_object(Tgo2_w, alpha=0.2, color="r", ax=ax)

    remove_frame(ax)
    plt.gca().set_aspect("auto")
    tm.plot_frames_in("w", ax=ax, alpha=0.4, s=0.2)
    fig.show()
    print("Hi")
