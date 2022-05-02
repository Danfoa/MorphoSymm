#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19/4/22
# @Author  : Daniel Ordonez 
# @email   : daniels.ordonez@gmail.com

import numpy as np
from pytransform3d import rotations as rt
from pytransform3d import transformations as tr
from pytransform3d.plot_utils import plot_vector, plot_box, plot_sphere, remove_frame
from pytransform3d.transform_manager import TransformManager

import matplotlib.pyplot as plt

def plot_object(A2B, ax=None, k=np.eye(4) , **kwargs):
    h, w, d = 0.1, 0.3, 0.5
    t_sphere = np.array([h/2, w/2, d/2, 1])
    plot_box(A2B=A2B, size=[h, w, d], wireframe=False, ax=ax, **kwargs)
    plot_box(A2B=A2B, size=[h, w, d], wireframe=True, ax=ax, **kwargs)
    plot_sphere(p=A2B @ k @ t_sphere, radius=h/2, wireframe=False, n_steps=20, ax=ax, **kwargs)

    return ax

def plot_lef_handed_axis(A2B, **kwargs):
    p = A2B[:3,3]
    x, y, z = A2B[:3, 0], A2B[:3, 1], A2B[:3, 2]
    ax = kwargs.pop("ax", None)
    ax = plot_vector(start=p, direction=x, ax=ax, color="r", **kwargs)
    ax = plot_vector(start=p, direction=y, ax=ax, color="g", **kwargs)
    ax = plot_vector(start=p, direction=z, ax=ax, color="b", **kwargs)
    return ax

def reflex_matrix(a):
    assert a.shape[1] == 1
    d = a.shape[0]
    return np.eye(d) - 2 * ((a @ a.T)/(a.T @ a))

if __name__ == "__main__":
    random_state = np.random.RandomState(42)
    mirror_c = 'r'
    c = 'b'
    tm = TransformManager()

    # a = np.random.random((3, 1))
    # A_n = np.array([1, 0, 0])[None, :].T
    A_n = np.random.random((3, 1))
    KA_w = reflex_matrix(A_n)
    A_t = np.array([0, 0, 0])
    TKA_w = tr.transform_from(KA_w, A_t)

    assert np.allclose(KA_w, KA_w.T), "Reflection matrices should be equal to their transpose"

    R_p2o = rt.matrix_from_euler_xyz(random_state.randn(3))
    K_o = reflex_matrix(np.array([[1], [0], [0]]))
    I_p = np.diag(np.random.randn(3)**2)

    a = KA_w @ I_p
    KI_p = KA_w.T @ I_p @ KA_w

    # assert np.allclose(I_p, KI_p), (I_p, KI_p)
    w_o = np.random.random((3, 1))
    w_op = KA_w @ w_o

    I_o = R_p2o @ I_p @ R_p2o.T
    I_op = (K_o @ R_p2o) @ I_p @ (K_o @ R_p2o).T
    I_op_true = (K_o @ R_p2o) @ I_p @ (K_o @ R_p2o).T

    E_o = w_o.T @ I_o @ w_o
    E_op = w_o.T @ I_op @ w_o

    # Plot mirror plane
    fig = plt.figure(figsize=(15, 10))
    ax = plot_box(A2B=tr.transform_from(rt.matrix_from_compact_axis_angle(A_n.flatten()), A_t),
                  size=[0.01, 1, 1], wireframe=False, alpha=0.1, color="c")

    # Define Original body
    t_o = np.array([-0.5, 0., 0.])
    R_o2w = rt.active_matrix_from_intrinsic_euler_xyz(random_state.randn(3))
    R_w2o = np.linalg.inv(R_o2w)
    T_o2w = tr.transform_from(R_o2w, t_o)
    # Plot body
    plot_object(T_o2w, alpha=0.2, color="b", ax=ax)

    # Symmetry reflection
    g_n = np.array([[1], [0], [0]])
    Kg_o = reflex_matrix(g_n)
    # Symmetry equivalent rotation
    Rop_w = KA_w @ R_o2w @ Kg_o
    t_op = KA_w @ t_o
    T_op2w = tr.transform_from(Rop_w, t_op)
    plot_object(T_op2w, alpha=0.2, color="r", ax=ax,)
    plot_lef_handed_axis(TKA_w @ T_o2w, ax=ax, s=0.1, lw=0.01)

    # Add transformation to history
    tm.add_transform(r'o', "w", T_o2w)
    tm.add_transform(r'$\bar{o}$', "w", T_op2w)

    tm.plot_frames_in("w", ax=ax, alpha=0.4, s=0.2)


    remove_frame(ax)
    plt.gca().set_aspect("auto")
    fig.show()

    I_p = np.diag(np.random.random((3)) * 10)



    hat_Ww = Ka_w @ W_w
    hat_Wo = Ka_w @ R_w2o @ hat_Ww

    # Define frame of principal axes w.r.t body ref frame o
    theta = np.pi/4
    euler = np.array([0, 0, theta])
    R_o2p = rt.active_matrix_from_intrinsic_euler_xyz(euler)
    R_o2p_neg = rt.active_matrix_from_intrinsic_euler_xyz(-euler)
    R_p2o = np.linalg.inv(R_o2p)


    # Obtain the body frame Inertia Matrix
    Io = R_p2o @ I_p @ R_p2o.T
    # Calculate original Kinetic Energy
    KinE_p = (R_o2p @ Wo).T @ I_p @ (R_o2p @ Wo)
    KinE_o = (Wo).T @ Io @ (Wo)
    assert np.allclose(KinE_p, KinE_o), KinE_o - KinE_p

    # Calculate Kinetic Energy of mirrored object.
    hat_Wo = Ka_w @ Wo