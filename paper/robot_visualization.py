import copy
import os
import pathlib
import time

import hydra
import numpy as np
from PIL import Image
from omegaconf import DictConfig
from pytransform3d import rotations as rt
from pytransform3d import transformations as tr

from groups.SparseRepresentation import SparseRepE3
from utils.pybullet_visual_utils import draw_vector, plot_reflection_plane, generate_rotating_view_gif, tint_robot
from utils.robot_utils import get_robot_params, load_robot_and_symmetries
from utils.utils import configure_bullet_simulation, matrix_to_quat_xyzw, SE3_2_gen_coordinates, quat_xyzw_to_SO3

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def display_robots_and_vectors(pb, robot, base_confs, Gq, Gdq, Ghg, Ghg_pin, forces, forces_points, surface_normals,
                               GX_g_bar, offset=1.5):
    """
    Plot side by side robots with different configutations, CoM momentums and expected CoM after an action g
    """
    # pb.resetSimulation()

    # Display origin.
    # draw_vector(pb, np.zeros(3), np.asarray([.1, 0, 0]), v_color=[1, 0, 0, 1])
    # draw_vector(pb, np.zeros(3), np.asarray([0, .1, 0]), v_color=[0, 1, 0, 1])
    # draw_vector(pb, np.zeros(3), np.asarray([0, 0, .1]), v_color=[0, 0, 1, 1])

    plane_height = robot.hip_height / 4.0
    plane_size = (0.01, robot.hip_height / 2, robot.hip_height / 4)

    # Sagittal plane
    X_g_sagittal = GX_g_bar[1]
    plot_reflection_plane(pb, R=rt.matrix_from_two_vectors(a=[0, 1, 0], b=[1, 0, 0]),
                          p=X_g_sagittal[:3, 3] / 2 + [0.0, 0.0, plane_height],
                          color=np.array([242, 242, 242, 40]) / 256., size=plane_size)
    if len(GX_g_bar) > 2:
        plot_reflection_plane(pb, R=rt.matrix_from_two_vectors(a=[0, 1, 0], b=[1, 0, 0]),
                              p=X_g_sagittal[:3, 3] / 2 + [offset, 0.0, plane_height],
                              color=np.array([242, 242, 242, 40]) / 256., size=plane_size)
        X_g_trans = GX_g_bar[2]
        plot_reflection_plane(pb, R=rt.matrix_from_two_vectors(a=[1, 0, 0], b=[0, 1, 0]),
                              p=X_g_trans[:3, 3] / 2 + [0.0, 0.0, plane_height],
                              color=np.array([255, 230, 230, 40]) / 256., size=plane_size)
        plot_reflection_plane(pb, R=rt.matrix_from_two_vectors(a=[1, 0, 0], b=[0, 1, 0]),
                              p=X_g_trans[:3, 3] / 2 + [0.0, offset, plane_height],
                              color=np.array([255, 230, 230, 40]) / 256., size=plane_size)

    robots = [robot]
    com_pos = None
    for i in range(0, len(Gq)):
        q_js, dq_js, XB_w, hg_B, ghg_B, rho_X_gbar = Gq[i], Gdq[i], base_confs[i], Ghg[i], Ghg_pin[i], GX_g_bar[i]
        RB_w = XB_w[:3, :3]
        tB_w = XB_w[:3, 3]
        grobot = robot
        if i > 0:
            grobot = robot if i == 0 else copy.copy(robot)
            grobot.configure_bullet_simulation(pb, world=None)
            tint_robot(pb, grobot)
            robots.append(grobot)
        # Place robots in env
        base_q = SE3_2_gen_coordinates(XB_w)
        # Set positions:
        grobot.reset_state(np.concatenate((base_q, q_js)), np.concatenate((np.zeros(6), dq_js)))
        # Add small offset to COM for visualization.
        if com_pos is None:
            com_pos = robot.pinocchio_robot.com(q=np.concatenate((base_q, q_js))) + (RB_w @ np.array([-0.4, 0.5, 0.05]))

        gcom_pos = tr.transform(rho_X_gbar, tr.vector_to_point(com_pos), strict_check=False)[:3]
        # Draw COM momentum and COM location
        com_id = pb.createVisualShape(shapeType=pb.GEOM_SPHERE, radius=0.02,
                                      rgbaColor=np.array([10, 10, 10, 255]) / 255.)
        com_body_id = pb.createMultiBody(baseMass=1, baseVisualShapeIndex=com_id,
                                         basePosition=gcom_pos,
                                         baseOrientation=matrix_to_quat_xyzw(np.eye(3)))
        lin_com_mom_id = draw_vector(pb, origin=gcom_pos, vector=RB_w @ ghg_B[:3],
                                     v_color=np.array([255, 153, 0, 255]) / 255.,
                                     scale=(1 / np.linalg.norm(ghg_B[:3]) * robot.hip_height * .3))
        ang_com_mom_id = draw_vector(pb, origin=gcom_pos, vector=RB_w @ ghg_B[3:],
                                     v_color=np.array([136, 204, 0, 255]) / 255.,
                                     scale=(1 / np.linalg.norm(ghg_B[3:]) * robot.hip_height * .3))

        # Draw forces and contact planes
        force_color = (0.590, 0.153, 0.510, 1.0)
        for force_orbit, rf_orbit, GRf_w in zip(forces, forces_points, surface_normals):
            draw_vector(pb, origin=rf_orbit[i], vector=force_orbit[i], v_color=force_color)
            body_id = pb.createVisualShape(shapeType=pb.GEOM_BOX, halfExtents=[.2 * robot.hip_height,
                                                                               .2 * robot.hip_height,
                                                                               0.01],
                                           rgbaColor=np.array([235, 255, 255, 150]) / 255.)
            mb = pb.createMultiBody(baseMass=1,
                                    baseInertialFramePosition=[0, 0, 0],
                                    baseCollisionShapeIndex=body_id,
                                    baseVisualShapeIndex=body_id,
                                    basePosition=rf_orbit[i] - GRf_w[i] @ np.array([0, 0, 0.03]),
                                    baseOrientation=matrix_to_quat_xyzw(GRf_w[i]))
        # Draw Base orientation
        draw_vector(pb, origin=tB_w + RB_w @ np.array((0.06, 0, 0.03)), vector=RB_w[:, 0], v_color=[1, 1, 1, 1],
                    scale=0.05)
        # draw_vector(pb, origin=tB_w + RB_w @ np.array((0, 0, 0.05)), vector=RB_w[:, 1], v_color=[0, 1, 0, 1], scale=0.05)
        # draw_vector(pb, origin=tB_w + RB_w @ np.array((0, 0, 0.05)), vector=RB_w[:, 2], v_color=[0, 0, 1, 1], scale=0.05)

    print("a")


@hydra.main(config_path='../cfg/supervised', config_name='config_visualization')
def main(cfg: DictConfig):
    robot, rep_E3, rep_QJ = load_robot_and_symmetries(robot_cfg=cfg.robot, debug=cfg.debug)
    
    robot_cfg = cfg.robot
    # Representations acting on the linear `l` and angular `k` component of CoM momentum `h`
    rep_l = rep_E3
    rep_k = rep_E3.set_pseudovector(True)
    rep_h = rep_l + rep_k

    # Representation acting on x=[q, dq]
    rep_x = rep_QJ + rep_QJ

    # 3D animation visualization parameters
    offset = 3.3 * robot.hip_height
    if len(rep_E3.G.discrete_actions) == 2:
        cam_target_pose = [0, offset / 2, robot.hip_height/2]
        rep_E3_offsets = [[0, offset, 0]]
    elif len(rep_E3.G.discrete_actions) == 4:
        cam_target_pose = [offset / 2, offset / 2, robot.hip_height/2]
        rep_E3_offsets = [[0, offset, 0], [offset, 0, 0], [offset, offset, 0]]

    pb = configure_bullet_simulation(gui=cfg.gui, debug=cfg.debug)
    robot.configure_bullet_simulation(pb, world=None)
    tint_robot(pb, robot)
    # Notation: [Variable]_[reference frame of variable] e.g, R_w : Rotation R in world w coordinates.
    # R: Rotation 3x3, T: Translation: 3x1, K: Reflection 3x3, X: Homogenous Matrix 4x4

    # Get initial random configuration of the system
    q, dq = robot.get_init_config(random=True, angle_sweep=robot_cfg.angle_sweep)
    robot.reset_state(q, dq)

    # Specify the configuration of the original robot base
    # Keep a "null" notation for computation of hb in Base reference frame.
    rB_w = np.array(q[:3])
    RB_w = quat_xyzw_to_SO3(q[3:7])
    XB_w = tr.transform_from(R=RB_w, p=rB_w)
    XBnull_w = tr.transform_from(R=np.eye(3), p=rB_w)


    # x: is the concatenation of q and dq. That is the state vector.
    x = np.concatenate((q[7:], dq[6:]))
    x = x.astype(np.float64)
    # Compute the robot COM momentum in base `B` coordinates
    hg_B = robot.pinocchio_robot.centroidalMomentum(q=np.concatenate((SE3_2_gen_coordinates(XBnull_w), q[7:])),
                                                    v=np.concatenate((np.zeros(6), dq[6:]))).np.astype(np.float64)
    # Note: For visualization purposes we make the vectors fixed and clear from the top view
    hg_B = np.array([0.1, 0.1, 0.0, -0.0, -0.2, -0.2])

    # Define a ground-reaction/contact force on two of the system links:
    # Points of application of forces in Base coordinates on the feets of the robot.
    R_B2w = RB_w
    R_w2B = np.linalg.inv(R_B2w)

    end_effectors = np.random.choice(robot.bullet_ids_allowed_floor_contacts, 2, replace=False)
    rf1_w, quatf1_w = (np.array(x) for x in pb.getLinkState(robot.robot_id, end_effectors[0])[0:2])
    rf2_w, quatf2_w = (np.array(x) for x in pb.getLinkState(robot.robot_id, end_effectors[1])[0:2])
    Rf1_w, Rf2_w = quat_xyzw_to_SO3(quatf1_w), quat_xyzw_to_SO3(quatf2_w)
    if not np.any([s in robot.robot_name for s in ["atlas"]]):
        Rf1_w = np.eye(3)
        Rf2_w = np.eye(3)
    f1_w = Rf1_w[:, 2] + [2 * np.random.rand() - 1, 2 * np.random.rand() - 1,
                          np.random.rand()]  # Rf_w[:, 2] := Surface normal
    f2_w = Rf2_w[:, 2] + [2 * np.random.rand() - 1, 2 * np.random.rand() - 1, np.random.rand()]
    # For visualization purposes we make the forces proportional to the robot height
    f1_w = f1_w / np.linalg.norm(f1_w) * robot_cfg.hip_height * .4
    f2_w = f2_w / np.linalg.norm(f2_w) * robot_cfg.hip_height * .4

    # _________________________________________________________________________________________________________________
    # Define the list holding the orbits of all the proprioceptive and exteroceptive measurements
    Gx, Ghg_B, Gy_B, GXB_w, GX_g_bar = [x], [hg_B], [hg_B], [XB_w], [tr.transform_from(R=np.eye(3), p=np.zeros(3))]
    Gf1_w, Gf2_w, Gr1_w, Gr2_w, GRf1_w, GRf2_w = [f1_w], [f2_w], [rf1_w], [rf2_w], [Rf1_w], [Rf2_w]

    for rho_x_g, rho_y_g, rho_qj_g, rho_E3_g_bar, rho_E3_r in zip(rep_x.G.discrete_actions[1:],
                                                                  rep_h.G.discrete_actions[1:],
                                                                  rep_QJ.G.discrete_actions[1:],
                                                                  rep_E3.G.discrete_actions[1:],
                                                                  rep_E3_offsets):
        # This example assumes we are working with the CoM experiment
        # x = [q, dq]
        # y = [l, k] l: Linear CoM momentum and k: Angular CoM momentum in base coordinates

        R_g_bar = np.asarray(rho_E3_g_bar.todense())  # True rotation/reflection of space
        # Rg_det = np.linalg.det(Rg_bar)
        r_g_bar = np.asarray(rho_E3_r)  # Add position to the planes of reflection
        X_g_bar = tr.transform_from(R=R_g_bar, p=r_g_bar)
        GX_g_bar.append(X_g_bar)

        # Get symmetric g.x=[g.q, g.dq], g.XB_w
        gx_w, gy_B = rho_x_g @ x, rho_y_g @ hg_B
        Gx.append(gx_w), Ghg_B.append(gy_B)

        gRB_w = R_g_bar @ RB_w @ R_g_bar
        gtXB_w = tr.transform(X_g_bar, XB_w[:, 3], strict_check=False)[:3]  # Transform the base position
        GXB_w.append(tr.transform_from(R=gRB_w, p=gtXB_w))

        # Get symmetric versions of euclidean vectors. We could also add some pseudo-vectors e.g. torque
        gf1_w, gf2_w = R_g_bar @ f1_w, R_g_bar @ f2_w
        gr_f1_w, gr_f2_w = tr.transform(X_g_bar, tr.vector_to_point(rf1_w), strict_check=False)[:3], \
                           tr.transform(X_g_bar, tr.vector_to_point(rf2_w), strict_check=False)[:3]
        Gf1_w.append(gf1_w[:3]), Gf2_w.append(gf2_w[:3]), Gr1_w.append(gr_f1_w), Gr2_w.append(gr_f2_w)
        # The environment is theoretically truly reflected/rotated, but we use "square" symmetric "estimations" of
        # terrain surface normal, so we can pre and post multiply to get a valid orientation matrix for the solid
        GRf1_w.append(R_g_bar @ Rf1_w @ R_g_bar), GRf2_w.append(R_g_bar @ Rf2_w @ R_g_bar)

        gy_B_true = robot.pinocchio_robot.centroidalMomentum(
            q=np.concatenate((SE3_2_gen_coordinates(XBnull_w), np.split(gx_w, 2)[0])),
            v=np.concatenate((np.zeros(6), np.split(gx_w, 2)[1]))).np.astype(np.float64)
        Gy_B.append(gy_B_true)

    # Get all possible group actions
    splited_orbits = [np.split(x, 2) for x in Gx]
    Gq, Gdq = [x[0] for x in splited_orbits], [x[1] for x in splited_orbits]

    display_robots_and_vectors(pb, robot, base_confs=GXB_w, Gq=Gq, Gdq=Gdq, Ghg=Gy_B, Ghg_pin=Ghg_B,
                               forces=[Gf1_w, Gf2_w], forces_points=[Gr1_w, Gr2_w], surface_normals=[GRf1_w, GRf2_w],
                               GX_g_bar=GX_g_bar, offset=offset)

    save_path = pathlib.Path("paper")
    save_path.mkdir(exist_ok=True)

    if cfg.make_gif:
        cam_distance = offset * 1.8
        generate_rotating_view_gif(pb, cam_target_pose=cam_target_pose, cam_distance=cam_distance,
                                   save_path=save_path / 'animations',
                                   yaw_sin_amplitude=np.rad2deg(np.arctan(robot_cfg.hip_height*0.8 / cam_distance)),
                                   file_name=f"{robot.robot_name}-{rep_E3.G}-symmetries_anim_static")
        print("Done enjoy your gif :). I hope you learned something new")

    for _ in range(500):
        time.sleep(0.1)

    pb.disconnect()


if __name__ == '__main__':
    main()
