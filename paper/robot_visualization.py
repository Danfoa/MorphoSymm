import copy
import os
import pathlib
import time

import numpy as np
from PIL import Image
from pytransform3d import rotations as rt
from pytransform3d import transformations as tr

from utils.pybullet_visual_utils import draw_vector, plot_reflection_plane, generate_rotating_view_gif
from utils.robot_utils import get_robot_params
from utils.utils import configure_bullet_simulation, matrix_to_quat_xyzw, SE3_2_gen_coordinates, quat_xyzw_to_matrix

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



def display_robots_and_vectors(pb, robot, base_confs, Gq, Gdq, Ghg, Ghg_pin, forces, forces_points, surface_normals,
                               GX_g_bar, offset=1.5):
    """
    Plot side by side robots with different configutations, CoM momentums and expected CoM after an action g
    """
    pb.resetSimulation()

    # Display origin.
    # draw_vector(pb, np.zeros(3), np.asarray([.1, 0, 0]), v_color=[1, 0, 0, 1])
    # draw_vector(pb, np.zeros(3), np.asarray([0, .1, 0]), v_color=[0, 1, 0, 1])
    # draw_vector(pb, np.zeros(3), np.asarray([0, 0, .1]), v_color=[0, 0, 1, 1])

    plane_height = robot.hip_height / 4.0
    plane_size = (0.01, robot.hip_height / 2,  robot.hip_height / 4)

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
        grobot = robot if i == 0 else copy.copy(robot)
        grobot.configure_bullet_simulation(pb, world=None)
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


if __name__ == "__main__":

    robot_name = "Bolt"  # "Solo12"
    # robot_name = "Bolt"  # "Solo12"

    robot_name = robot_name.lower()
    robot, rep_data_in, rep_data_out, rep_model_in, rep_model_out, rep_E3, rep_qj = get_robot_params(robot_name)

    offset = 1.5
    cam_target_pose = [0, offset/2, robot.hip_height]
    if robot_name == "atlas":
        feet1, feet2 = 23, 29
        offset = 2.5
        cam_target_pose = [0, offset/2, robot.hip_height]
        rep_E3_offsets = [[0, offset, 0]]
    elif robot_name == "solo12":
        feet1, feet2 = 7, 11
        cam_target_pose = [offset/2, offset/2, robot.hip_height]
        rep_E3_offsets = [[0, offset, 0], [offset, 0, 0], [offset, offset, 0]]
    elif robot_name == "bolt":
        feet1, feet2 = 3, 7
        rep_E3_offsets = [[0, offset, 0]]
    else:
        raise NotImplementedError(f"Robot {robot_name} not implemented")

    pb = configure_bullet_simulation(gui=True)
    robot.configure_bullet_simulation(pb, world=None)

    # Notation: [Variable]_[reference frame of variable] e.g, R_w : Rotation R in world w coordinates.
    # R: Rotation 3x3, T: Translation: 3x1, K: Reflection 3x3, X: Homogenous Matrix 4x4

    # Get initial random configuration of the system
    q, dq = robot.get_init_config(random=True)
    #       |-----FL--------|------FR---------||-----HL--------|------HR--------|
    # q[7:] = [0.1, 1.65, -1.9, -0.45, 0.7, -1.0, 0.35, 0.7, -1.0, -0.2, 1.8, -1.6]

    # Specify the configuration of the original robot base
    # Keep a "null" notation for computation of hb in Base reference frame.
    rB_w = np.array(q[:3])
    RB_w = rt.active_matrix_from_intrinsic_euler_xyz(q[3:6])
    # RB_w = rt.active_matrix_from_intrinsic_euler_xyz([np.deg2rad(-10), np.deg2rad(-5), np.deg2rad(-5)])
    XB_w = tr.transform_from(R=RB_w, p=rB_w)
    XBnull_w = tr.transform_from(R=np.eye(3), p=rB_w)

    q[3:7] = SE3_2_gen_coordinates(XB_w)[3:7]
    robot.reset_state(q, dq)

    # x: is the concatenation of q and dq. That is the state vector.
    x = np.concatenate((q[7:], dq[6:]))
    x = x.astype(np.float64)
    # Compute the robot COM momentum in base `B` coordinates
    hg_B = robot.pinocchio_robot.centroidalMomentum(q=np.concatenate((SE3_2_gen_coordinates(XBnull_w), q[7:])),
                                                    v=np.concatenate((np.zeros(6), dq[6:]))).np.astype(np.float64)

    # TODO: (remove) For visualization purposes we make the vectors fixed and clear from the top view
    hg_B = np.array([0.1, 0.1, 0.0, -0.0, -0.2, -0.2])

    # Define a ground-reaction/contact force on two of the system links:
    # Points of application of forces in Base coordinates on the feets of the robot.
    R_B2w = RB_w
    R_w2B = np.linalg.inv(R_B2w)
    rf1_w, quatf1_w = (np.array(x) for x in pb.getLinkState(robot.robot_id, feet1)[0:2])
    rf2_w, quatf2_w = (np.array(x) for x in pb.getLinkState(robot.robot_id, feet2)[0:2])
    Rf1_w, Rf2_w = quat_xyzw_to_matrix(quatf1_w), quat_xyzw_to_matrix(quatf2_w)
    if "solo" in robot_name.lower() or "bolt" in robot_name.lower():
        Rf1_w = np.eye(3)
        Rf2_w = np.eye(3)
    f1_w = Rf1_w[:, 2] + np.random.random(3)  # Rf_w[:, 2] := Surface normal
    f2_w = Rf2_w[:, 2] + np.random.random(3)
    # For visualization purposes we make the forces proportional to the robot height
    f1_w = f1_w / np.linalg.norm(f1_w) * robot.hip_height * .5
    f2_w = f2_w / np.linalg.norm(f2_w) * robot.hip_height * .5

    # _________________________________________________________________________________________________________________
    # Define the list holding the orbits of all the proprioceptive and exteroceptive measurements
    Gx, Ghg_B, Gy_B, GXB_w, GX_g_bar = [x], [hg_B], [hg_B], [XB_w], [tr.transform_from(R=np.eye(3), p=np.zeros(3))]
    Gf1_w, Gf2_w, Gr1_w, Gr2_w, GRf1_w, GRf2_w = [f1_w], [f2_w], [rf1_w], [rf2_w], [Rf1_w], [Rf2_w]

    for rho_x_g, rho_y_g, rho_qj_g, rho_E3_g_bar, rho_E3_r in zip(rep_data_in.G.discrete_actions[1:],
                                                                  rep_data_out.G.discrete_actions[1:],
                                                                  rep_qj.G.discrete_actions[1:],
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


    # save_path = pathlib.Path("paper")
    # save_path.mkdir(exist_ok=True)
    #
    # frontal_view_matrix = pb.computeViewMatrixFromYawPitchRoll(
    #     cameraTargetPosition=cam_target_pose, distance=offset, upAxisIndex=2, roll=0, pitch=-0, yaw=90)
    # frontal_projection_matrix = pb.computeProjectionMatrixFOV(fov=80, aspect=1.0, nearVal=0.1, farVal=5.0)
    #
    # front_img_arr = pb.getCameraImage(int(2*812), int(2*812), viewMatrix=frontal_view_matrix,
    #                                   projectionMatrix=frontal_projection_matrix, shadow=1,
    #                                   lightDirection=[0, 0, 0.5], lightDistance=1.0,
    #                                   renderer=pb.ER_TINY_RENDERER)[2]
    # im = Image.fromarray(front_img_arr)
    # im.save(save_path / f"images/{robot_name}/{robot_name}_front_symmetry_view.png")

    # generate_rotating_view_gif(pb, cam_target_pose=cam_target_pose, cam_distance=offset * 2,
    #                            save_path=save_path / 'animations', yaw_sin_amplitude=15,
    #                            file_name=f"{robot_name}-symmetries_anim_static")
    # print("Done enjoy your gif :). I hope you learned something new")

    for _ in range(500):
        time.sleep(0.5)

    pb.disconnect()