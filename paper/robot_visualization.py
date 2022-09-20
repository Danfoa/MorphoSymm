import copy
import os
import pathlib
import time

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from pytransform3d import rotations as rt
from pytransform3d import transformations as tr
from scipy.sparse import issparse

from utils.robot_utils import get_robot_params
from utils.utils import configure_bullet_simulation

# from pytransform3d.transform_manager import TransformManager

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
offset = 0.8


def draw_vector(origin, vector, v_color, scale=1.0):
    """
    Shitty pybullet doesn't allow you to draw vectors, so I had to write up the code for it. Inefficient but does the
    job
    """
    linewidth = 4
    if np.linalg.norm(vector) == 0:
        return None
    pb.addUserDebugLine(lineFromXYZ=origin, lineToXYZ=origin + vector * scale,
                        lineColorRGB=v_color[:3], lineWidth=linewidth, lifeTime=0)

    v_norm = np.linalg.norm(vector) * scale

    vector_radius = 0.0025
    vector_body_id = pb.createVisualShape(shapeType=pb.GEOM_CYLINDER, radius=vector_radius,
                                          length=v_norm,
                                          rgbaColor=v_color,
                                          specularColor=[0.4, .4, 0], )
    vector_head_id = pb.createVisualShape(shapeType=pb.GEOM_MESH,
                                          fileName="paper/stl_files/Cone.obj",
                                          rgbaColor=v_color,
                                          specularColor=[0.4, .4, 0],
                                          meshScale=np.array([1, 1, 1]) * (vector_radius * 2 * 30))
    # Get rotation where the x axis is aligned with the vector orientation
    v2 = np.random.rand(3)
    v3 = np.cross(vector, v2)
    R = rt.matrix_from_two_vectors(a=vector, b=v3)[:, [1, 2, 0]]
    body_origin = origin + (vector * scale / 2.)
    R_head = rt.active_matrix_from_intrinsic_euler_xyz([np.deg2rad(90), np.deg2rad(0), np.deg2rad(0)])
    vector_id = pb.createMultiBody(baseMass=1,
                                   baseInertialFramePosition=[0, 0, 0],
                                   baseCollisionShapeIndex=vector_body_id,
                                   baseVisualShapeIndex=vector_body_id,
                                   basePosition=body_origin,
                                   baseOrientation=matrix_to_quat_xyzw(R),
                                   linkMasses=[0.01],
                                   linkVisualShapeIndices=[vector_head_id],
                                   linkCollisionShapeIndices=[vector_head_id],
                                   linkPositions=[np.array([0, 0, v_norm / 2.])],
                                   linkOrientations=[matrix_to_quat_xyzw(R_head)],
                                   linkInertialFramePositions=[np.array([0, 0, v_norm / 2.])],
                                   linkInertialFrameOrientations=[matrix_to_quat_xyzw(R_head)],
                                   linkParentIndices=[0],
                                   linkJointTypes=[pb.JOINT_FIXED],
                                   linkJointAxis=[(1, 0, 0)])
    return vector_id


def display_robots_and_vectors(pb, robot, base_confs, qs, dqs, hgs, ghgs, forces, forces_points, space_transformations):
    """
    Plot side by side robots with different configutations, CoM momentums and expected CoM after an action g
    """
    pb.resetSimulation()

    # Display origin.
    draw_vector(np.zeros(3), np.asarray([.1, 0, 0]), v_color=[1, 0, 0, 1])
    draw_vector(np.zeros(3), np.asarray([0, .1, 0]), v_color=[0, 1, 0, 1])
    draw_vector(np.zeros(3), np.asarray([0, 0, .1]), v_color=[0, 0, 1, 1])
    
    plane_height = 0.05
    plane_size = (0.01, 0.25, 0.1)
    plot_reflection_plane(pb, R=rt.matrix_from_two_vectors(a=[0, 1, 0], b=[1, 0, 0]), p=[0.0, offset, plane_height],
                          color=np.array([242, 242, 242, 40]) / 256., size=plane_size)
    plot_reflection_plane(pb, R=rt.matrix_from_two_vectors(a=[0, 1, 0], b=[1, 0, 0]), p=[2 * offset, offset, plane_height],
                          color=np.array([242, 242, 242, 40]) / 256., size=plane_size)
    plot_reflection_plane(pb, R=rt.matrix_from_two_vectors(a=[1, 0, 0], b=[0, 1, 0]), p=[offset, 0.0, plane_height],
                          color=np.array([255, 230, 230, 40]) / 256., size=plane_size)
    plot_reflection_plane(pb, R=rt.matrix_from_two_vectors(a=[1, 0, 0], b=[0, 1, 0]), p=[offset, 2 * offset, plane_height],
                          color=np.array([255, 230, 230, 40]) / 256., size=plane_size)

    robots = [robot]
    com_pos = None
    for i in range(0, len(qs)):
        q_js, dq_js, XB_w, hg_B, ghg_B, rho_X_gbar = qs[i], dqs[i], base_confs[i], hgs[i], ghgs[i], \
                                                     space_transformations[i]
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
        lin_com_mom_id = draw_vector(origin=gcom_pos, vector=RB_w @ ghg_B[:3],
                                     v_color=np.array([255, 153, 0, 255]) / 255.,
                                     scale=0.1 / np.linalg.norm(ghg_B[:3]))
        ang_com_mom_id = draw_vector(origin=gcom_pos, vector=RB_w @ ghg_B[3:],
                                     v_color=np.array([136, 204, 0, 255]) / 255.,
                                     scale=0.1 / np.linalg.norm(ghg_B[3:]))

        # Draw forces and contact planes
        force_color = (0.590, 0.153, 0.510, 1.0)
        for force_orbit, rf_orbit in zip(forces, forces_points):
            draw_vector(origin=rf_orbit[i], vector=force_orbit[i], v_color=force_color)
            body_id = pb.createVisualShape(shapeType=pb.GEOM_BOX, halfExtents=[0.1, 0.1, 0.01],
                                           rgbaColor=np.array([235, 255, 255, 150]) / 255.)
            mb = pb.createMultiBody(baseMass=1,
                                    baseInertialFramePosition=[0, 0, 0],
                                    baseCollisionShapeIndex=body_id,
                                    baseVisualShapeIndex=body_id,
                                    basePosition=rf_orbit[i] - np.array([0, 0, 0.03]),
                                    baseOrientation=matrix_to_quat_xyzw(np.eye(3)))
        # Draw Base orientation
        draw_vector(origin=tB_w + RB_w @ np.array((0.06, 0, 0.03)), vector=RB_w[:, 0], v_color=[1, 1, 1, 1], scale=0.05)
        # draw_vector(origin=tB_w + RB_w @ np.array((0, 0, 0.05)), vector=RB_w[:, 1], v_color=[0, 1, 0, 1], scale=0.05)
        # draw_vector(origin=tB_w + RB_w @ np.array((0, 0, 0.05)), vector=RB_w[:, 2], v_color=[0, 0, 1, 1], scale=0.05)

    print("a")


def reflex_matrix(a):
    assert np.any([d == 3 for d in a.shape]), "We expect a 3D vector"
    assert a.shape[1] == 1
    d = a.shape[0]
    return np.eye(d) - 2 * ((a @ a.T) / (a.T @ a))


def homogenousMatrix(R: np.ndarray, T=np.zeros(3)):
    X = np.zeros((4, 4), dtype=R.dtype)
    X[3, 3] = 1
    X[:, 3] = T
    X[:3, :3] = R
    return X


def reflection_transformation(vnorm, point_in_plane):
    """Generates the Homogenous trasformation matrix of a reflection"""
    if vnorm.ndim == 1:
        vnorm = np.expand_dims(vnorm, axis=1)
    KA = reflex_matrix(vnorm)
    # The plane position is defined as a function of a point in the plane.
    tKA = np.squeeze(-2 * vnorm * (-point_in_plane.dot(vnorm)))
    TK = tr.transform_from(R=KA, p=tKA)
    return TK


def matrix_to_quat_xyzw(R):
    assert R.shape == (3, 3)
    return rt.quaternion_xyzw_from_wxyz(rt.quaternion_from_matrix(R))


def SE3_2_gen_coordinates(X):
    assert X.shape == (4, 4)
    pos = X[:3, 3]
    quat = matrix_to_quat_xyzw(X[:3, :3])
    return np.concatenate((pos, quat))


def plot_reflection_plane(pb, R, p, color, size=(0.01, 0.25, 0.25)):
    body_id = pb.createVisualShape(shapeType=pb.GEOM_BOX, halfExtents=size,
                                   rgbaColor=color)
    plane_id = pb.createMultiBody(baseMass=1,
                                  baseInertialFramePosition=[0, 0, 0],
                                  baseCollisionShapeIndex=body_id,
                                  baseVisualShapeIndex=body_id,
                                  basePosition=p,
                                  baseOrientation=matrix_to_quat_xyzw(R))
    return plane_id


def generate_rotating_view_gif(pb, cam_target_pose, save_path:pathlib.Path, n_frames='auto', file_name="animation",
                               anim_time=10):
    from moviepy.editor import ImageSequenceClip
    from tqdm import tqdm
    print(f"Generating rotating Gif animation with {n_frames} viewpoints")
    file_name = file_name.replace(".gif", '')
    n_frames = int(anim_time*20) if isinstance(n_frames,str) else n_frames
    fps = int(n_frames/anim_time)  # Animation should take n seconds, compute frames per second
    # Adapted from https://colab.research.google.com/drive/1u6j7JOqM05vUUjpVp5VNk0pd8q-vqGlx#scrollTo=7tbOVtFp1_5K
    frames = []  # frames to create animated png
    yaw = 45
    yaw_update = 360/n_frames
    freq = 1/n_frames
    for r in tqdm(range(n_frames), desc="Capturing frames"):
        yaw += yaw_update
        pitch = -20.0 + 15*np.sin((2*np.pi*freq) *r)
        roll = 0
        upAxisIndex = 2
        camDistance = 3
        # int(2 * 1024), int(2 * 812)
        pixelWidth = 1024
        pixelHeight = 812
        nearPlane = 0.01
        farPlane = 100
        fov = 60
        viewMatrix = pb.computeViewMatrixFromYawPitchRoll(cam_target_pose, camDistance, yaw, pitch, roll, upAxisIndex)
        aspect = pixelWidth / pixelHeight
        projectionMatrix = pb.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)

        img_arr = pb.getCameraImage(pixelWidth, pixelHeight, viewMatrix, projectionMatrix, shadow=1,
                                    lightDirection=[0, 0, 0.5], lightDistance=1.0, renderer=pb.ER_TINY_RENDERER)
        w = img_arr[0]  # width of the image, in pixels
        h = img_arr[1]  # height of the image, in pixels
        rgb = img_arr[2]  # color data RGB
        np_img_arr = np.reshape(rgb, (h, w, 4))
        frame = np_img_arr[:, :, :3]
        frames.append(frame)

    file_path = save_path / f'{file_name}.gif'
    clip = ImageSequenceClip(list(frames), fps=fps)
    clip.write_gif(file_path, fps=fps, loop=0)
    print(f"Animation saved to {file_path.absolute()}")

if __name__ == "__main__":

    robot_name = "Solo12"  # "Solo12"
    robot, Gin, Gout, Gin_model, Gout_model = get_robot_params(robot_name)

    pb = configure_bullet_simulation(gui=True)
    robot.configure_bullet_simulation(pb, world=None)

    # Notation: [Variable]_[reference frame of variable] e.g, R_w : Rotation R in world w coordinates.
    # R: Rotation 3x3, T: Translation: 3x1, K: Reflection 3x3, X: Homogenous Matrix 4x4

    q, dq = robot.get_init_config(random=True)
    #       |-----FL--------|------FR---------||-----HL--------|------HR--------|
    q[7:] = [0.1, 1.65, -1.9, -0.45, 0.7, -1.0, 0.35, 0.7, -1.0, -0.2, 1.8, -1.6]

    # Specify the configuration of the original robot base
    # Keep a "null" notation for computation of hb in Base reference frame.
    rB_w = np.array(q[:3])
    # RB_w = rt.active_matrix_from_intrinsic_euler_xyz([np.deg2rad(8), np.deg2rad(-15), np.deg2rad(0)])
    RB_w = rt.active_matrix_from_intrinsic_euler_xyz([np.deg2rad(-10), np.deg2rad(-5), np.deg2rad(-5)])
    XB_w = tr.transform_from(R=RB_w, p=rB_w)
    XBnull_w = tr.transform_from(R=np.eye(3), p=rB_w)

    q[3:7] = SE3_2_gen_coordinates(XB_w)[3:7]
    robot.reset_state(q, dq)

    # x: is the concatenation of q and dq. That is the state vector.
    x = np.concatenate((q[7:], dq[6:]))
    x = x.astype(np.float64)
    # Compute the COM momentum in base B coordinates
    hg_B = robot.pinocchio_robot.centroidalMomentum(q=np.concatenate((SE3_2_gen_coordinates(XBnull_w), q[7:])),
                                                    v=np.concatenate((np.zeros(6), dq[6:]))).np.astype(np.float64)
    # For visualization purposes we make the vectors fixed and clear from the top view
    hg_B = np.array([0.1, 0.1, 0.0, -0.0, -0.2, -0.2])

    # Define a ground reaction force on two of the legs:
    f1_w, f2_w = np.array([0.1, -0.1, 0.1]), np.array([-0.1, 0.1, 0.15])
    # Points of application of forces in Base coordinates on the feets of the robot.
    R_B2w = RB_w
    R_w2B = np.linalg.inv(R_B2w)
    rf1_w = np.array(pb.getLinkState(robot.robot_id, 7)[0])
    rf2_w = np.array(pb.getLinkState(robot.robot_id, 11)[0])
    # _________________________________________________________________________________________________________________
    # Define the list holding the orbits of all the vectors in system and joint spaces.
    x_orbit, hg_B_orbit, hgpin_B, XB_w_orbit = [x], [hg_B], [hg_B], [XB_w]
    f1_w_orbit, f2_w_orbit, r1_w_orbit, r2_w_orbit = [f1_w], [f2_w], [rf1_w], [rf2_w]
    # The base of the robot has two planes of symmetry Sagittal and Traversal
    Ksag_B = reflex_matrix(np.array([[0, 1, 0]]).T)
    Ktra_B = reflex_matrix(np.array([[1, 0, 0]]).T)
    body_reflections = [np.eye(3), Ksag_B, Ktra_B, Ksag_B @ Ktra_B]

    # The symmetry group can be understood as 3D space reflections/rotations
    rho_X_gbar_x = reflection_transformation(vnorm=np.array([1, 0, 0]), point_in_plane=np.array([1, 0, 0]) * offset)
    rho_X_gbar_y = reflection_transformation(vnorm=np.array([0, 1, 0]), point_in_plane=np.array([0, 1, 0]) * offset)
    Rg_xy = rho_X_gbar_x @ rho_X_gbar_y
    space_reflections = [tr.transform_from(R=np.eye(3), p=np.zeros(3)), rho_X_gbar_y, rho_X_gbar_x, Rg_xy]

    # Get all possible group actions
    for rho_Q_g, rho_hg_g, KB_B, rho_X_gbar in zip(Gin.discrete_actions[1:], Gout.discrete_actions[1:],
                                                   body_reflections[1:], space_reflections[1:]):
        rho_Q_g, rho_hg_g = (rho_Q_g.todense(), rho_hg_g.todense()) if issparse(rho_Q_g) else (rho_Q_g, rho_hg_g)
        rho_Q_g, rho_hg_g = np.asarray(rho_Q_g), np.asarray(rho_hg_g)

        # Improper/unfeasible transformation
        Kg_w = rho_hg_g[:3, :3]

        # Get symmetric g.x=[g.q, g.dq], g.XB_w
        gx_w, ghg_B = rho_Q_g @ x, rho_hg_g @ hg_B
        x_orbit.append(gx_w)
        hg_B_orbit.append(ghg_B)
        gRB_w = Kg_w @ RB_w @ KB_B
        gtXB_w = (rho_X_gbar @ XB_w)[:3, 3]
        XB_w_orbit.append(tr.transform_from(R=gRB_w, p=gtXB_w))

        # Get symmetric versions of euclidean vectors and pseudovectors.
        gf1_w, gf2_w = rho_X_gbar[:3, :3] @ f1_w, rho_X_gbar[:3, :3] @ f2_w
        grf1_w, grf2_w = tr.transform(rho_X_gbar, tr.vector_to_point(rf1_w), strict_check=False)[:3], \
                         tr.transform(rho_X_gbar, tr.vector_to_point(rf2_w), strict_check=False)[:3]
        f1_w_orbit.append(gf1_w[:3]), f2_w_orbit.append(gf2_w[:3]), r1_w_orbit.append(grf1_w), r2_w_orbit.append(grf2_w)

        gy_true = robot.pinocchio_robot.centroidalMomentum(
            q=np.concatenate((SE3_2_gen_coordinates(XBnull_w), np.split(gx_w, 2)[0])),
            v=np.concatenate((np.zeros(6), np.split(gx_w, 2)[1]))).np.astype(np.float64)
        hgpin_B.append(gy_true)

    splited_orbits = [np.split(x, 2) for x in x_orbit]
    qs, dqs = [x[0] for x in splited_orbits], [x[1] for x in splited_orbits]

    display_robots_and_vectors(pb, robot, base_confs=XB_w_orbit, qs=qs, dqs=dqs, hgs=hgpin_B, ghgs=hg_B_orbit,
                               forces=[f1_w_orbit, f2_w_orbit], forces_points=[r1_w_orbit, r2_w_orbit],
                               space_transformations=space_reflections)

    # while True:
    #     time.sleep(0.5)

    # frontal_view_matrix = pb.computeViewMatrixFromYawPitchRoll(
    #     cameraTargetPosition=XB_w[:3, 3] + + np.array([0, offset, 0]),
    #     distance=1.3,
    #     upAxisIndex=2,
    #     roll=0,
    #     pitch=-35,
    #     yaw=-90)
    # frontal_projection_matrix = pb.computeProjectionMatrixFOV(fov=90, aspect=1.0, nearVal=0.1, farVal=5.0)
    #
    # top_view_matrix = pb.computeViewMatrixFromYawPitchRoll(
    #     cameraTargetPosition=XB_w[:3, 3] + np.array([offset, offset, 0]),
    #     distance=1.20,
    #     upAxisIndex=2,
    #     roll=0,
    #     pitch=-90,
    #     yaw=0)
    # top_projection_matrix = pb.computeProjectionMatrixFOV(fov=90, aspect=1.2, nearVal=0.1, farVal=5.0)
    #
    # single_view_matrix = pb.computeViewMatrixFromYawPitchRoll(
    #     cameraTargetPosition=XB_w[:3, 3] + np.array([0, 0, -0.15]),
    #     distance=0.8,
    #     upAxisIndex=2,
    #     roll=0,
    #     pitch=-20,
    #     yaw=-170)
    # single_projection_matrix = pb.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.1, farVal=1.5)
    #
    # back_img_arr = pb.getCameraImage(int(2 * 1024), int(2 * 812), viewMatrix=frontal_view_matrix,
    #                                  projectionMatrix=frontal_projection_matrix, shadow=1,
    #                                  lightDirection=[0, 0, 0.5], lightDistance=1.0,
    #                                  renderer=pb.ER_TINY_RENDERER)[2]
    #
    # top_img_arr = pb.getCameraImage(width=int(2 * 1024), height=int(2 * 1024), viewMatrix=top_view_matrix,
    #                                 projectionMatrix=top_projection_matrix, shadow=1,
    #                                 lightDirection=[0, 0, 0.5], lightDistance=1.0,
    #                                 renderer=pb.ER_TINY_RENDERER)[2]
    #
    # single_img_arr = pb.getCameraImage(width=int(2 * 1024), height=int(2 * 1024), viewMatrix=single_view_matrix,
    #                                    projectionMatrix=single_projection_matrix, shadow=1,
    #                                    lightDirection=[0, 0, 0.5], lightDistance=1.0,
    #                                    # lightAmbientCoeff=0.5,
    #                                    renderer=pb.ER_TINY_RENDERER)[2]
    #
    # save_path = pathlib.Path("paper/images")
    # save_path.mkdir(exist_ok=True)
    #
    # print(back_img_arr.shape)

    # traversal_plane_id = plot_reflection_plane(pb, XB_w[:3,:3], p=XB_w[:3,3], size=(0.002,0.2,0.1),
    #                                           color=np.array([255,0,0,50])/255.)
    # sagittal_plane_id = plot_reflection_plane(pb, XB_w[:3,:3], p=XB_w[:3,3], size=(0.2,0.002,0.1),
    #                                            color=np.array([0, 0, 255, 50]) / 255.)
    # plt.figure()
    # plt.imshow(single_img_arr)
    # plt.axis('off')
    # plt.show()
    # im = Image.fromarray(single_img_arr)
    # im.save(save_path / "K4_solo_single_view.png")

    # pb.removeBody(sagittal_plane_id)
    # pb.removeBody(traversal_plane_id)

    # generate_rotating_view_gif(pb, cam_target_pose=[offset,offset, 0], save_path=save_path,
    #                            file_name="solo-K4-symmetries_anim_static")

    while True:
        time.sleep(0.5)
    #
    # plt.figure()
    # plt.imshow(back_img_arr)
    # # plt.axis('off')
    # plt.show()
    # im = Image.fromarray(back_img_arr)
    # im.save(save_path / "K4_solo_back_view.png")
    #
    # plt.figure()
    # plt.imshow(top_img_arr)
    # plt.show()
    # im = Image.fromarray(top_img_arr)
    # im.save(save_path / "K4_solo_top_view.png")

    # while True:
    #     time.sleep(0.5)
