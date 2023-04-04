import copy
import pathlib

import numpy as np
from pytransform3d import rotations as rt, transformations as tr

from .utils import matrix_to_quat_xyzw, SE3_2_gen_coordinates


def draw_vector(pb, origin, vector, v_color, scale=1.0):
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

    vector_radius = max(0.0025, 0.0025 * v_norm * 4.0)
    vector_body_id = pb.createVisualShape(shapeType=pb.GEOM_CYLINDER, radius=vector_radius,
                                          length=v_norm,
                                          rgbaColor=v_color,
                                          specularColor=[0.4, .4, 0], )
    cone_path = pathlib.Path(__file__).parent.parent / "paper/stl_files/Cone.obj"
    vector_head_id = pb.createVisualShape(shapeType=pb.GEOM_MESH,
                                          fileName=str(cone_path),
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


def generate_rotating_view_gif(pb, cam_target_pose, cam_distance, save_path: pathlib.Path, n_frames='auto', file_name="animation",
                               anim_time=10, yaw_sin_amplitude=15):
    from moviepy.editor import ImageSequenceClip
    from tqdm import tqdm
    print(f"Generating rotating Gif animation with {n_frames} viewpoints")
    file_name = file_name.replace(".gif", '')
    n_frames = int(anim_time * 20) if isinstance(n_frames, str) else n_frames
    fps = int(n_frames / anim_time)  # Animation should take n seconds, compute frames per second
    # Adapted from https://colab.research.google.com/drive/1u6j7JOqM05vUUjpVp5VNk0pd8q-vqGlx#scrollTo=7tbOVtFp1_5K
    frames = []  # frames to create animated png
    yaw = 45
    yaw_update = 360 / n_frames
    freq = 1 / n_frames
    for r in tqdm(range(n_frames), desc="Capturing frames"):
        yaw += yaw_update
        pitch = -20.0 + yaw_sin_amplitude * np.sin((2 * np.pi * freq) * r)
        roll = 0
        upAxisIndex = 2
        camDistance = cam_distance
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
    clip.write_gif(file_path, fps=fps, loop=True)
    print(f"Animation saved to {file_path.absolute()}")


# Setup debug sliders
def setup_debug_sliders(pb, robot):
    for i, joint_name in enumerate(robot.joint_names):
        bullet_joint_id = robot.joint_aux_vars[joint_name].bullet_id
        joint_info = pb.getJointInfo(robot.robot_id, bullet_joint_id)
        joint_name = joint_info[1].decode("UTF-8")
        lower_limit, upper_limit = joint_info[8], joint_info[9]
        pb.addUserDebugParameter(paramName=f"{joint_name}_{i}", rangeMin=lower_limit, rangeMax=upper_limit, startValue=0.0)

# Read param values
def listen_update_robot_sliders(pb, robot):
    import time
    while True:
        pb_q = np.zeros(pb.getNumJoints(robot.robot_id))
        for i, joint_name in enumerate(robot.joint_names):
            bullet_joint_id = robot.joint_aux_vars[joint_name].bullet_id
            pb_q[bullet_joint_id] = pb.readUserDebugParameter(itemUniqueId=i)
            pb.resetJointState(robot.robot_id, bullet_joint_id, pb_q[bullet_joint_id])
        time.sleep(0.1)

def tint_robot(pb, robot):
    robot_color = [0.054, 0.415, 0.505, 1.0]
    FL_leg_color = [0.698, 0.376, 0.082, 1.0]
    FR_leg_color = [0.260, 0.263, 0.263, 1.0]
    HL_leg_color = [0.800, 0.480, 0.000, 1.0]
    HR_leg_color = [0.710, 0.703, 0.703, 1.0]
    endeff_color = [0, 0, 0, 1]
    for i in range(pb.getNumJoints(robot.robot_id)):
        link_name = pb.getJointInfo(robot.robot_id, i)[12].decode("UTF-8")
        joint_name = pb.getJointInfo(robot.robot_id, i)[1].decode("UTF-8")
        if link_name in robot.endeff_names or (joint_name in robot.endeff_names):
            color = endeff_color
        elif np.any([s in joint_name.lower() for s in ["fl_", "left"]]):
            color = FL_leg_color
        elif np.any([s in joint_name.lower() for s in ["fr_", "right"]]):
            color = FR_leg_color
        elif np.any([s in joint_name.lower() for s in ["rl_", "hl_", "left"]]):
            color = HL_leg_color
        elif np.any([s in joint_name.lower() for s in ["rr_", "hr_", "right"]]):
            color = HR_leg_color
        else:
            color = robot_color
        pb.changeVisualShape(objectUniqueId=robot.robot_id, linkIndex=i, rgbaColor=color, specularColor=[0, 0, 0])
    pb.changeVisualShape(objectUniqueId=robot.robot_id, linkIndex=-1, rgbaColor=robot_color, specularColor=[0, 0, 0])


def display_robots_and_vectors(pb, robot, base_confs, Gq_js, Gdq_js, Ghg, Ghg_pin, forces, forces_points, surface_normals,
                               GX_g_bar, offset=1.5):
    """
    Plot side by side robots with different configutations, CoM momentums and expected CoM after an action g
    """
    # pb.resetSimulation()

    # Optional: Display origin.
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
    for i in range(0, len(Gq_js)):
        q_js, dq_js, XB_w, hg_B, ghg_B, rho_X_gbar = Gq_js[i], Gdq_js[i], base_confs[i], Ghg[i], Ghg_pin[i], GX_g_bar[i]
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

    print("a")
