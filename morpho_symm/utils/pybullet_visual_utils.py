import copy
import pathlib
from typing import List, Optional

import numpy as np
import pybullet
from pybullet_utils.bullet_client import BulletClient
from pytransform3d import rotations as rt
from pytransform3d import transformations as tr
from tqdm import tqdm

from morpho_symm.robots.PinBulletWrapper import PinBulletWrapper
from morpho_symm.utils.algebra_utils import SE3_2_gen_coordinates, matrix_to_quat_xyzw, quat_xyzw_to_SO3


def draw_vector(pb, origin, vector, v_color, scale=1.0):
    """Shitty pybullet doesn't allow you to draw vectors, so I had to write up the code for it.

    Inefficient but does the job.
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
    import morpho_symm
    cone_path = pathlib.Path(morpho_symm.__file__).parent / "resources/stl_files/Cone.obj"
    vector_head_id = pb.createVisualShape(shapeType=pb.GEOM_MESH,
                                          fileName=str(cone_path),
                                          rgbaColor=v_color,
                                          specularColor=[0.4, .4, 0],
                                          meshScale=np.array([1, 1, 1]) * (vector_radius * 2 * 30))
    # Get rotation where the `x` axis is aligned with the vector orientation
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


def plot_reflection_plane(pb, R, p, color, size=(0.01, 0.25, 0.25), cylinder=False):
    """Plots a plane with a given rotation and position."""
    if not cylinder:
        body_id = pb.createVisualShape(shapeType=pb.GEOM_BOX, halfExtents=size, rgbaColor=color)
    else:
        body_id = pb.createVisualShape(shapeType=pb.GEOM_CYLINDER, radius=size[0], length=size[1], rgbaColor=color)

    plane_id = pb.createMultiBody(baseMass=1,
                                  baseInertialFramePosition=[0, 0, 0],
                                  baseCollisionShapeIndex=body_id,
                                  baseVisualShapeIndex=body_id,
                                  basePosition=p,
                                  baseOrientation=matrix_to_quat_xyzw(R))
    return plane_id


def render_orbiting_animation(
        pb, cam_target_pose, cam_distance, save_path: pathlib.Path, fps=20, file_name="animation", periods=1,
        anim_time=10, pitch_sin_amplitude=15, init_roll_pitch_yaw=(0, -20, 45), invert_roll=False, gen_gif=True,
        gen_imgs=True):
    """Renders an orbiting animation around a fix target camera position."""
    n_frames = anim_time * fps
    render_width, render_height, fov, shadow = (812, 812, 60, True) if gen_gif else (3024, 3024, 40, False)
    print(f"Generating rotating Gif animation with {n_frames} viewpoints")

    roll0, pitch0, yaw0 = init_roll_pitch_yaw
    freq = 1 / (periods * n_frames)

    # Define camera trajectory in polar coordinates with a constant radius.
    t = np.asarray(range(n_frames))
    # We assume pitch will do a full period of a sine wave
    pitch = -np.asarray((pitch0 + pitch_sin_amplitude * np.sin((2 * np.pi * freq) * t)))
    yaw = np.linspace(yaw0, (360 + yaw0) * periods, n_frames)
    roll = np.ones_like(t) * roll0

    light_distance, light_directions = 4, (0.5, 0.5, 1)
    frames = render_camera_trajectory(pb, pitch, roll, yaw, n_frames, cam_distance, cam_target_pose,
                                      light_direction=light_directions, light_distance=light_distance,
                                      render_width=render_width, render_height=render_height, fov=fov, shadow=shadow
                                      )[:-1]

    if invert_roll:
        # Generate linear transition of roll from 0 to 180 degrees
        # Define transition frames from 0 to 180 degrees
        rotation_frames = n_frames // 3
        t_rot = np.asarray(range(rotation_frames))
        roll_rot = np.linspace(roll0, roll0 + 180, rotation_frames)
        yaw_rot = np.ones_like(t_rot) * yaw[-1]
        pitch_rot = -np.abs(np.linspace(pitch[-1], -pitch[-1], rotation_frames))
        rot_frames = render_camera_trajectory(
            pb, pitch_rot, roll_rot, yaw_rot, rotation_frames, cam_distance, cam_target_pose,
            light_direction=light_directions, light_distance=light_distance, shadow=shadow,
            render_width=render_width, render_height=render_height, fov=fov
        )

        # Add final loop
        roll_inv = np.ones_like(t) * 180
        yaw_inv = yaw
        pitch_inv = pitch
        frames_inv = render_camera_trajectory(
            pb, pitch_inv, roll_inv, yaw_inv, n_frames, cam_distance, cam_target_pose,
            light_direction=light_directions, light_distance=light_distance, shadow=shadow,
            render_width=render_width, render_height=render_height, fov=fov
        )[:-1]
        # Add transition frames
        # frames = np.concatenate([rot_frames, list(reversed(rot_frames))], axis=0)
        if gen_gif:
            frames = np.concatenate([frames, rot_frames, frames_inv, list(reversed(rot_frames))], axis=0)
        elif gen_imgs:
            frames = np.concatenate([frames, frames_inv], axis=0)

    if gen_gif:
        from moviepy.editor import ImageSequenceClip
        # Save animation
        file_name = file_name.replace(".gif", '')
        file_path = save_path / f'{file_name}.gif'
        file_count = 1
        while file_path.exists():
            file_path = save_path / f'{file_name}({file_count}).gif'
            file_count += 1
        clip = ImageSequenceClip(list(frames), fps=fps)
        clip.write_gif(file_path, fps=fps, loop=False, )  # program='ffmpeg', progress_bar=True, fuzz=0.05)
        print(f"Animation saved to {file_path.absolute()}")
    elif gen_imgs:
        import matplotlib.pyplot as plt
        for i, frame in enumerate(frames):
            file_path = save_path / f'{file_name}-{i}'
            # Create a figure and axis
            fig, ax = plt.subplots()
            ax.imshow(frame)
            ax.axis('off')
            plt.savefig(file_path, dpi=300, bbox_inches='tight', pad_inches=0)
            print(f"Frame saved to {file_path.absolute()}")
            plt.close(fig)
        return frames


def render_camera_trajectory(pb, pitch, roll, yaw, n_frames, cam_distance, cam_target_pose, upAxisIndex=2,
                             render_width=812, render_height=812, nearPlane=0.01, farPlane=100, fov=60,
                             light_direction=(0, 0, 0.5), light_distance=1.0, shadow=True,
                             ):
    """Renders a camera trajectory given a set of yaw, pitch, roll angle trajectories."""
    # Set rendering constants
    aspect = render_width / render_height
    # Capture frames
    frames = []  # frames to create animated png
    for s in tqdm(range(n_frames), desc="Capturing frames"):
        yaw_t, pitch_t, roll_t = yaw[s], pitch[s], roll[s]
        # Compute view and projection matrices from yaw, pitch, roll
        viewMatrix = pb.computeViewMatrixFromYawPitchRoll(
            cam_target_pose, cam_distance, yaw_t, pitch_t, roll_t, upAxisIndex)
        projectionMatrix = pb.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)
        # Render image
        img_arr = pb.getCameraImage(render_width, render_height, viewMatrix, projectionMatrix, shadow=shadow,
                                    lightDirection=light_direction, lightDistance=light_distance,
                                    renderer=pb.ER_TINY_RENDERER,
                                    # renderer=pb.ER_BULLET_HARDWARE_OPENGL,
                                    lightColor=[1, 1, 1],
                                    lightSpecularCoeff=0.32)
        w = img_arr[0]  # width of the image, in pixels
        h = img_arr[1]  # height of the image, in pixels
        rgb = img_arr[2]  # color data RGB
        np_img_arr = np.reshape(rgb, (h, w, 4))
        frame = np_img_arr[:, :, :]
        frames.append(frame)
    return frames


# Setup debug sliders
def setup_debug_sliders(pb, robot):
    """Setup debug sliders for each joint of the robot in pybullet."""
    for i, joint_name in enumerate(robot.joint_space_names):
        joint = robot.joint_space[joint_name]
        lower_limit = max(joint.sim_joint.pos_limit_low, -np.pi)
        upper_limit = min(joint.sim_joint.pos_limit_high, np.pi)
        if lower_limit > upper_limit:
            lower_limit, upper_limit = -np.pi, np.pi
        pb.addUserDebugParameter(paramName=f"{i}:{joint_name}", rangeMin=lower_limit,
                                 rangeMax=upper_limit, startValue=0.0)


# Read param values
def listen_update_robot_sliders(pb, robot):
    """Read the values of the debug sliders and update the robot accordingly in pybullet."""
    import time
    while True:
        for i, joint_name in enumerate(robot.joint_space_names):
            joint = robot.joint_space[joint_name]
            theta = pb.readUserDebugParameter(itemUniqueId=i)
            pb.resetJointState(robot.robot_id, joint.bullet_idx, theta)
        time.sleep(0.1)


def change_robot_appearance(pb, robot: PinBulletWrapper, change_color=True, alpha: float = 1.0):
    """Tint the robot in pybullet to get similar visualization of symmetric robots."""
    if not change_color and alpha == 1.0:
        return

    # Define repo awsome colors. Lets call it Danfoa's color palette :)
    robot_color = [0.054, 0.415, 0.505 , alpha]  # This is a nice teal
    FL_leg_color = [0.698, 0.376, 0.082, alpha]  # This is a nice orange
    FR_leg_color = [0.260, 0.263, 0.263, alpha]  # This is a nice grey
    HL_leg_color = [0.800, 0.480, 0.000, alpha]  # This is a nice yellow
    HR_leg_color = [0.710, 0.703, 0.703, alpha]  # This is a nice light grey

    # Get robot bodies visual data.
    # visual_data = pb.getVisualShapeData(robot.robot_id)
    # Pybullet makes it hard to match joints and links visual data. This is an approach to get it done.
    # get_link_visual_data = lambda link_idx: [data for data in visual_data if data[1] == link_idx][0]

    for joint_idx in range(pb.getNumJoints(robot.robot_id)):
        joint_info = pb.getJointInfo(robot.robot_id, joint_idx)
        joint_info[12].decode("UTF-8")
        joint_name = joint_info[1].decode("UTF-8")
        # link_data = get_link_visual_data(link_idx)
        # link_body_id, link_color = link_data[0], link_data[1], link_data[7]
        # thigh_fr_to_knee_fr_j
        if change_color:
            # if link_name in robot.endeff_names or (joint_name in robot.endeff_names):
            #     color = endeff_color
            if np.any([s in joint_name.lower() for s in ["fl_", "lf_", "left", "_0"]]):
                color = FL_leg_color
            elif np.any([s in joint_name.lower() for s in ["fr_", "rf_", "right", "_120"]]):
                color = FR_leg_color
            elif np.any([s in joint_name.lower() for s in ["rl_", "hl_", "lh_", "left",]]):
                color = HL_leg_color
            elif np.any([s in joint_name.lower() for s in ["rr_", "hr_", "rh_", "right"]]):
                color = HR_leg_color
            else:
                color = robot_color

            pb.changeVisualShape(objectUniqueId=robot.robot_id, linkIndex=joint_idx,
                                 rgbaColor=color, specularColor=[0, 0, 0])

    if change_color:
        pb.changeVisualShape(objectUniqueId=robot.robot_id, linkIndex=-1, rgbaColor=robot_color,
                             specularColor=[0, 0, 0])


def spawn_robot_instances(
        robot: PinBulletWrapper, bullet_client: Optional[BulletClient],
        base_positions: List[List], base_orientations: Optional[List[List]] = None,
        tint: bool = False, alpha: float = 1.0,
        ) -> List[PinBulletWrapper]:
    """Spawn multiple instances of the same robot in pybullet in de defined locations and orientations.

    Args:
        robot (PinBulletWrapper): Original robot instance
        bullet_client (Optional[BulletClient]): Pybullet client to spawn the robots.
        base_positions (Union[List[List], List]): List of base positions for each robot base.
        base_orientations (Union[List[List], List]): List of base orientation quaternions for each robot base.
        tint (bool, optional): Whether to change the color of the robot bodies.
        alpha (float, optional): Alpha value for robot body colors.

    Returns:
        spawned_robots (List[PinBulletWrapper]): List of the spawned robots.

    """
    if bullet_client is None:
        bullet_client = BulletClient(connection_mode=pybullet.DIRECT)

    n_instances = len(base_positions)
    if base_orientations is None:
        base_orientations = [[0, 0, 0, 1] for _ in range(n_instances)]

    assert n_instances == len(base_orientations), "Need to provide a base position and orientation per robot instance"

    # TODO: Copy error from Pinocchio. To be checked
    # robots = [PinBulletWrapper.from_instance(robot) for _ in range(n_instances)]
    kwargs = dict(robot_name=robot.robot_name, init_q=robot._init_q, hip_height=robot.hip_height,
                  endeff_names=robot.endeff_names, q_zero=robot._q_zero)
    robots = [PinBulletWrapper(**kwargs) for _ in range(n_instances)]
    world = robot.world
    for r, pos, ori in zip(robots, base_positions, base_orientations):
        r.configure_bullet_simulation(bullet_client=bullet_client, world=world, base_pos=pos, base_ori=ori)
        change_robot_appearance(bullet_client, r, change_color=tint, alpha=alpha)
        world = r.world

    return robots


def display_robots_and_vectors(pb, robot, base_confs, Gq_js, Gdq_js, Ghg, forces, forces_points, surface_normals,
                               GX_g_bar, tint=True, draw_floor=True):
    """Plot side by side robots with different configurations, CoM momentums and expected CoM after an action g."""
    # pb.resetSimulation()

    # Optional: Display origin.
    draw_vector(pb, np.zeros(3), np.asarray([.1, 0, 0]), v_color=[1, 0, 0, 1])
    draw_vector(pb, np.zeros(3), np.asarray([0, .1, 0]), v_color=[0, 1, 0, 1])
    draw_vector(pb, np.zeros(3), np.asarray([0, 0, .1]), v_color=[0, 0, 1, 1])

    plane_height = 0
    plane_size = (0.01, robot.hip_height / 2, robot.hip_height / 2)

    # Sagittal plane
    plot_reflection_plane(pb, R=rt.matrix_from_two_vectors(a=[0, 1, 0], b=[1, 0, 0]),
                          p=[0.0, 0.0, plane_height],
                          color=np.array([230, 230, 256, 40]) / 256., size=plane_size)
    plot_reflection_plane(pb, R=rt.matrix_from_two_vectors(a=[1, 0, 0], b=[0, 1, 0]),
                          p=[0.0, 0.0, plane_height],
                          color=np.array([256, 230, 230, 40]) / 256., size=plane_size)
    plot_reflection_plane(pb, R=rt.matrix_from_two_vectors(a=[0, 0, 1], b=[0, 1, 0]),
                          p=[0.0, 0.0, 0.0],
                          color=np.array([250, 250, 250, 80]) / 256.,
                          size=(0.01, robot.hip_height * 6, robot.hip_height * 6))

    robots = [robot]
    com_pos = None
    for i in range(0, len(Gq_js)):
        q_js, dq_js, XB_w, ghg_B, rho_X_gbar = Gq_js[i], Gdq_js[i], base_confs[i], Ghg[i], GX_g_bar[i]
        assert q_js.size == robot.nq - 7, f"Invalid joint-space position dim(Q_js)={robot.nq - 7}!={q_js.size}"
        assert dq_js.size == robot.nv - 6, f"Invalid joint-space velocity dim(TqQ_js)={robot.nv - 6}!={dq_js.size}"
        RB_w = XB_w[:3, :3]
        tB_w = XB_w[:3, 3]
        grobot = robot
        if i > 0:
            grobot = robot if i == 0 else copy.copy(robot)
            grobot.configure_bullet_simulation(pb, world=None)
            if tint:
                change_robot_appearance(pb, grobot)
            robots.append(grobot)
        # Place robots in env
        base_q = SE3_2_gen_coordinates(XB_w)
        # Set positions:
        grobot.reset_state(np.concatenate((base_q, q_js)), np.concatenate((np.zeros(6), dq_js)))
        # Add small offset to COM for visualization.
        if com_pos is None:
            com_pos = robot.pinocchio_robot.com(q=np.concatenate((base_q, q_js))) + \
                      (RB_w @ np.array([robot.hip_height, robot.hip_height, 0.05]))

        gcom_pos = tr.transform(rho_X_gbar, tr.vector_to_point(com_pos), strict_check=False)[:3]
        # Draw COM momentum and COM location
        com_id = pb.createVisualShape(shapeType=pb.GEOM_SPHERE, radius=0.02,
                                      rgbaColor=np.array([10, 10, 10, 255]) / 255.)
        pb.createMultiBody(baseMass=1, baseVisualShapeIndex=com_id, basePosition=gcom_pos,
                           baseOrientation=matrix_to_quat_xyzw(np.eye(3)))
        draw_vector(pb, origin=gcom_pos, vector=ghg_B[:3],
                    v_color=np.array([255, 153, 0, 255]) / 255.,
                    scale=(1 / np.linalg.norm(ghg_B[:3]) * robot.hip_height * .3))
        draw_vector(pb, origin=gcom_pos, vector=ghg_B[3:],
                    v_color=np.array([136, 204, 0, 255]) / 255.,
                    scale=(1 / np.linalg.norm(ghg_B[3:]) * robot.hip_height * .3))

        # Draw forces and contact planes
        force_color = (0.590, 0.153, 0.510, 1.0)
        for force_orbit, rf_orbit, GRf_w in zip(forces, forces_points, surface_normals):
            draw_vector(pb, origin=rf_orbit[i], vector=force_orbit[i], v_color=force_color)
            if draw_floor:
                body_id = pb.createVisualShape(shapeType=pb.GEOM_BOX, halfExtents=[.2 * robot.hip_height,
                                                                                   .2 * robot.hip_height,
                                                                                   0.01],
                                               rgbaColor=np.array([115, 140, 148, 150]) / 255.)
                pb.createMultiBody(baseMass=1,
                                   baseInertialFramePosition=[0, 0, 0],
                                   baseCollisionShapeIndex=body_id,
                                   baseVisualShapeIndex=body_id,
                                   basePosition=rf_orbit[i],
                                   baseOrientation=matrix_to_quat_xyzw(GRf_w[i]))
        # Draw Base orientation
        if robot.nq == 12:  # Only for Solo
            draw_vector(pb, origin=tB_w + RB_w @ np.array((0.06, 0, 0.03)), vector=RB_w[:, 0], v_color=[1, 1, 1, 1],
                        scale=0.05)


def get_mock_ground_reaction_forces(pb, robot, robot_cfg):
    """Get mock ground reaction forces for visualization purposes. Simply to show transformation of vectors."""
    end_effectors = np.random.choice(robot.bullet_ids_allowed_floor_contacts,
                                     len(robot.bullet_ids_allowed_floor_contacts), replace=False)
    # Get positions and orientations of end effector links of the robot, used to place the forces used in visualization
    rf1_w, quatf1_w = (np.array(x) for x in pb.getLinkState(robot.robot_id, end_effectors[0])[0:2])
    rf2_w, quatf2_w = (np.array(x) for x in pb.getLinkState(robot.robot_id, end_effectors[1])[0:2])
    Rf1_w, Rf2_w = quat_xyzw_to_SO3(quatf1_w), quat_xyzw_to_SO3(quatf2_w)

    if not np.any([s in robot.robot_name for s in ["atlas"]]):  # Ignore
        Rf1_w, Rf2_w = np.eye(3), np.eye(3)
    rf1_w -= Rf1_w @ np.array([0, 0, 0.03])
    rf2_w -= Rf2_w @ np.array([0, 0, 0.03])
    # Add some random force magnitures to the vectors. # Rf_w[:, 2] := Surface normal
    f1_w = Rf1_w[:, 2] + [2 * np.random.rand() - 1, 2 * np.random.rand() - 1, np.random.rand()]
    f2_w = Rf2_w[:, 2] + [2 * np.random.rand() - 1, 2 * np.random.rand() - 1, np.random.rand()]
    # For visualization purposes we make the forces proportional to the robot height
    f1_w = f1_w / np.linalg.norm(f1_w) * robot_cfg.hip_height * .4
    f2_w = f2_w / np.linalg.norm(f2_w) * robot_cfg.hip_height * .4
    return Rf1_w, Rf2_w, f1_w, f2_w, rf1_w, rf2_w


def configure_bullet_simulation(gui=True, debug=False):
    """Configure bullet simulation."""
    import pybullet_data
    from pybullet import (
        COV_ENABLE_DEPTH_BUFFER_PREVIEW,
        COV_ENABLE_GUI,
        COV_ENABLE_MOUSE_PICKING,
        COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
        DIRECT,
        GUI,
    )
    from pybullet_utils import bullet_client

    BACKGROUND_COLOR = '--background_color_red=%.2f --background_color_green=%.2f --background_color_blue=%.2f' % \
                       (1.0, 1.0, 1.0)

    if gui:
        pb = bullet_client.BulletClient(connection_mode=GUI, options=BACKGROUND_COLOR)
    else:
        pb = bullet_client.BulletClient(connection_mode=DIRECT)
    pb.configureDebugVisualizer(COV_ENABLE_GUI, debug)
    pb.configureDebugVisualizer(COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
    pb.configureDebugVisualizer(COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    pb.configureDebugVisualizer(COV_ENABLE_MOUSE_PICKING, 0)

    pb.resetSimulation()
    pb.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    # Load floor
    # floor_id = pb.loadURDF("plane.urdf", basePosition=[0, 0, 0.0], useFixedBase=1)
    return pb
