import pathlib

import numpy as np
from pytransform3d import rotations as rt

from utils.utils import matrix_to_quat_xyzw


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
    clip.write_gif(file_path, fps=fps, loop=0)
    print(f"Animation saved to {file_path.absolute()}")
