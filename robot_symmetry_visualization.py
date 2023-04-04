import os
import pathlib
import time

import hydra
import numpy as np
from omegaconf import DictConfig
from pytransform3d import transformations as tr

from utils.pybullet_visual_utils import generate_rotating_view_gif, tint_robot, \
    display_robots_and_vectors
from utils.robot_utils import load_robot_and_symmetries
from utils.utils import configure_bullet_simulation, SE3_2_gen_coordinates, quat_xyzw_to_SO3

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@hydra.main(config_path='cfg/supervised', config_name='config_visualization')
def main(cfg: DictConfig):
    # Get robot instance, along with representations of the symmetry group on the Euclidean space (in which the robot
    # base B evolves in) and Joint Space (in which the internal configuration of the robot evolves in).
    robot, rep_E3, rep_QJ = load_robot_and_symmetries(robot_cfg=cfg.robot, debug=cfg.debug)

    # Configuration of the robot. Check cfg/robot/
    robot_cfg = cfg.robot
    # Configuration of the 3D visualization -------------------------------------------------------------------------
    offset = 3.3 * robot.hip_height
    if len(rep_E3.G.discrete_actions) == 2:
        cam_target_pose = [0, offset / 2, robot.hip_height/2]
        rep_E3_offsets = [[0, offset, 0]]
    elif len(rep_E3.G.discrete_actions) == 4:
        cam_target_pose = [offset / 2, offset / 2, robot.hip_height/2]
        rep_E3_offsets = [[0, offset, 0], [offset, 0, 0], [offset, offset, 0]]
    else:
        raise NotImplementedError("Ups have to setup visualization params for other groups...soon")

    pb = configure_bullet_simulation(gui=cfg.gui, debug=cfg.debug)
    robot.configure_bullet_simulation(pb, world=None)
    tint_robot(pb, robot)

    # Get initial random configuration of the system
    q, dq = robot.get_init_config(random=True, angle_sweep=robot_cfg.angle_sweep)
    robot.reset_state(q, dq)
    # --------------------------------------------------------------------------------------------------------------

    # NOTATION:
    # We use the following code notation which tries to replicate as much as possible the notation of the paper:
    # [Variable][Frame]_[reference frame of variable] e.g, RB_w : Rotation (R) of Base (B) in World (w) coordinates.
    # - `Ed`: Euclidean space of d dimensions
    # - `QJ`: Joint Space of the robot
    # - `TQJ`: Tangent space to the joint space.
    # - `r`: Vector in `Ed` describing the position of a frame of the robot bodies. e.g. rB: Robot base position
    # - `R`: SO3 matrix describing a rotation in E3. e.g. RB: Robot base orientation
    # - `X`: Homogenous Matrix 4x4 transformation holding a rotation matrix and a translation X = [[R , r], [0, 1]]



    # Specify the configuration of the original robot base
    # Keep a "null" notation for computation of hb in Base reference frame.
    rB_w = np.array(q[:3])                                # Base position in R^3
    RB_w = quat_xyzw_to_SO3(q[3:7])                       # Base Rotation in SO3
    XB_w = tr.transform_from(R=RB_w, p=rB_w)              # Base Homogenous Transformation
    XBnull_w = tr.transform_from(R=np.eye(3), p=rB_w)     # Aux variable of original base configuration.

    # x: is the concatenation of q_js and dq_js. That is the joint-space state-space vector.
    x = np.concatenate((q[7:], dq[6:]))
    x = x.astype(np.float64)
    # Get the representation of the symmetry group of the robot on the joint space (JS) state-space x=[q_js, dq_js]
    rep_x = rep_QJ + rep_QJ  # Since rep_QJ acts linearly on q, the representation on dq is the same (Eq.2 of paper)

    # As example data we will see how the symmetries of the robotic system affect propioceptive and exteroceptive
    # data measurements. We will use as example:
    # - Propioceptive: `hg_B` The center of mass linear and angular momentum of the robot, and `f1/2` contact forces
    # affecting the robot
    # - Exteroceptive: We will assume to have measurements of the terrain elevation and orientation.
    # ================================================================================================================
    # Use Pinocchio to compute the robot Center of Mass (CoM) momentum in base `B` coordinates
    hg_B = robot.pinocchio_robot.centroidalMomentum(q=np.concatenate((SE3_2_gen_coordinates(XBnull_w), q[7:])),
                                                    v=np.concatenate((np.zeros(6), dq[6:]))).np.astype(np.float64)
    # Note: For visualization purposes we make the vectors fixed and clear from the top view
    hg_B = np.array([0.1, 0.1, 0.0, -0.0, -0.2, -0.2])
    # Compute the representations of the symmetries of the robot acting on the linear `l` and angular `k` components
    # of Center of Mass (CoM) momentum `h=[l,k]`
    rep_l = rep_E3
    rep_k = rep_E3.set_pseudovector(True)
    rep_h = rep_l + rep_k   # Additions of representations amounts to block-diagonal matrix concatenation.

    # Define a ground-reaction/contact force on two of the system links/end effectos:
    # Points of application of forces in Base coordinates on the feets of the robot.
    R_B2w = RB_w        # Rotation matrix transforming from Base reference frame to worl reference frame
    R_w2B = np.linalg.inv(R_B2w)
    end_effectors = np.random.choice(robot.bullet_ids_allowed_floor_contacts, 2, replace=False)
    # Get positions and orientations of end effector links of the robot, used to place the forces used in visualization
    rf1_w, quatf1_w = (np.array(x) for x in pb.getLinkState(robot.robot_id, end_effectors[0])[0:2])
    rf2_w, quatf2_w = (np.array(x) for x in pb.getLinkState(robot.robot_id, end_effectors[1])[0:2])
    Rf1_w, Rf2_w = quat_xyzw_to_SO3(quatf1_w), quat_xyzw_to_SO3(quatf2_w)
    if not np.any([s in robot.robot_name for s in ["atlas"]]):   # Ignore
        Rf1_w = np.eye(3)
        Rf2_w = np.eye(3)
    # Add some random force magnitures to the vectors.
    f1_w = Rf1_w[:, 2] + [2 * np.random.rand() - 1, 2 * np.random.rand() - 1,
                          np.random.rand()]  # Rf_w[:, 2] := Surface normal
    f2_w = Rf2_w[:, 2] + [2 * np.random.rand() - 1, 2 * np.random.rand() - 1, np.random.rand()]
    # For visualization purposes we make the forces proportional to the robot height
    f1_w = f1_w / np.linalg.norm(f1_w) * robot_cfg.hip_height * .4
    f2_w = f2_w / np.linalg.norm(f2_w) * robot_cfg.hip_height * .4
    # ================================================================================================================

    # Main part of the script. =======================================================================================
    # Here we use symmetry actions to visually depict symmetries in robot configuration, and propioceptive and
    # exteroceptive measurements.
    # Start by defining lists holding the orbits Gof all the proprioceptive and exteroceptive measurements
    # An orbit is all unique symmetric states of a variable G·x = {g·x | g in G}
    Gx, Ghg_B, Gy_B, GXB_w, GX_g_bar = [x], [hg_B], [hg_B], [XB_w], [tr.transform_from(R=np.eye(3), p=np.zeros(3))]
    Gf1_w, Gf2_w, Gr1_w, Gr2_w, GRf1_w, GRf2_w = [f1_w], [f2_w], [rf1_w], [rf2_w], [Rf1_w], [Rf2_w]

    # For each symmetry `g` of the system, get the representations of the action in the relevant vector spaces to
    # compute the symmetric states of robot configuration and data
    for rho_x_g, rho_y_g, rho_qj_g, rho_E3_g_bar, rho_E3_r in zip(rep_x.G.discrete_actions[1:],
                                                                  rep_h.G.discrete_actions[1:],
                                                                  rep_QJ.G.discrete_actions[1:],
                                                                  rep_E3.G.discrete_actions[1:],
                                                                  rep_E3_offsets):
        # Let x = [q, dq], and y = [l, k] = h
        # Get the euclidean rotation/reflection of the symmetry in Ed
        R_g_bar = np.asarray(rho_E3_g_bar.todense())  # True rotation/reflection of space
        # For visualization, we assume reflections are performed w.r.t a given plane (not relevant)
        r_g_bar = np.asarray(rho_E3_r)  # Add position to the planes of reflection
        # Homogenous transformation matrix representing g
        X_g_bar = tr.transform_from(R=R_g_bar, p=r_g_bar)
        GX_g_bar.append(X_g_bar)   # Add to orbit of Homogenous transformations

        # Get symmetric g.x=[g.q, g.dq], g·y=[g·l, g·k] = [g·h]
        gx_w, gy_B = rho_x_g @ x, rho_y_g @ hg_B
        Gx.append(gx_w), Ghg_B.append(gy_B)

        # Compute new robot base configuration
        gRB_w = R_g_bar @ RB_w @ R_g_bar
        gtXB_w = tr.transform(X_g_bar, XB_w[:, 3], strict_check=False)[:3]  # Transform the base position
        # Add new robot base configuration (homogenous matrix) to the orbit of base configs.
        GXB_w.append(tr.transform_from(R=gRB_w, p=gtXB_w))

        # Use symmetry representations to get symmetric versions of euclidean vectors, representing measurements of data
        # We could also add some pseudo-vectors e.g. torque, and augment them as we did with `k`
        gf1_w, gf2_w = R_g_bar @ f1_w, R_g_bar @ f2_w
        gr_f1_w, gr_f2_w = tr.transform(X_g_bar, tr.vector_to_point(rf1_w), strict_check=False)[:3], \
                           tr.transform(X_g_bar, tr.vector_to_point(rf2_w), strict_check=False)[:3]
        Gf1_w.append(gf1_w[:3]), Gf2_w.append(gf2_w[:3]), Gr1_w.append(gr_f1_w), Gr2_w.append(gr_f2_w)

        # The environment is theoretically truly reflected/rotated, but we use "cuboids" for representing estimations of
        # terrain elevation and surface normal, so we can pre and post multiply to get a valid config of the "cuboids"
        GRf1_w.append(R_g_bar @ Rf1_w @ R_g_bar), GRf2_w.append(R_g_bar @ Rf2_w @ R_g_bar)

        # Get the true center of mass momentum of the robot using pinocchio.
        gy_B_true = robot.pinocchio_robot.centroidalMomentum(
            q=np.concatenate((SE3_2_gen_coordinates(XBnull_w), np.split(gx_w, 2)[0])),
            v=np.concatenate((np.zeros(6), np.split(gx_w, 2)[1]))).np.astype(np.float64)
        Gy_B.append(gy_B_true)
    # =============================================================================================================

    # Visualization of orbits of robot states and of data ==========================================================
    # Use Ctrl and mouse-click+drag to rotate the 3D environment.
    # Get the robot joint state (q_js, dq_js) from the state x for all system configurations.
    splited_orbits = [np.split(x, 2) for x in Gx]
    Gq_js, Gdq_js = [x[0] for x in splited_orbits], [x[1] for x in splited_orbits]
    display_robots_and_vectors(pb, robot, base_confs=GXB_w, Gq_js=Gq_js, Gdq_js=Gdq_js, Ghg=Gy_B, Ghg_pin=Ghg_B,
                               forces=[Gf1_w, Gf2_w], forces_points=[Gr1_w, Gr2_w], surface_normals=[GRf1_w, GRf2_w],
                               GX_g_bar=GX_g_bar, offset=offset)
    
    save_path = pathlib.Path("paper/animations")
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
