import os
import pathlib
import time

import hydra
import numpy as np
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation
from utils.pybullet_visual_utils import (
    change_robot_appearance,
    display_robots_and_vectors,
    get_mock_ground_reaction_forces,
    render_orbiting_animation,
)
from utils.robot_utils import load_symmetric_system

import morpho_symm
from morpho_symm.utils.pybullet_visual_utils import configure_bullet_simulation

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@hydra.main(config_path="cfg/", config_name="config_visualization", version_base="1.3")
def main(cfg: DictConfig):
    """Visualize the effect of DMSs transformations in 3D animation.

    This script visualizes the DMSs transformations on robot state and on proprioceptive and exteroceptive measurements.
    """
    cfg.robot.seed = cfg.robot.seed if cfg.robot.seed >= 0 else np.random.randint(0, 1000)
    np.random.seed(cfg.robot.seed)
    # Get robot instance, along with representations of the symmetry group on the Euclidean space (in which the robot
    # base B evolves in) and Joint Space (in which the internal configuration of the robot evolves in).
    robot, G = load_symmetric_system(robot_cfg=cfg.robot, debug=cfg.debug_joints)

    # Get the group representations of joint-state space, and Euclidean space.
    rep_QJ = G.representations["Q_js"]  # rep_QJ(g) is a permutation matrix ∈ R^nqj
    rep_TqQJ = G.representations["TqQ_js"]  # rep_TqQJ(g) is a permutation matrix ∈ R^nvj
    rep_Ed = G.representations["E3"]  # rep_Ed(g) is a homogenous transformation matrix ∈ R^(3+1)x(3+1)
    rep_Rd = G.representations["R3"]  # rep_Rd(g) is an orthogonal matrix ∈ R^3x3
    rep_Rd_pseudo = G.representations["R3_pseudo"]  # rep_Rd_pseudo(g) is an orthogonal matrix ∈ R^3x3
    rep_euler_xyz = G.representations["euler_xyz"]  # rep_euler_xyz(g) is an euler angle vector ∈ R^3

    # Configuration of the 3D visualization -------------------------------------------------------------------------
    # Not really relevant to understand.
    offset = max(0.2, 1.8 * robot.hip_height)  # Offset between robot base and reflection planes.
    pb = configure_bullet_simulation(gui=cfg.gui, debug=cfg.debug_joints)  # Load pybullet environment
    robot.configure_bullet_simulation(pb, world=None)  # Load robot in pybullet environment
    change_robot_appearance(pb, robot, change_color=cfg.robot.tint_bodies)  # Add color and style to boring grey robots

    # Get initial random configuration of the system
    q, v = robot.get_init_config(random=True, angle_sweep=cfg.robot.angle_sweep, fix_base=cfg.robot.fix_base)
    # q[7:] *= 0
    rB0 = np.array([-offset if G.order() != 2 else 0, -offset] + [robot.hip_height * 1.5])
    q[:3] = rB0  # Place base of the robot with some `offset` from origin.
    robot.reset_state(q, v)  # Reset robot state in pybullet and pinocchio.
    # --------------------------------------------------------------------------------------------------------------

    # NOTATION:
    # We use the following code notation which tries to replicate as much as possible the notation of the paper:
    # [Variable][Frame]_[reference frame of variable] e.g, RB_w : Rotation (R) of Base (B) in World (w) coordinates.
    # For compactness the reference frame is omitted if variables are measured w.r.t (w) world frame.
    # - `Ed`: Euclidean space of d dimensions
    # - `QJ`: Joint Space of the robot
    # - `TQJ`: Tangent space to the joint space.
    # - `r`: Vector in `Ed` describing the position of a frame of the robot bodies. e.g. rB: Robot base position
    # - `R`: SO3 matrix describing a rotation in E3. e.g. RB: Robot base orientation
    # - `X`: Homogenous Matrix 4x4 transformation holding a rotation matrix and a translation X = [[R , r], [0, 1]]

    # Get the robot's base configuration XB ∈ Ed as a homogenous transformation matrix.
    XB = robot.get_base_configuration()
    base_ori_euler_xyz = Rotation.from_matrix(XB[:3, :3]).as_euler("xyz", degrees=True)

    # Get joint space position and velocity coordinates  (q_js, v_js) | q_js ∈ QJ, dq_js ∈ TqQJ
    q_js, v_js = robot.get_joint_space_state()

    # To visualize how the symmetries of the robotic system affect proprioceptive and exteroceptive measurements we use:
    # - Proprioceptive: `hg_B` The center of mass linear and angular momentum of the robot, and `f1/2` contact forces
    # - Exteroceptive: We will assume to have measurements of the terrain elevation and orientation.
    # Consider the robot Center of Mass (CoM) momentum in base `B` coordinates. A vector and a pseudo-vector.
    hg_B = np.array([0.1, 0.1, 0.0, -0.0, -0.2, -0.2])  # Set to fix value for visualization.
    # The representation for the linear `l` and angular `k` components  of Center of Mass (CoM) momentum `h=[l,k]` is:
    rep_h = rep_Rd + rep_Rd_pseudo  # Additions of representations amounts to block-diagonal matrix concatenation.

    # Define mock surface orientations `Rf_w`, force contact points `rf_w` and contact forces `f_w`
    Rf1, Rf2, f1, f2, rf1, rf2 = get_mock_ground_reaction_forces(pb, robot, cfg.robot)

    # Main part of the script. =======================================================================================
    # Start by defining the dict representing the orbits of all the proprioceptive and exteroceptive measurements
    # An orbit is all unique symmetric states of a variable G·z = {g·z | g in G}
    e = G.identity  # Identity element of the group
    orbit_q_js, orbit_v_js = {e: q_js}, {e: v_js}
    orbit_hg_B, orbit_XB_w = {e: hg_B}, {e: XB}
    orbit_f1, orbit_f2, orbit_rf1, orbit_rf2 = (
        {e: f1},
        {e: f2},
        {e: rf1},
        {e: rf2},
    )
    orbit_Rf1, orbit_Rf2 = {e: Rf1}, {e: Rf2}
    orbit_ori_euler_xyz = {e: base_ori_euler_xyz}
    # For each symmetry action g ∈ G, we get the representations of the action in the relevant vector spaces to
    # compute the symmetric states of robot configuration and measurements.
    for g in G.elements[1:]:
        # Get symmetric joint-space state (g.q_js, g.v_js), and CoM momentum g·h=[g·l, g·k] ---------------------------
        # gx_w, gh_B = (rep_x(g) @ x).astype(x.dtype), (rep_h(g) @ hg_B).astype(hg_B.dtype)
        orbit_q_js[g], orbit_v_js[g] = rep_QJ(g) @ q_js, rep_TqQJ(g) @ v_js
        orbit_hg_B[g] = rep_h(g) @ hg_B

        # Compute new robot base configuration -----------------------------------------------------------------------
        # gXB_w = rep_Ed(g) @ RB_w @ np.linalg.inv(rep_Ed(g))
        gXB = rep_Ed(g) @ XB @ rep_Ed(g).T
        orbit_XB_w[g] = gXB  # Add new robot base configuration (homogenous matrix) to the orbit of base configs.
        orbit_ori_euler_xyz[g] = Rotation.from_matrix(gXB[:3, :3]).as_euler("xyz", degrees=True)
        # If people use euler xyz angles to represent the orientation of the robot base, we can also compute the
        # symmetric states of the robot base orientation:
        g_euler_xyz = rep_euler_xyz(g) @ base_ori_euler_xyz
        # Check the analytic transformation to elements of SO(3) is equivalent to the transformation in euler xyz angles
        g_euler_xyz_true = Rotation.from_matrix(gXB[:3, :3]).as_euler("xyz", degrees=True)
        assert np.allclose(g_euler_xyz, g_euler_xyz_true, rtol=1e-6, atol=1e-6)

        # Use symmetry representations to get symmetric versions of Euclidean vectors, representing measurements of data
        orbit_f1[g], orbit_f2[g] = rep_Rd(g) @ f1, rep_Rd(g) @ f2
        orbit_rf1[g] = (rep_Ed(g) @ np.concatenate((rf1, np.ones(1))))[:3]  # using homogenous coordinates
        orbit_rf2[g] = (rep_Ed(g) @ np.concatenate((rf2, np.ones(1))))[:3]

        # Apply transformations to the terrain elevation estimations/measurements
        orbit_Rf1[g] = rep_Rd(g) @ Rf1 @ rep_Rd(g).T
        orbit_Rf2[g] = rep_Rd(g) @ Rf2 @ rep_Rd(g).T

    for g in G.elements:
        print(f"Element: {g} euler_xyz(g): \n{rep_euler_xyz(g)}")
        print(f"Element: {g} Rd_pseudo(g): \n{rep_Rd_pseudo(g)}")

    # Visualization of orbits of robot states and of data ==========================================================
    # Use Ctrl and mouse-click+drag to rotate the 3D environment.
    display_robots_and_vectors(
        pb,
        robot,
        group=G,
        base_confs=orbit_XB_w,
        orbit_q_js=orbit_q_js,
        orbit_v_js=orbit_v_js,
        orbit_com_momentum=orbit_hg_B,
        forces=[orbit_f1, orbit_f2],
        forces_points=[orbit_rf1, orbit_rf2],
        surface_normals=[orbit_Rf1, orbit_Rf2],
        tint=cfg.robot.tint_bodies,
        draw_floor=cfg.robot.draw_floor,
    )

    root_path = pathlib.Path(morpho_symm.__file__).parents[1].absolute()
    if cfg.make_gif:
        # To get the optimal visualization you might need to play a bit with the rendering parameters for each robot.
        # this params seem to work mostly ok for all.
        cam_distance = offset * 5
        cam_target_pose = [0, 0, 0]
        save_path = root_path / "docs/static/animations"
        save_path.mkdir(exist_ok=True)
        render_orbiting_animation(
            pb,
            cam_target_pose=cam_target_pose,
            cam_distance=cam_distance,
            save_path=save_path,
            anim_time=10,
            fps=15,
            periods=1,
            init_roll_pitch_yaw=(0, 35, 0),
            invert_roll="dh" in cfg.robot.group_label.lower(),
            pitch_sin_amplitude=20,
            file_name=f"{robot.name}-{G.name}-symmetries_anim_static",
            gen_gif=True,
            gen_imgs=False,
        )
        print("Done enjoy your gif :). I hope you learned something new")
    elif cfg.make_imgs:
        cam_distance = offset * 6
        cam_target_pose = [0, 0, 0]
        save_path = root_path / f"paper/images/{cfg.robot.name}"
        save_path.mkdir(exist_ok=True)
        render_orbiting_animation(
            pb,
            cam_target_pose=cam_target_pose,
            cam_distance=cam_distance,
            anim_time=2,
            fps=2,
            periods=1,
            pitch_sin_amplitude=0,
            init_roll_pitch_yaw=(0, 90, 0) if G.order() > 2 else (0, 0, 0),
            invert_roll="dh" in cfg.robot.group_label.lower(),
            save_path=save_path,
            gen_gif=False,
            gen_imgs=True,
        )
    if cfg.gui:
        while True:
            time.sleep(0.1)

        pb.disconnect()


if __name__ == "__main__":
    main()
