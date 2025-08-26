import logging
import pathlib
import time
from collections import OrderedDict

import mujoco
import numpy as np
from omegaconf import DictConfig

from morpho_symm.loaders.joint_data import JointData

log = logging.getLogger(__name__)


def load_robot(robot_cfg: DictConfig, q_zero: np.ndarray) -> tuple[JointData, mujoco.MjModel]:
    """Loads a Mujoco robot model from a given configuration."""
    if getattr(robot_cfg, "mjcf_path", None) is not None:
        path = pathlib.Path(robot_cfg.mjcf_path).absolute()
        assert path.exists(), f"Could not find MJCF file at {path}"
        model = mujoco.MjModel.from_xml_path(str(path))
    else:
        from robot_descriptions.loaders.mujoco import load_robot_description

        model = load_robot_description(
            f"{robot_cfg.name}_description",
            variant=getattr(robot_cfg, "variant", None),
            commit=getattr(robot_cfg, "commit", None),
        )

    if q_zero is not None:
        model.qpos0[7:] = q_zero

    joint_info = extract_mj_joint_info(model)
    print("Loaded MJCF with the following joint configuration:")
    print(_format_joint_info_table(joint_info))

    return joint_info, model


def extract_mj_joint_info(model: mujoco.MjModel) -> OrderedDict[str, "JointData"]:
    """Returns the joint-space information of the model.

    Thanks to the obscure Mujoco API, this function tries to do the horrible hacks to get the joint information
    we need to do a minimum robotics project with a rigid body system.

    Returns:
    -------
            A dictionary with the joint names as keys and the JointInfo namedtuple as values.
            each JointInfo namedtuple contains the following fields:
            - name: The joint name.
            - type: The joint type (mujoco.mjtJoint).
            - body_id: The body id to which the joint is attached.
            - range: The joint range.
            - nq: The number of joint position variables.
            - nv: The number of joint velocity variables.
            - qpos_idx: The indices of the joint position variables in the qpos array.
            - qvel_idx: The indices of the joint velocity variables in the qvel array.
    """
    joint_info = OrderedDict()
    for joint_id in range(model.njnt):
        # Get the starting index of the joint name in the model.names string
        name_start_index = model.name_jntadr[joint_id]
        # Extract the joint name from the model.names bytes and decode it
        joint_name = model.names[name_start_index:].split(b"\x00", 1)[0].decode("utf-8")
        joint_type = model.jnt_type[joint_id]
        qpos_idx_start = model.jnt_qposadr[joint_id]
        qvel_idx_start = model.jnt_dofadr[joint_id]

        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            joint_nq, joint_nv = 7, 6
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            joint_nq, joint_nv = 4, 3
        elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE or joint_type == mujoco.mjtJoint.mjJNT_HINGE:
            joint_nq, joint_nv = 1, 1
        else:
            raise RuntimeError(f"Unknown mujoco joint type: {joint_type} available {mujoco.mjtJoint}")

        qpos_idx = np.arange(qpos_idx_start, qpos_idx_start + joint_nq)
        qvel_idx = np.arange(qvel_idx_start, qvel_idx_start + joint_nv)

        joint_info[joint_name] = JointData(
            name=joint_name,
            type=joint_type,
            body_id=model.jnt_bodyid[joint_id],
            range=model.jnt_range[joint_id],
            nq=joint_nq,
            nv=joint_nv,
            qpos_idx=qpos_idx,
            qvel_idx=qvel_idx,
        )

    # Iterate over all actuators
    current_dim = 0
    for acutator_idx in range(model.nu):
        name_start_index = model.name_actuatoradr[acutator_idx]
        act_name = model.names[name_start_index:].split(b"\x00", 1)[0].decode("utf-8")
        mj_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
        # Get the joint index associated with the actuator
        joint_id = model.actuator_trnid[mj_actuator_id, 0]
        # Get the joint name from the joint index
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)

        # Add the actuator indx to the joint_info
        joint_info[joint_name].actuator_id = mj_actuator_id
        joint_info[joint_name].tau_idx = tuple(range(current_dim, current_dim + joint_info[joint_name].nv))
        current_dim += joint_info[joint_name].nv
    return joint_info


def _format_joint_info_table(joint_info: OrderedDict[str, "JointData"]) -> str:
    """Format joint info as a table string."""
    lines = []

    # Helper function to get joint type name
    def get_joint_type_name(joint_type):
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            return "FREE_FLYER"
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            return "SPHERICAL"
        elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
            return "PRISMATIC"
        elif joint_type == mujoco.mjtJoint.mjJNT_HINGE:
            return "REVOLUTE"
        else:
            return f"UNKNOWN({joint_type})"

    # Calculate column widths based on content
    rows = []
    q_sdim, v_sdim = 0, 0
    for j_name, j_info in joint_info.items():
        joint_type_name = get_joint_type_name(j_info.type)
        qpos_range = str(list(j_info.qpos_idx))
        qvel_range = str(list(j_info.qvel_idx))
        rows.append((j_name, joint_type_name, qpos_range, qvel_range))
        q_sdim += j_info.nq
        v_sdim += j_info.nv

    # Calculate column widths
    headers = ("Joint Name", "Type", "qpos dims", "qvel dims")
    col_widths = [max(len(str(item)) for item in col) for col in zip(headers, *rows)]

    # Build table
    header_row = " ".join(f"{headers[i]:<{col_widths[i] + 4}}" for i in range(len(headers)))
    lines.append(header_row)
    lines.append("-" * len(header_row))

    for row in rows:
        lines.append(" ".join(f"{row[i]:<{col_widths[i] + 4}}" for i in range(len(row))))

    return "\n".join(lines)


def debug_joints(robot_cfg: DictConfig):
    """Launch MuJoCo viewer with joint sliders for debugging robot configuration.

    Args:
        robot_cfg: Robot configuration containing model path and parameters
    """
    try:
        import mujoco.viewer
    except ImportError:
        log.error("mujoco.viewer not available. Install with: pip install mujoco[viewer]")
        return

    # Load the robot model
    if robot_cfg.q_zero is not None:
        q_zero = np.array([eval(str(s)) for s in robot_cfg.q_zero], dtype=float)
    else:
        q_zero = None

    joint_info, model = load_robot(robot_cfg, q_zero)

    # Print model info for debugging
    print("\nModel Info:")
    print(f"- Joints: {model.njnt}")
    print(f"- Actuators: {model.nu}")
    print(f"- DOFs: {model.nv}")

    # Create MuJoCo data
    data = mujoco.MjData(model)

    # Disable gravity and damping for easier manipulation
    model.opt.gravity[:] = 0

    # Disable integrator to prevent drift
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    model.opt.timestep = 0.001

    # Forward kinematics to update the state
    mujoco.mj_forward(model, data)

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Set white background using model options
            model.vis.rgba.fog[0:3] = [1.0, 1.0, 1.0]  # White fog color
            model.vis.rgba.fog[3] = 1.0  # Full opacity

            # Disable skybox and set background
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SKYBOX] = 0
            viewer.scn.background[0:4] = [1.0, 1.0, 1.0, 1.0]  # White background RGBA

            # Enable UI panels
            if hasattr(viewer, "ui"):
                viewer.ui.show_info = True
                if model.nu > 0:
                    viewer.ui.show_control = True

            while viewer.is_running():
                # Step the simulation to update actuator controls
                mujoco.mj_step(model, data)
                viewer.sync()
                # Lower refresh rate to prevent issues
                time.sleep(1 / 60)  # 60 Hz refresh rate

    except KeyboardInterrupt:
        log.info("Viewer closed by user")
    except Exception as e:
        log.error(f"Viewer error: {e}")
        log.info("Trying fallback viewer method...")
        try:
            # Use the blocking viewer which might have better UI support
            mujoco.viewer.launch(model, data)
        except Exception as e2:
            log.error(f"All viewer methods failed: {e2}")


if __name__ == "__main__":
    import morpho_symm

    robot_name = "tiago++_mj"
    morpho_symm.load_symmetric_system(robot_name=robot_name, debug=True)
