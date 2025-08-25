import logging
import pathlib
from collections import OrderedDict
from enum import IntEnum

import numpy as np
import pinocchio as pin
from omegaconf import DictConfig

from morpho_symm.loaders.joint_data import JointData


log = logging.getLogger(__name__)


def load_robot(robot_cfg: DictConfig, q_zero: np.ndarray):
    """Loads a Pinocchio robot model from a given configuration."""
    if getattr(robot_cfg, "urdf_path", None) is not None:
        path = pathlib.Path(robot_cfg.urdf_path).absolute()
        assert path.exists(), f"Could not find URDF file at {path}"
        robot_model = pin.RobotWrapper.BuildFromURDF(
            filename=str(path),
            root_joint=pin.JointModelFreeFlyer(),
        )
    else:
        from robot_descriptions.loaders.pinocchio import load_robot_description

        robot_model = load_robot_description(
            f"{robot_cfg.name}_description",
            commit=getattr(robot_cfg, "commit", None),
            root_joint=pin.JointModelFreeFlyer(),
        )

    if q_zero is not None:
        robot_model.q0[7:] = q_zero

    joint_info = extract_pin_joint_info(robot_model)
    print("Loaded UDRF with the following joint configuration:")
    print(_format_joint_info_table(joint_info))

    return joint_info, robot_model


def extract_pin_joint_info(model: pin.RobotWrapper) -> OrderedDict[str, JointData]:
    """Extract joint information from a Pinocchio robot model.

    Args:
        model: Pinocchio RobotWrapper instance

    Returns:
        A dictionary with joint names as keys and JointData instances as values.
        Each JointData contains information about joint type, indices, and limits.
    """
    joint_info = OrderedDict()

    # Iterate through all joints in the pinocchio model
    for joint, joint_name in zip(model.model.joints, model.model.names):
        if joint.idx_q == -1:
            continue  # Ignore universe
        # Get joint type as string
        joint_type_str = joint.shortname()
        joint_type = PinJointType.from_str(joint_type_str)

        qpos_idx = np.arange(joint.idx_q, joint.idx_q + joint.nq)
        qvel_idx = np.arange(joint.idx_v, joint.idx_v + joint.nv)

        joint_range = None

        joint_info[joint_name] = JointData(
            name=joint_name,
            type=joint_type,
            body_id=-1,  # Not available in Pinocchio, leave blank
            nq=joint.nq,
            nv=joint.nv,
            qpos_idx=tuple(qpos_idx),
            qvel_idx=tuple(qvel_idx),
            range=joint_range,
            tau_idx=tuple(),  # Will be filled if actuators are available
            actuator_id=-1,  # Will be filled if actuators are available
        )

    return joint_info


def _format_joint_info_table(joint_info: OrderedDict[str, JointData]) -> str:
    """Format joint info as a table string."""
    lines = []

    # Calculate column widths based on content
    rows = []
    q_sdim, v_sdim = 0, 0
    for j_name, j_info in joint_info.items():
        joint_type_name = PinJointType.to_str(j_info.type)
        nq = str(j_info.nq)
        nv = str(j_info.nv)
        qpos_range = str(list(j_info.qpos_idx))
        qvel_range = str(list(j_info.qvel_idx))
        rows.append((j_name, joint_type_name, nq, nv, qpos_range, qvel_range))
        q_sdim += j_info.nq
        v_sdim += j_info.nv

    # Calculate column widths
    headers = ("Joint Name", "Type", "nq", "nv", "qpos dims", "qvel dims")
    col_widths = [max(len(str(item)) for item in col) for col in zip(headers, *rows)]

    # Build table
    header_row = " ".join(f"{headers[i]:<{col_widths[i] + 4}}" for i in range(len(headers)))
    lines.append(header_row)
    lines.append("-" * len(header_row))

    for row in rows:
        lines.append(" ".join(f"{row[i]:<{col_widths[i] + 4}}" for i in range(len(row))))

    return "\n".join(lines)


class PinJointType(IntEnum):
    """Pinocchio Joint Type enumeration based on joint.shortname() values."""

    UNKNOWN = -1
    FREE_FLYER = 0
    REVOLUTE_X = 1
    REVOLUTE_Y = 2
    REVOLUTE_Z = 3
    PRISMATIC_X = 4
    PRISMATIC_Y = 5
    PRISMATIC_Z = 6
    SPHERICAL_ZYX = 7
    PLANAR = 8
    REVOLUTE_UNBOUNDED = 9
    REVOLUTE_UNALIGNED = 10
    PRISMATIC_UNALIGNED = 11

    @staticmethod
    def from_str(joint_type: str) -> "PinJointType":  # noqa: D102
        if joint_type == "JointModelRX":
            return PinJointType.REVOLUTE_X
        elif joint_type == "JointModelRY":
            return PinJointType.REVOLUTE_Y
        elif joint_type == "JointModelRZ":
            return PinJointType.REVOLUTE_Z
        elif joint_type == "JointModelPX":
            return PinJointType.PRISMATIC_X
        elif joint_type == "JointModelPY":
            return PinJointType.PRISMATIC_Y
        elif joint_type == "JointModelPZ":
            return PinJointType.PRISMATIC_Z
        elif joint_type == "JointModelSphericalZYX":
            return PinJointType.SPHERICAL_ZYX
        elif joint_type == "JointModelFreeFlyer":
            return PinJointType.FREE_FLYER
        elif joint_type == "JointModelPlanar":
            return PinJointType.PLANAR
        elif joint_type == "JointModelRevoluteUnbounded":
            return PinJointType.REVOLUTE_UNBOUNDED
        elif joint_type == "JointModelRevoluteUnaligned":
            return PinJointType.REVOLUTE_UNALIGNED
        elif joint_type == "JointModelPrismaticUnaligned":
            return PinJointType.PRISMATIC_UNALIGNED
        else:
            return PinJointType.UNKNOWN

    @staticmethod
    def to_str(id: int) -> str:  # noqa: D102
        if id == PinJointType.REVOLUTE_X:
            return "REVOLUTE_X"
        elif id == PinJointType.REVOLUTE_Y:
            return "REVOLUTE_Y"
        elif id == PinJointType.REVOLUTE_Z:
            return "REVOLUTE_Z"
        elif id == PinJointType.PRISMATIC_X:
            return "PRISMATIC_X"
        elif id == PinJointType.PRISMATIC_Y:
            return "PRISMATIC_Y"
        elif id == PinJointType.PRISMATIC_Z:
            return "PRISMATIC_Z"
        elif id == PinJointType.SPHERICAL_ZYX:
            return "SPHERICAL_ZYX"
        elif id == PinJointType.FREE_FLYER:
            return "FREE_FLYER"
        elif id == PinJointType.PLANAR:
            return "PLANAR"
        elif id == PinJointType.REVOLUTE_UNBOUNDED:
            return "REVOLUTE_UNBOUNDED"
        elif id == PinJointType.REVOLUTE_UNALIGNED:
            return "REVOLUTE_UNALIGNED"
        elif id == PinJointType.PRISMATIC_UNALIGNED:
            return "PRISMATIC_UNALIGNED"
        else:
            return "UNKNOWN"


if __name__ == "__main__":
    import morpho_symm

    robot_name = "baxter"
    morpho_symm.load_symmetric_system(robot_name=robot_name)
