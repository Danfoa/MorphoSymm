import copy
import logging
from abc import ABC, abstractmethod
from typing import Collection, Optional, Tuple, Union

import numpy as np
import scipy
from pinocchio import JointModelFreeFlyer, RobotWrapper
from pinocchio import pinocchio_pywrap as pin
from robot_descriptions.loaders.pinocchio import load_robot_description as pin_load_robot_description
from scipy.linalg import inv

log = logging.getLogger(__name__)

NameList = Collection[str]
Vector = Collection[float]
State = Tuple[np.ndarray, np.ndarray]


class PinSimWrapper(ABC):

    def __init__(self, robot_name: str, endeff_names: Optional[NameList] = None, fixed_base=False,
                 reference_robot: Optional['PinSimWrapper'] = None, hip_height=1.0, init_q=None, q_zero=None):
        """Initializes the wrapper.

        Args:
            robot_name (str): Name of this robot instance
            endeff_names (List[str]): Names of the end-effectors.
            fixed_base (bool, optional): Determines if the robot base if fixed. Defaults to False.
            reference_robot (PinBulletWrapper, optional): Instance to copy the pinocchio model from. Defaults to None.
            hip_height (float): Height of the hip of the robot. Hip height on resting configuration.
            init_q (List[float]): Initial configuration of the robot of length nq.
            q_zero (List[float]): Zero configuration. This can be different from the Zero config defined in the URDF.
        """
        self.robot_name = str.lower(robot_name)
        self.fixed_base = fixed_base
        self._endeff_names = endeff_names
        self.hip_height = hip_height

        # Default to URDF values
        self._qj_high_limit, self._qj_low_limit, self._dqj_limit = None, None, None
        # Initialize Pinocchio Robot.
        self.load_pinocchio_robot(reference_robot)

        self.n_js = self.nq - 7
        self.nf = len(self.endeff_names) if self.endeff_names is not None else -1

        # Mappings between joint names pin and bullet ids, and pinocchio generalized q and dq coordinates
        self.pin_joint_space = {}
        self.joint_space_names = []
        self._q_zero = np.zeros(self.nq)
        for joint, joint_name in zip(self.pinocchio_robot.model.joints, self.pinocchio_robot.model.names):
            if joint.idx_q == -1: continue  # Ignore universe
            if joint.nq == 7: continue  # Ignore floating-base
            pin_joint_type = joint.shortname()
            log.debug(f"[{pin_joint_type}]:{joint_name} - DoF(nq):{joint.nq}, idx_q:{joint.idx_q}, idx_v:{joint.idx_v}")

            self.joint_space_names.append(joint_name)
            low_lim = self.pinocchio_robot.model.lowerPositionLimit[joint.idx_q:joint.idx_q + joint.nq]
            high_lim = self.pinocchio_robot.model.upperPositionLimit[joint.idx_q:joint.idx_q + joint.nq]
            self.pin_joint_space[joint_name] = JointWrapper(type=pin_joint_type,
                                                            nq=joint.nq, nv=joint.nv,
                                                            idx_q=joint.idx_q, idx_v=joint.idx_v,
                                                            pos_limit_low=low_lim, pos_limit_high=high_lim)

            q_j_zero, _ = self.pin_joint_space[joint_name].zero()
            # self._q_zero[joint.idx_q:joint.idx_q + joint.nq] = q_j_zero

        self._q_zero = self._q_zero if q_zero is None else np.array(q_zero)
        assert self._q_zero.size == self.nq, f"Expected |q_0|=3+4+nj={self.nq}, but received {q_zero}"
        self._init_q = np.concatenate((np.zeros(6), [1], np.zeros(self.n_js))) if init_q is None else np.array(init_q)
        assert len(self._init_q) == self.nq, f"Expected |q_init|=3+4+nj={self.nq}, but received {self._init_q.size}"
        # TODO: ensure joint state is within configuration space.

        self.joint_space = {}
        log.debug(f"Robot loaded {self}")

    def get_state(self):
        """Fetch state of the system from a physics simulator and return pinocchio convention (q, dq).

        Obtains the system state from the simulator and returns the generalized positions and velocities,
        following the pinocchio convention for q and dq.

        Returns:
            q (ndarray): Generalized position coordinates of shape (self.nq,) in pinocchio convention.
            dq (ndarray): Generalized velocity coordinates of shape (self.nv,) in pinocchio convention.
        """
        q_sim, dq_sim = self.get_state_sim()
        q_pin, dq_pin = self.sim2pin(q_sim, dq_sim)
        q_pin_centered = self.center_state(q_pin)
        return q_pin_centered, dq_pin

    @abstractmethod
    def get_state_sim(self) -> State:
        """Get the system state in simulator convention.

        Returns:
             q (ndarray): Generalized position coordinates of shape in simulator convention.
            dq (ndarray): Generalized velocity coordinates of shape in simulator convention.
        """

    def reset_state(self, q, v, update_pin=False) -> None:
        """Sets the state of the system in the simulator and Pinocchio if requested.

        Args:
            q (ndarray): Generalized position coordinates of shape (self.nq,) in pinocchio convention.
            v (ndarray): Generalized velocity coordinates of shape (self.nv,) in pinocchio convention.
            update_pin (bool): If True, update the pinocchio model with the new state.
        """
        q_centered = self.center_state(q) # - self._q_zero

        assert np.isclose(np.linalg.norm(q_centered[3:7]), 1), f"Invalid base orientation quaterion with norm " \
                                                               f":{np.linalg.norm(q_centered[3:7]):.2f}"

        q_sim, dq_sim = self.pin2sim(q_centered, v)
        self.reset_state_sim(q_sim, dq_sim)
        if update_pin:
            self.update_pinocchio(q_centered, v)

    @abstractmethod
    def reset_state_sim(self, q, v) -> None:
        """Reset the state of the robot in simulation.

        Returns:
             q (ndarray): Generalized position coordinates of shape in simulator convention.
            dq (ndarray): Generalized velocity coordinates of shape in simulator convention.
        """

    @abstractmethod
    def sim2pin(self, q, dq) -> State:
        """Converts system state configuration from simulator to pinocchio convention.

        Args:
             q (ndarray): Generalized position coordinates of shape in simulator convention.
            dq (ndarray): Generalized velocity coordinates of shape in simulator convention.

        Returns:
            q (ndarray): Generalized position coordinates of shape (self.nq,) in pinocchio convention.
            dq (ndarray): Generalized velocity coordinates of shape (self.nv,) in pinocchio convention.
        """

    @abstractmethod
    def pin2sim(self, q, dq) -> State:
        """Converts system state configuration from simulator to pinocchio convention.

        Args:
            q (ndarray): Generalized position coordinates of shape (self.nq,) in pinocchio convention.
            dq (ndarray): Generalized velocity coordinates of shape (self.nv,) in pinocchio convention.

        Returns:
             q (ndarray): Generalized position coordinates of shape in simulator convention.
            dq (ndarray): Generalized velocity coordinates of shape in simulator convention.
        """

    def center_state(self, q) -> Vector:
        """If there is a defined Zero configuration this is used for centering the pinocchio state position coordinates.

        This function handles the centering of the state configuration position coordinates, considering the topology
        of the joints configuration space. That is if the joint is a continuous revolute or a spherical, the
        configuration of the joint is a point in the unit circle or a unit quaterion, respectively. In this case,
        the substraction of two points in these topological spaces has to be handled accordingly.

        Args:
            q (ndarray): Generalized position coordinates of shape (self.nq,) in pinocchio convention.

        Returns:
            q_centered (ndarray): Generalized position coordinates of shape (self.nq,) in pinocchio convention.
        """
        q_centered = np.array(q)
        # If no custom zero configuration is defined return the input configuration.
        if np.sum(self._q_zero) == 0:
            return q

        for joint_name, joint in self.pin_joint_space.items():
            idx_q, idx_v = joint.state_idx
            qj = joint.add_configuration(q[idx_q], self._q_zero[idx_q])
            q_centered[idx_q] = qj

        return q_centered

    def uncenter_state(self, q) -> Vector:
        """If there is a defined Zero configuration this is used for uncenter the pinocchio state position coordinates.

        See center_state for more details.

        Args:
            q (ndarray): Generalized position coordinates of shape (self.nq,) in pinocchio convention.

        Returns:
            q_centered (ndarray): Generalized position coordinates of shape (self.nq,) in pinocchio convention.
        """
        q_uncentered = np.array(q)
        # If no custom zero configuration is defined return the input configuration.
        if np.sum(self._q_zero) == 0:
            return q

        for joint_name, joint in self.pin_joint_space.items():
            idx_q, idx_v = joint.state_idx()
            qj = joint.substract_configuration(q[idx_q], self._q_zero[idx_q])
            q_uncentered[idx_q] = qj

        return q_uncentered

    def update_pinocchio(self, q, dq):
        """Updates the pinocchio robot.

        This includes updating:
        - kinematics
        - joint and frame Jacobian
        - Centroidal momentum

        Args:
            q (ndarray): Generalized position coordinates of shape (self.nq,) in pinocchio convention.
            dq (ndarray): Generalized velocity coordinates of shape (self.nv,) in pinocchio convention.
        """
        self.pinocchio_robot.forwardKinematics(q, dq)
        self.pinocchio_robot.framesForwardKinematics(q)

        pin.ccrba(self.pinocchio_robot.model, self.pinocchio_robot.data, q, dq)
        pin.computeKineticEnergy(self.pinocchio_robot.model, self.pinocchio_robot.data)
        pin.computePotentialEnergy(self.pinocchio_robot.model, self.pinocchio_robot.data)
        inv(self.pinocchio_robot.data.Ig) @ self.pinocchio_robot.data.hg

    def load_pinocchio_robot(self, reference_robot: Optional['PinSimWrapper'] = None):
        """Function to load and configure the pinocchio instance of your robot.

        Load the pinocchio robot from the URDF file and expose it for user use in the class property `pinocchio_robot`.


        Args:
            reference_robot: Instance of your robot class. If provided, the pinocchio robot will be copied from the
            reference avoiding copying the robot `model`.


        Returns:
            RobotWrapper: Instance of your pinocchio robot.
        """
        if reference_robot is not None:
            import sys
            pin_robot = copy.copy(reference_robot.pinocchio_robot)
            pin_robot.data = copy.copy(reference_robot.pinocchio_robot.data)
            assert sys.getrefcount(pin_robot.data) <= 2
        else:
            pin_robot = pin_load_robot_description(f"{self.robot_name}_description", root_joint=JointModelFreeFlyer())
        self._mass = float(np.sum([i.mass for i in pin_robot.model.inertias]))  # [kg]
        self._pinocchio_robot = pin_robot

    @property
    def pinocchio_robot(self) -> RobotWrapper:
        """Get a handle of the pinocchio robot instance.

        Returns:
            pin_robot (RobotWrapper): Pinocchio robot instance.
        """
        if self._pinocchio_robot is None:
            raise AttributeError("Pinocchio robot has not been loaded")
        return self._pinocchio_robot

    @property
    def mass(self) -> float:
        """Get the total mass of the robot.

        Returns:
            mass (float): Total mass of the robot in [kg] computed from the robot's URDF file.
        """
        return float(np.sum([i.mass for i in self.pinocchio_robot.model.inertias]))  # [kg]

    @property
    def velocity_limits(self, q=None, dq=None) -> np.ndarray:
        """Get the velocity limits of the robot.

        Args:
            q: Generalized positions (nq,).
            dq: Generalized velocities (nv,).

        Returns:
            vel_lims (ndarray): Velocity limits of shape (self.nv,).
        """
        if self.pinocchio_robot is None:
            raise AttributeError("Pinocchio robot has not been loaded")
        elif self._dqj_limit is None:
            self._dqj_limit = []
            for joint_name in self.joint_names:
                vel_limit = self.pin_joint_space[joint_name].vel_lim
                self._dqj_limit.append(vel_limit)
            self._dqj_limit = np.asarray(self._dqj_limit).flatten()
        return self._dqj_limit

    @property
    def joint_pos_limits(self, q=None, dq=None):
        """Get the joint position limits of the robot.

        Args:
            q: generalized positions (nq,).
            dq: generalized velocities (nv,).

        Returns:
            qj_low_limit (ndarray): Lower joint position limits of shape (self.nj,).
        """
        if self.pinocchio_robot is None:
            raise AttributeError("Pinocchio robot has not been loaded")
        elif self._qj_low_limit is None or self._qj_high_limit is None:
            self._qj_high_limit, self._qj_low_limit = [], []
            for joint_name in self.joint_names:
                low, high = self.pin_joint_space[joint_name].pos_lims
                self._qj_low_limit.append(low)
                self._qj_high_limit.append(high)
            self._qj_high_limit = np.asarray(self._qj_high_limit).flatten()
            self._qj_low_limit = np.asarray(self._qj_low_limit).flatten()
        return self._qj_low_limit, self._qj_high_limit

    @property
    def endeff_names(self) -> Collection[str]:
        """Returns the names of the end-effectors of the robot.

        Returns:
            Collection: List of end-effector names.
        """
        return self._endeff_names

    def get_init_config(self, random=False, angle_sweep=None, fix_base=False):
        """Get initial configuration of the robot.

        Args:
            random (bool): if True, randomize the initial configuration.
            angle_sweep (float): if not None, randomize the initial configuration within the given angle sweep.
            fix_base (bool): if True, do not randomize base orientation.

        Returns:
            q (ndarray): generalized positions .
            dq (ndarray): generalized velocities.
        """
        q = self._init_q
        v = np.zeros(self.nv)

        if random:
            pitch = np.random.uniform(low=-np.deg2rad(25), high=np.deg2rad(25)) if not fix_base else 0
            roll = np.random.uniform(low=-np.deg2rad(25), high=np.deg2rad(25)) if not fix_base else 0
            yaw = np.random.uniform(low=-np.deg2rad(25), high=np.deg2rad(25)) if not fix_base else 0

            base_ori = scipy.spatial.transform.Rotation.from_euler("xyz", np.array([roll, pitch, yaw])).as_quat()

            for joint_name, joint in self.pin_joint_space.items():
                idx_q, idx_v = joint.state_idx
                q_j, v_j = joint.random_configuration(max_range=angle_sweep)
                q[idx_q] = joint.add_configuration(q[idx_q], q_j)
                v[idx_v] += v_j
            q[3:7] = base_ori

        return q, v

    @property
    def nq(self) -> int:
        """The dimension of the generalized positions coordinates of the robot."""
        return self.pinocchio_robot.nq

    @property
    def nv(self) -> int:
        """The dimension of the generalized velocity coordinates of the robot."""
        return self.pinocchio_robot.nv

    @classmethod
    def from_instance(other: 'PinSimWrapper') -> 'PinSimWrapper':
        """Creates another instance of this robot wrapper without duplicating the model or data from pinocchio robot.

        This is usefull when we want to spawn multiple instances of the same robot on the physics simulator.


        Args:
            other (PinBulletWrapper): The instance from which to get the model and data for the pinocchio model

        Returns:
            PinBulletWrapper: A new instance of this robot wrapper
        """
        raise NotImplementedError("This method is not implemented yet")

    def __repr__(self):
        """."""
        return f"{self.robot_name}-nq:{self.nq}-nv:{self.nv}"


class JointWrapper:

    def __init__(self, type: Union[str, int], idx_q: int, idx_v: int, nq: int, nv: int,
                 pos_limit_low: Union[list[float], float] = -np.inf,
                 pos_limit_high: Union[list[float], float] = np.inf):
        self.type = type
        self.idx_q = idx_q
        self.idx_v = idx_v
        self.nq = nq
        self.nv = nv
        self.pos_limit_low = pos_limit_low
        self.pos_limit_high = pos_limit_high

    def random_configuration(self, max_range=None):
        """Random configuration for the joint considering dimensions of the position space and tangent space.

        Returns:
            q (ndarray): Random configuration of the joint.
            v (ndarray): Zero velocity.
        """
        if self.nq == 1:
            theta = np.random.uniform(self.pos_limit_low, self.pos_limit_high)
            theta = min(max_range, theta)
            theta = max(-max_range, theta)
            return theta, np.array([0])
        if self.nq == 2 and self.nv == 1:  # Unit circle
            max_range = np.pi if max_range is None else max_range
            theta = np.random.uniform(-max_range, max_range)
            return np.array([np.cos(theta), np.sin(theta)]), np.array([0])
        if self.nq == 4 and self.nv == 3:  # Spherical joint
            # Generate random quaternion
            u = np.random.uniform(-1, 1, size=3)
            theta1 = max_range * u[0]
            theta2 = max_range * u[1]
            theta3 = max_range * u[2]

            w = np.cos(theta3 / 2) * np.sin(theta2 / 2) * np.sin(theta1 / 2) + np.sin(theta3 / 2) * np.cos(
                theta2 / 2) * np.cos(theta1 / 2)
            x = np.sin(theta3 / 2) * np.cos(theta2 / 2) * np.sin(theta1 / 2) - np.cos(theta3 / 2) * np.sin(
                theta2 / 2) * np.cos(theta1 / 2)
            y = np.cos(theta3 / 2) * np.cos(theta2 / 2) * np.sin(theta1 / 2) + np.sin(theta3 / 2) * np.sin(
                theta2 / 2) * np.cos(theta1 / 2)
            z = np.cos(theta3 / 2) * np.sin(theta2 / 2) * np.cos(theta1 / 2) - np.sin(theta3 / 2) * np.cos(
                theta2 / 2) * np.sin(theta1 / 2)

            return np.array([x, y, z, w]), np.array([0, 0, 0])

    def zero(self) -> State:
        if self.nq == 1:
            return np.array([0]), np.array([0])
        if self.nq == 2 and self.nv == 1:  # Unit circle
            return np.array([1, 0]), np.array([0])
        if self.nq == 4 and self.nv == 3:  # Spherical joint
            return np.array([0, 0, 0, 1]), np.array([0, 0, 0])

    def add_configuration(self, q1, q2) -> State:
        # TODO: handle these through pinocchio
        if self.nq == 1:
            return q1 + q2
        if self.nq == 2 and self.nv == 1:  # Unit circle
            theta1 = np.arctan2(q1[1], q1[0])
            theta2 = np.arctan2(q2[1], q2[0])
            theta = theta1 + theta2
            return np.array([np.cos(theta), np.sin(theta)])
        if self.nq == 4 and self.nv == 3:  # Spherical joint
            raise NotImplementedError()

    def substract_configuration(self, q1, q2) -> State:
        # TODO: handle these through pinocchio
        if self.nq == 1:
            return q1 - q2
        if self.nq == 2 and self.nv == 1:  # Unit circle
            theta1 = np.arctan2(q1[1], q1[0])
            theta2 = np.arctan2(q2[1], q2[0])
            theta = theta1 - theta2
            return np.array([np.cos(theta), np.sin(theta)])
        if self.nq == 4 and self.nv == 3:  # Spherical joint
            raise NotImplementedError()

    @property
    def state_idx(self) -> Tuple[list[int], list[int]]:
        idx_q = list(range(self.idx_q, self.idx_q + self.nq))
        idx_v = list(range(self.idx_v, self.idx_v + self.nv))
        return idx_q, idx_v

class SimPinJointWrapper(ABC):

    def __init__(self) -> object:
        self._pin_joint = None
        self._sim_joint = None

    def sim2pin(self, q, dq) -> State:
        return q, dq

    def pin2sim(self, q, dq) -> State:
        return q, dq
