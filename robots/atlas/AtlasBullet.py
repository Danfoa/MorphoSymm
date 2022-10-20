import copy
import os
import pathlib
import time
from contextlib import contextmanager
from math import pi
from typing import Collection, Tuple, Union, Optional, List

import numpy as np  # Numpy library
import scipy.spatial.transform
import torch
from pinocchio import Quaternion, Force, JointModelFreeFlyer
from pinocchio.robot_wrapper import RobotWrapper
from pybullet_utils.bullet_client import BulletClient

from ..AccelerationBounds import BatchedContraintAccelerationBound
from ..PinBulletWrapper import PinBulletWrapper, ControlMode
from robot_properties_solo.resources import Resources

# Terrible hack, should modify
from hydra.utils import get_original_cwd


class AtlasBullet(PinBulletWrapper):
    urdf_subpath = "atlas_v4_with_multisense.urdf"

    def __init__(self,  resources: pathlib.Path, control_mode=ControlMode('torque'), power_coeff=1.0,
                 reference_robot: Optional['PinBulletWrapper'] = None,
                 gen_xacro=False, useFixedBase=False, **kwargs):
        self._mass = np.NAN
        self.power_coeff = power_coeff
        if gen_xacro:
            raise NotImplementedError()

        # Super initialization: will call load_bullet_robot and load_pinocchio_robot
        super(AtlasBullet, self).__init__(resources=resources, recontrol_mode=control_mode, useFixedBase=useFixedBase,
                                          reference_robot=reference_robot, **kwargs)

        self._masses = [M.mass for M in self.pinocchio_robot.model.inertias][1:]

    def load_bullet_robot(self, base_pos=None, base_ori=None) -> int:
        if base_ori is None: base_ori = [0, 0, 0, 1]
        if base_pos is None: base_pos = [0, 0, 0.35]

        robot_id = super(AtlasBullet, self).load_bullet_robot(base_pos, base_ori)
        return robot_id

    def configure_bullet_simulation(self, bullet_client: BulletClient, world, base_pos=None, base_ori=None):
        super(AtlasBullet, self).configure_bullet_simulation(bullet_client, world, base_pos, base_ori)
        # Initial config
        q0, dq0 = self.get_init_config(random=False)
        q0[:2] += base_pos[:2] if base_pos is not None else [0, 0]
        self.reset_state(q0, dq0)

        self.bullet_ids_allowed_floor_contacts += []

    def get_observation(self, q=None, dq=None) -> Collection:
        return super(AtlasBullet, self).get_observation(q, dq)

    @property
    def torque_limits(self, q=None, dq=None) -> [Collection]:
        return self._max_servo_torque

    @property
    def acceleration_limits(self, q=None, dq=None) -> Union[float, Collection]:
        return self.max_joint_acc

    @property
    def velocity_limits(self, q=None, dq=None) -> Union[float, Collection]:
        self._joint_vel_limits = [12.00000, 9.00000, 12.00000, 12.00000, 12.00000, 12.00000, 12.00000, 12.00000,
                                  12.00000, 1.00000, 6.28000, 12.00000, 12.00000, 12.00000, 12.00000, 12.00000,
                                  12.00000, 1.00000, 12.00000, 12.00000, 12.00000, 12.00000, 12.00000, 12.00000,
                                  12.00000, 12.00000, 12.00000, 12.00000, 12.00000, 12.00000]
        return self._joint_vel_limits

    @property
    def joint_pos_limits(self) -> Tuple[Collection, Collection]:
        # Pinocchio cannot load the URDF values by default.
        self._joint_lower_limits = [-0.66322, -0.21939, -0.52360, -1.57080, -1.57080, 0.00000, 0.00000, 0.00000,
                                    -1.17810, -0.00100, -0.60214, -0.78540, -1.57080, 0.00000, -2.35619, 0.00000,
                                    -1.17810, -0.00100, -0.17436, -0.52360, -1.61234, 0.00000, -1.00000, -0.80000,
                                    -0.78679, -0.52360, -1.61234, 0.00000, -1.00000, -0.80000]
        self._joint_upper_limits = [0.66322, 0.53878, 0.52360, 0.78540, 1.57080, 3.14159, 2.35619, 3.14159, 1.17810,
                                    0.00100, 1.14319, 1.57080, 1.57080, 3.14159, 0.00000, 3.14159, 1.17810, 0.00100,
                                    0.78679, 0.52360, 0.65764, 2.35637, 0.70000, 0.80000, 0.17436, 0.52360, 0.65764,
                                    2.35637, 0.70000, 0.80000]
        return self._joint_lower_limits, self._joint_upper_limits

    @property
    def hip_height(self) -> float:
        return 1.0  # [m]

    @property
    def joint_names(self) -> List:
        return ['back_bkz', 'back_bky', 'back_bkx',
                'l_arm_shz', 'l_arm_shx', 'l_arm_ely', 'l_arm_elx', 'l_arm_wry', 'l_arm_wrx', 'l_arm_wry2',
                'neck_ry',
                'r_arm_shz', 'r_arm_shx', 'r_arm_ely', 'r_arm_elx', 'r_arm_wry', 'r_arm_wrx', 'r_arm_wry2',
                'l_leg_hpz', 'l_leg_hpx', 'l_leg_hpy', 'l_leg_kny', 'l_leg_aky', 'l_leg_akx',
                'r_leg_hpz', 'r_leg_hpx', 'r_leg_hpy', 'r_leg_kny', 'r_leg_aky', 'r_leg_akx']

    @property
    def endeff_names(self) -> List:
        return ['l_arm_wry2', 'r_arm_wry2', 'l_leg_akx', 'r_leg_akx']

    @property
    def mirrored_joint_names(self) -> List:
        return ['back_bkz', 'back_bky', 'back_bkx',
                'r_arm_shz', 'r_arm_shx', 'r_arm_ely', 'r_arm_elx', 'r_arm_wry', 'r_arm_wrx', 'r_arm_wry2',
                'neck_ry',
                'l_arm_shz', 'l_arm_shx', 'l_arm_ely', 'l_arm_elx', 'l_arm_wry', 'l_arm_wrx', 'l_arm_wry2',
                'r_leg_hpz', 'r_leg_hpx', 'r_leg_hpy', 'r_leg_kny', 'r_leg_aky', 'r_leg_akx',
                'l_leg_hpz', 'l_leg_hpx', 'l_leg_hpy', 'l_leg_kny', 'l_leg_aky', 'l_leg_akx', ]

    @property
    def mirrored_endeff_names(self) -> Collection:
        return ['r_arm_wry2', 'l_arm_wry2', 'r_leg_akx', 'l_leg_akx']

    @property
    def mirror_joint_signs(self) -> List:
        return [-1, 1, -1,  # ['back_bkz', 'back_bky', 'back_bkx',
                -1, -1, 1, -1, 1, -1, -1,
                # 'r_arm_shz', 'r_arm_shx', 'r_arm_ely', 'r_arm_elx', 'r_arm_wry', 'r_arm_wrx', 'r_arm_wry2',
                1,  # 'neck_ry',
                -1, -1, 1, -1, 1, -1, -1,
                # 'l_arm_shz', 'l_arm_shx', 'l_arm_ely', 'l_arm_elx', 'l_arm_wry', 'l_arm_wrx', 'l_arm_wry2',
                -1, -1, 1, 1, 1, -1,  # 'r_leg_hpz', 'r_leg_hpx', 'r_leg_hpy', 'r_leg_kny', 'r_leg_aky', 'r_leg_akx',
                -1, -1, 1, 1, 1, -1]  # 'l_leg_hpz', 'l_leg_hpx', 'l_leg_hpy', 'l_leg_kny', 'l_leg_aky', 'l_leg_akx',]

    def get_init_config(self, random=False):

        qj0, dqj0 = np.zeros(self.nj), np.zeros(self.nj)

        base_ori = [0, 0, 0, 1]
        if random:
            pitch = np.random.uniform(low=-np.deg2rad(7), high=np.deg2rad(0))
            base_ori = scipy.spatial.transform.Rotation.from_euler("xyz", [0, pitch, 0]).as_quat()

            dq_max = np.asarray(self.velocity_limits)
            q_min, q_max = (np.asarray(lim) for lim in self.joint_pos_limits)
            qj0 = np.random.uniform(q_min, q_max, size=None)
            dqj0 = np.random.uniform(-dq_max, dq_max, size=None)

        q0 = np.array([0., 0., self.hip_height] + list(base_ori) + list(qj0))
        dq0 = np.array([0., 0., 0., 0., 0., 0.] + list(dqj0))

        if random and np.random.rand() > 0.5:
            q0, dq0 = self.mirror_base(q0, dq0)
            q0, dq0 = self.mirror_joints_sagittal(q0, dq0)

        return q0, dq0

    @property
    def mass(self) -> float:
        if np.isnan(self._mass):
            return super().mass
        return self._mass


# Hack because I dont want to install ros to build a single description package.
@contextmanager
def cwd(path):
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)
