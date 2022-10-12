# coding: utf8
import copy
import pathlib
from math import pi
from typing import Collection, Tuple, Union, Optional, List

import numpy as np  # Numpy library
import scipy.spatial.transform
from pinocchio import Quaternion, Force, JointModelFreeFlyer
from pinocchio.robot_wrapper import RobotWrapper
from pybullet_utils.bullet_client import BulletClient

from .SoloBullet import Solo8Bullet
from ..PinBulletWrapper import PinBulletWrapper


class Solo12Bullet(Solo8Bullet):
    urdf_subpath = 'urdf/solo12.urdf'
    max_joint_acc = np.array([500, 570, 4000, 500, 570, 4000, 500, 570, 4000, 500, 570, 4000])  # [rad/s^2]

    JOINT_NAMES = ["FL_HAA", "FL_HFE", "FL_KFE", "FR_HAA", "FR_HFE", "FR_KFE",
                   "HL_HAA", "HL_HFE", "HL_KFE", "HR_HAA", "HR_HFE", "HR_KFE"]

    def __init__(self, resources: pathlib.Path, control_mode=True, power_coeff=1.0, reference_robot: Optional['PinBulletWrapper']=None,
                 gen_xacro=False, useFixedBase=False, **kwargs):

        # Super initialization: will call load_bullet_robot and load_pinocchio_robot
        super(Solo12Bullet, self).__init__(resources=resources, control_mode=control_mode,
                                           power_coeff=power_coeff, reference_robot=reference_robot,
                                           useFixedBase=useFixedBase, gen_xacro=gen_xacro, **kwargs)

        kps = [8.5, 7.5, 7.5] * 4
        kds = [0.3, 0.2, 0.2] * 4
        self._Kp = np.diagflat(kps)
        self._Kd = np.diagflat(kds)

    def get_observation(self, q=None, dq=None) -> Collection:
        return super(Solo12Bullet, self).get_observation(q, dq)

    @property
    def torque_limits(self, q=None, dq=None) -> [Collection]:
        return self._max_servo_torque

    @property
    def hip_height(self) -> float:
        return 0.24  # [m]

    @property
    def joint_names(self) -> List:
        return Solo12Bullet.JOINT_NAMES

    @property
    def mirrored_joint_names(self) -> List:
             # ["FL_HAA", "FL_HFE", "FL_KFE", "FR_HAA", "FR_HFE", "FR_KFE", "HL_HAA", "HL_HFE", "HL_KFE", "HR_HAA", "HR_HFE", "HR_KFE"]
        return ["FR_HAA", "FR_HFE", "FR_KFE", "FL_HAA", "FL_HFE", "FL_KFE", "HR_HAA", "HR_HFE", "HR_KFE", "HL_HAA", "HL_HFE", "HL_KFE"]

    @property
    def mirrored_endeff_names(self) -> Collection:
        return ["HR_ANKLE", "HL_ANKLE", "FR_ANKLE", "FL_ANKLE"]

    @property
    def mirror_joint_signs(self) -> List:
            # ["FR_HAA", "FR_HFE", "FR_KFE", "FL_HAA", "FL_HFE", "FL_KFE", "HR_HAA", "HR_HFE", "HR_KFE", "HL_HAA", "HL_HFE",  "HL_KFE"]
        return [-1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0]

    @property
    def endeff_names(self) -> List:
        return Solo12Bullet.EEF_NAMES

    def get_init_config(self, random=False):

        leg_pos = np.array([0.0, 0.8, -1.6])
        leg_pos_offset1 = np.random.rand(3) * [np.deg2rad(10), -np.deg2rad(15), -np.deg2rad(15)] if random else np.zeros(3)
        leg_pos_offset2 = np.random.rand(3) * [np.deg2rad(10), np.deg2rad(15), np.deg2rad(15)] if random else np.zeros(3)
        leg_vel = np.array([0.0, 0.0, 0.0])
        leg_vel_offset1 = np.random.uniform(-np.deg2rad(3), np.deg2rad(3), 3) if random else np.zeros(3)
        leg_vel_offset2 = np.random.uniform(-np.deg2rad(3), np.deg2rad(3), 3) if random else np.zeros(3)

        base_ori = [0, 0, 0, 1]
        if random:
            pitch = np.random.uniform(low=-np.deg2rad(5), high=np.deg2rad(5))
            roll = np.random.uniform(low=-np.deg2rad(5), high=np.deg2rad(5))
            base_ori = scipy.spatial.transform.Rotation.from_euler("xyz", [roll, pitch, 0]).as_quat()

        q_legs = np.concatenate((leg_pos + leg_pos_offset1,
                                 leg_pos + leg_pos_offset2 * [-1, 1, 1],
                                 leg_pos + leg_pos_offset2,
                                 leg_pos + leg_pos_offset1 * [-1, 1, 1]))
        dq_legs = np.concatenate((leg_vel + leg_vel_offset1, leg_vel + leg_vel_offset2, leg_vel + leg_vel_offset2,
                                  leg_vel + leg_vel_offset1))

        q0 = np.array([0., 0., self.hip_height + 0.02] + list(base_ori) + list(q_legs))
        dq0 = np.array([0., 0., 0., 0., 0., 0.] + list(dq_legs))

        if random and np.random.rand() > 0.5:
            q0, dq0 = self.mirror_base(q0, dq0)
            q0, dq0 = self.mirror_joints_sagittal(q0, dq0)

        return q0, dq0