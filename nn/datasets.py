import numpy as np
from torch.utils.data import Dataset

np.set_printoptions(precision=4)
from robots import PinBulletWrapper
from robots.bolt.BoltBullet import BoltBullet
from robots.solo.Solo12Bullet import Solo12Bullet
from robot_kinematic_symmetries import JointSpaceSymmetry


class COMMomentum(Dataset):

    def __init__(self, robot, size=5000, angular_momentum=False):

        # self.robot = robot
        self.size = size
        self.angular_momentum = angular_momentum
        dq_max = robot.velocity_limits
        q_min, q_max = robot.joint_pos_limits
        q, dq = robot.get_init_config(random=False)

        Y = np.zeros((size, 6), dtype=np.float32)
        X = np.zeros((size, robot.nj*2), dtype=np.float32)
        for i in range(size):
            q[7:] = np.random.uniform(q_min, q_max, robot.nj).astype(dtype=np.float32)
            dq[6:] = np.random.uniform(-dq_max, dq_max, robot.nj).astype(dtype=np.float32)
            hg = robot.pinocchio_robot.centroidalMomentum(q, dq).np.astype(dtype=np.float32)

            Y[i, :] = hg
            X[i, :] = np.concatenate((q[7:], dq[6:]))

        self.X = X
        self.Y = Y

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        if self.angular_momentum:
            return self.X[i, :], self.Y[i, :],
        return self.X[i, :], self.Y[i, :3],

    def get_hg(self, q, dq):
        hg = robot.pinocchio_robot.centroidalMomentum(q, dq)


