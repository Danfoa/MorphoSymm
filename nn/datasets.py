import copy
import functools
import itertools
import logging
import random
from typing import Optional

import numpy as np
import torch
from emlp import Group
from torch.utils.data import Dataset
import torch.nn.functional as F

import logging

from groups.SymmetricGroups import Sym

log = logging.getLogger(__name__)

np.set_printoptions(precision=4)

class Standarizer:

    def __init__(self, X_mean, X_std, Y_mean, Y_std):
        self.X_mean, self.X_std = torch.tensor(X_mean), torch.tensor(X_std)
        self.Y_mean, self.Y_std = torch.tensor(Y_mean), torch.tensor(Y_std)

    def normalize(self, x=None, y=None):
        if isinstance(x, np.ndarray):
            X_mean, X_std = self.X_mean.cpu().numpy(), self.X_std.cpu().numpy()
            Y_mean, Y_std = self.Y_mean.cpu().numpy(), self.Y_std.cpu().numpy()
        else:
            X_mean, X_std = self.X_mean, self.X_std
            Y_mean, Y_std = self.Y_mean, self.Y_std

        if x is not None and y is not None:
            return (x - X_mean) / X_std, (y - Y_mean) / Y_std
        elif x is not None:
            return (x - X_mean) / X_std
        elif y is not None:
            return (y - Y_mean) / Y_std

    def unstandarize(self, xn=None, yn=None):
        if isinstance(xn, np.ndarray):
            X_mean, X_std = self.X_mean.cpu().numpy(), self.X_std.cpu().numpy()
            Y_mean, Y_std = self.Y_mean.cpu().numpy(), self.Y_std.cpu().numpy()
        else:
            X_mean, X_std = self.X_mean, self.X_std
            Y_mean, Y_std = self.Y_mean, self.Y_std

        if xn is not None and yn is not None:
            return xn * X_std + X_mean, yn * Y_std + Y_mean
        elif xn is not None:
            return xn * X_std + X_mean
        elif yn is not None:
            return yn * Y_std + Y_mean


class COMMomentum(Dataset):

    def __init__(self, robot, Gin: Sym, Gout: Sym, size=5000, angular_momentum=False, normalize=True, augment=False,
                 dtype=np.float32):

        self.size = size
        self.dtype = dtype
        self.angular_momentum = angular_momentum
        self.robot = robot
        self.normalize = normalize

        self.Gin = Gin
        self.Gout = Gout
        self.group_actions = [(np.asarray(gin), np.asarray(gout)) for gin, gout in zip(self.Gin.discrete_actions,
                                                                                       self.Gout.discrete_actions)]
        self._pb = None  # GUI debug
        self.augment = augment

        dq_max = robot.velocity_limits
        q_min, q_max = robot.joint_pos_limits
        q, dq = robot.get_init_config(random=False)
        self.base_q = q[:7]
        self.base_dq = dq[:6]

        self.test_equivariance()

        Y = np.zeros((size, 6), dtype=self.dtype)
        X = np.zeros((size, robot.nj * 2), dtype=self.dtype)
        for i in range(size):
            q[7:] = np.random.uniform(q_min, q_max, robot.nj).astype(dtype=self.dtype)
            dq[6:] = np.random.uniform(-dq_max, dq_max, robot.nj).astype(dtype=self.dtype)
            hg = robot.pinocchio_robot.centroidalMomentum(q, dq).np.astype(dtype=self.dtype)

            Y[i, :] = hg
            X[i, :] = np.concatenate((q[7:], dq[6:]))

        X, Y = X.astype(dtype), Y.astype(dtype)
        # Normalize D
        X_mean, X_std, Y_mean, Y_std = self.compute_normalization(X, Y)
        self.standarizer = Standarizer(X_mean, X_std, Y_mean, Y_std)
        X, Y = self.standarizer.normalize(X, Y)
        self.X = X
        self.Y = Y[:, :(6 if angular_momentum else 3)]

        self.loss_fn = F.mse_loss

        log.info(f"CoM[{robot.__class__.__name__}]-Samples:{size}-Aug:{'True' if augment else 'False'}-"
                 f"Normalize:{normalize}")

    def compute_normalization(self, X, Y):
        idx = 6 if self.angular_momentum else 3
        X_mean, Y_mean, X_std, Y_std = 0., 0., 1., 1.
        if self.normalize:
            # TODO: Obtain analytic formula for mean and std along orbit of discrete and continuous groups.
            X_aug = np.vstack([X] + [np.asarray(g @ X.T).T for g in self.Gin.discrete_actions])
            Y_aug = np.vstack([Y[:, :idx]] + [np.asarray(g @ Y[:, :idx].T).T for g in self.Gout.discrete_actions])
            X_mean = np.mean(X_aug, axis=0)
            Y_mean = np.mean(Y_aug[:, :idx], axis=0)
            X_std = np.std(X_aug, axis=0)
            Y_std = np.std(Y_aug, axis=0)

        return X_mean, X_std, Y_mean, Y_std

    def test_equivariance(self):
        decimals = 15
        q, dq = self.robot.get_init_config(random=True)
        x = np.concatenate((q[7:]*1.4, dq[6:] * 10))
        x = np.round(x, decimals).astype(np.float64)
        y = self.get_hg(*np.split(x, 2))
        y = np.round(y, decimals).astype(np.float64)

        # Get all possible group actions
        for g_in, g_out in zip(self.Gin.discrete_actions, self.Gout.discrete_actions):
            gx, gy = np.round(g_in @ x, decimals), np.round(g_out @ y, decimals)
            gy_true = self.get_hg(*np.split(gx, 2))
            error = gy_true - gy
            if not np.allclose(error, 0.0, rtol=1e-3, atol=1e-3):
                try:
                    self.gui_debug(*np.split(x, 2), *np.split(gx, 2), hg1=y, hg2=gy_true, ghg2=gy)
                except Exception as e:
                    logging.warning(f"Unable to start GUI of pybullet: {str(e)}")
                raise AttributeError(f"Ground truth hg(q,dq) = Ag(q)dq is not equivariant to provided groups: \n" +
                                     f"x:{x}\ng*x:{gx}\ny:{y} \ng*y:{gy}\n" +
                                     f"Aq(g*q)g*dq:{gy_true}\nError:{error}")

    def compute_metrics(self, y, y_pred) -> dict:
        with torch.no_grad():
            metrics = {}

            y_dn = self.standarizer.unstandarize(yn=y.cpu())
            y_pred_dn = self.standarizer.unstandarize(yn=y_pred.cpu())

            lin, lin_pred = y_dn[:, :3], y_pred_dn[:, :3]
            metrics["lin_cos_sim"] = F.cosine_similarity(lin, lin_pred, dim=-1)
            metrics["lin_err"] = torch.linalg.norm(lin - lin_pred, dim=-1)

            if self.angular_momentum:
                ang, ang_pred = y_dn[:, 3:], y_pred_dn[:, 3:]
                metrics["ang_cos_sim"] = F.cosine_similarity(ang, ang_pred, dim=-1)
                metrics["ang_err"] = torch.linalg.norm(ang - ang_pred, dim=-1)
        return metrics

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        if self.angular_momentum:
            x, y = self.X[i, :], self.Y[i, :],
        else:
            x, y = self.X[i, :], self.Y[i, :3],

        if self.augment:  # Sample uniformly among symmetry actions including identity
            g_in, g_out = random.choice(self.group_actions)
            x, y = (g_in @ x).astype(self.dtype), (g_out @ y).astype(self.dtype)
        return x, y

    @property
    def augment(self):
        return self._augment

    @augment.setter
    def augment(self, new_normalize):
        assert isinstance(new_normalize, bool)
        self._augment = new_normalize
        log.info(f"Dataset Group Augmentation {'ACTIVATED' if self.augment else 'DEACTIVATED'}")

    def get_hg(self, q, dq):
        hg = self.robot.pinocchio_robot.centroidalMomentum(q=np.concatenate((self.base_q, q)),
                                                           v=np.concatenate((self.base_dq, dq)))
        hg = np.array(hg).astype(self.dtype)
        if self.angular_momentum:
            return hg
        return hg[:3]

    def gui_debug(self, q1, dq1, q2, dq2, hg1, hg2, ghg2):
        from utils.utils import configure_bullet_simulation

        def tint_robot(robot, color=(0.227, 0.356, 0.450), alpha=0.5):
            num_joints = self.robot.bullet_client.getNumJoints(self.robot.robot_id)
            for i in range(num_joints):
                self._pb.changeVisualShape(objectUniqueId=robot.robot_id, linkIndex=i, rgbaColor=color + (alpha,))
            self._pb.changeVisualShape(objectUniqueId=robot.robot_id, linkIndex=-1, rgbaColor=color + (alpha,))

        def draw_momentum_vector(p1, p2, v_color, scale=1.0, show_axes=True, text=None, offset=0.0):
            linewidth = 4
            x_color, y_color, z_color = (0, 1, 0), (1, 0, 0), (0, 0, 1)
            if show_axes:
                # x
                self._pb.addUserDebugLine(lineFromXYZ=p1, lineToXYZ=p1 + np.array([p2[0], 0., 0.]) * scale,
                                          lineColorRGB=x_color, lineWidth=linewidth, lifeTime=0)
                # y
                self._pb.addUserDebugLine(lineFromXYZ=p1, lineToXYZ=p1 + np.array([0., p2[1], 0.]) * scale,
                                          lineColorRGB=y_color, lineWidth=linewidth, lifeTime=0)
                # z
                self._pb.addUserDebugLine(lineFromXYZ=p1, lineToXYZ=p1 + np.array([0., 0., p2[2]]) * scale,
                                          lineColorRGB=z_color, lineWidth=linewidth, lifeTime=0)
            # v
            self._pb.addUserDebugLine(lineFromXYZ=p1, lineToXYZ=p1 + p2 * scale,
                                      lineColorRGB=v_color, lineWidth=linewidth, lifeTime=0)
            if text:
                self._pb.addUserDebugText(text=text, textPosition=p1 + p2 * scale + np.array([0, 0, 0.2 + offset]),
                                          textSize=1.2, lifeTime=0, textColorRGB=(0, 0, 0))
        if self._pb:
            self._pb.resetSimulation()
        if self._pb is None:
            self._pb = configure_bullet_simulation(gui=True)

        robot1 = self.robot
        robot2 = copy.copy(self.robot)
        offset = 1.0
        robot1.configure_bullet_simulation(self._pb, world=None)
        robot2.configure_bullet_simulation(self._pb, world=None)
        tint_robot(robot2, alpha=0.4)
        tint_robot(robot1, alpha=0.4)
        # Place robots in env
        q, dq = robot1.get_init_config(random=True)
        q[:7] = self.base_q
        dq[:6] = self.base_dq
        base_q1 = q[:7]
        base_q2 = np.array(base_q1)
        base_q2[1] += offset

        # Set positions:
        robot1.reset_state(np.concatenate((base_q1, q1)), dq)
        robot2.reset_state(np.concatenate((base_q2, q2)), dq)
        # Draw linear momentum
        draw_momentum_vector(base_q1[:3], hg1[:3], v_color=(0, 0, 0), scale=5.0, text=f"hg(q,dq)={hg1}")
        draw_momentum_vector(base_q2[:3], hg2[:3], v_color=(0, 0, 0), scale=5.0, text=f"hg(g*q, g*dq)={hg2}")
        draw_momentum_vector(base_q2[:3], ghg2[:3], v_color=(0.125, 0.709, 0.811), scale=5.0,
                             text=f"g*hg(q, dq)={ghg2}", show_axes=False, offset=0.2)
        print("a")

