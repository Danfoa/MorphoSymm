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
from torch.utils.data._utils.collate import default_collate

import logging

from groups.SymmetricGroups import Sym

log = logging.getLogger(__name__)

np.set_printoptions(precision=4)

class Standarizer:

    def __init__(self, X_mean, X_std, Y_mean, Y_std):
        self.X_mean, self.X_std = torch.tensor(X_mean), torch.tensor(X_std)
        self.Y_mean, self.Y_std = torch.tensor(Y_mean), torch.tensor(Y_std)

    def transform(self, x=None, y=None):
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

    def __init__(self, robot, Gin: Sym, Gout: Sym, size=5000, angular_momentum=False, standarize=True, augment=False,
                 dtype=torch.float32):

        self.dtype = dtype
        self.angular_momentum = angular_momentum
        self.robot = robot
        self.normalize = standarize

        self.Gin = Gin
        self.Gout = Gout
        self.group_actions = [(np.asarray(gin), np.asarray(gout)) for gin, gout in zip(self.Gin.discrete_actions,
                                                                                       self.Gout.discrete_actions)]
        self.t_group_actions = [(torch.tensor(np.asarray(gin)), torch.tensor(np.asarray(gout))) for gin, gout in
                                zip(self.Gin.discrete_actions, self.Gout.discrete_actions)]
        self._pb = None  # GUI debug
        self.augment = augment

        dq_max = np.asarray(robot.velocity_limits)
        q_min, q_max = (np.asarray(lim) for lim in robot.joint_pos_limits)
        q, dq = robot.get_init_config(random=False)
        self.base_q = q[:7]
        self.base_dq = dq[:6]

        self.test_equivariance()

        Y = np.zeros((size, 6))
        X = np.zeros((size, robot.nj * 2))
        for i in range(size):
            q[7:] = np.random.uniform(q_min, q_max, size=None)
            dq[6:] = np.random.uniform(-dq_max, dq_max, size=None)
            hg = robot.pinocchio_robot.centroidalMomentum(q, dq)

            Y[i, :] = hg.np
            X[i, :] = np.concatenate((q[7:], dq[6:]))

        # Normalize D
        X_mean, X_std, Y_mean, Y_std = self.compute_normalization(X, Y)
        self.standarizer = Standarizer(X_mean, X_std, Y_mean, Y_std)
        X, Y = self.standarizer.transform(X, Y)
        self.X = X
        self.Y = Y

        self.loss_fn = F.mse_loss

        log.info(f"CoM[{robot.__class__.__name__}]-Samples:{size}-Aug:{'True' if augment else 'False'}-"
                 f"Normalize:{standarize}")

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
        trials = 5
        for trial in range(trials):
            q, dq = self.robot.get_init_config(random=True)
            q[:10] = 0.0
            dq[:9] = 0.0
            # Block arms
            # q[10:25], dq[10:25] = 0.0, 0.0
            # Block arms
            q[25:], dq[25:] = 0.0, 0.0

            # q[19:26], dq[19:26] = 0.0, 0.0
            x = np.concatenate((q[7:], dq[6:]))
            x = x.astype(np.float64)
            y = self.get_hg(*np.split(x, 2))
            y = y.astype(np.float64)


            # Get all possible group actions
            for g_in, g_out in zip(self.Gin.discrete_actions[1:], self.Gout.discrete_actions[1:]):
                gx, gy = np.asarray(g_in) @ x, np.asarray(g_out) @ y
                assert gx.dtype == x.dtype, (gx.dtype, x.dtype)
                assert gy.dtype == y.dtype, (gy.dtype, y.dtype)
                ggx, ggy = g_in.astype(np.float64) @ gx, g_out.astype(np.float64) @ gy
                # Check if there is numerical error in group actions application.
                action_error_x = ggx - x
                action_error_y = ggy - y
                gy_true = self.get_hg(*np.split(gx, 2))
                assert gy_true.dtype == y.dtype, (gy_true.dtype, y.dtype)
                error = gy_true - gy
                if not np.allclose(error, 0.0, rtol=1e-3, atol=1e-3):
                    try:
                        self.gui_debug(*np.split(x, 2), *np.split(gx, 2), hg1=y, hg2=gy_true, ghg2=gy)
                    except Exception as e:
                        logging.warning(f"Unable to start GUI of pybullet: {str(e)}")
                    raise AttributeError(f"Ground truth hg(q,dq) = Ag(q)dq is not equivariant to provided groups: \n" +
                                         f"x:{x}\ng*x:{gx}\ny:{y} \ng*y:{gy}\n" +
                                         f"Aq(g*q)g*dq:{gy_true}\nError:{error}")
        return None

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
        return self.Y.shape[0]

    def __getitem__(self, i):
        if self.angular_momentum:
            x, y = self.X[i, :], self.Y[i, :],
        else:
            x, y = self.X[i, :], self.Y[i, :3],

        # if self.augment:  # Sample uniformly among symmetry actions including identity
        #     g_in, g_out = random.choice(self.group_actions)
        #     x, y = (g_in @ x).astype(self.dtype), (g_out @ y).astype(self.dtype)
        return x, y

    def collate_fn(self, batch):
        # Enforce data type in batched array

        # Small hack to do batched augmentation. TODO: Although efficient this should be done somewhere else.
        x_batch, y_batch = default_collate(batch)

        if self.augment:  # Sample uniformly among symmetry actions including identity
            g_in, g_out = random.choice(self.t_group_actions[1:])
            g_x_batch = torch.matmul(x_batch.unsqueeze(1), g_in.unsqueeze(0).to(x_batch.dtype)).squeeze()
            g_y_batch = torch.matmul(y_batch.unsqueeze(1), g_out.unsqueeze(0).to(y_batch.dtype)).squeeze()
            # x, xx = x_batch[0], g_x_batch[0]
            # y, yy = y_batch[0], g_y_batch[0]
            x_batch, y_batch = g_x_batch, g_y_batch
        return [x_batch.to(self.dtype), y_batch.to(self.dtype)]

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
        hg = np.array(hg)
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
        offset = 2 * self.robot.hip_height
        robot1.configure_bullet_simulation(self._pb, world=None)
        robot2.configure_bullet_simulation(self._pb, world=None)
        # tint_robot(robot2, alpha=0.9)
        # tint_robot(robot1, alpha=0.9)
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
        draw_momentum_vector(base_q1[:3], hg1[:3], v_color=(0, 0, 0), scale=1/np.linalg.norm(hg1[:3]), text=f"hg(q,dq)={hg1}")
        draw_momentum_vector(base_q2[:3], hg2[:3], v_color=(0, 0, 0), scale=1/np.linalg.norm(hg2[:3]), text=f"hg(g*q, g*dq)={hg2}")
        draw_momentum_vector(base_q2[:3], ghg2[:3], v_color=(0.125, 0.709, 0.811), scale=1/np.linalg.norm(ghg2[:3]),
                             text=f"g*hg(q, dq)={ghg2}", show_axes=False, offset=0.2)
        print("a")

