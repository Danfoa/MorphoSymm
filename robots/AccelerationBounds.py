#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/11/21
# @Author  : Daniel Ordonez 
# @email   : daniels.ordonez@gmail.com
import copy
from typing import Tuple, Sequence, Collection

import numpy as np
import torch


class ContrainAccelerationBound:

    nn = np
    vector = np.array

    def __init__(self, q_max: Collection, q_min: Collection, qdot_max: Collection, ddq_max: Collection, dt,
                 dt_scaler=2.0):
        self.nq = len(q_max)
        self.q_max = np.array(q_max)
        self.q_min = np.array(q_min)
        self.qdot_max = np.array(qdot_max)
        self.ddq_max = np.array(ddq_max)
        self.dt = copy.copy(dt * dt_scaler)

    def acceletation_bounds(self, q, dq) -> Tuple[Sequence, Sequence]:
        ddq_ub = np.zeros((4, self.nq))
        ddq_lb = np.zeros((4, self.nq))
        ddq_ub[3, ...] = self.ddq_max
        ddq_lb[3, ...] = -self.ddq_max
        ddq_lb[0, ...], ddq_ub[0, :] = self.acceleration_bounds_from_pos_limits(q, dq)
        ddq_lb[1, ...] = (-self.qdot_max - dq) / self.dt
        ddq_ub[1, ...] = (self.qdot_max - dq) / self.dt
        ddq_lb[2, ...], ddq_ub[2, :] = self.acceleration_bounds_from_viability(q, dq)

        ddq_lb_f = np.max(ddq_lb, axis=0)
        ddq_ub_f = np.min(ddq_ub, axis=0)
        # Ensure that in cases of extreme accelerations the feasible original bounds are not violated.
        ddq_lb_f = np.clip(ddq_lb_f, a_min=-self.ddq_max, a_max=self.ddq_max)
        ddq_ub_f = np.clip(ddq_ub_f, a_min=-self.ddq_max, a_max=self.ddq_max)
        return ddq_lb_f, ddq_ub_f

    def acceleration_bounds_from_pos_limits(self, q, qdot):
        q1ddot_M = -qdot/self.dt
        q2ddot_M = -qdot**2 / (2*(self.q_max - q))
        q3ddot_M = 2*(self.q_max - q - self.dt*qdot) / (self.dt**2)

        q2ddot_m = qdot**2 / (2*(q - self.q_min))
        q3ddot_m = 2*(self.q_min - q - self.dt*qdot) / (self.dt**2)

        ddq_lb = np.ones_like(q) * -self.ddq_max
        ddq_ub = np.ones_like(q) * self.ddq_max

        for i, xdot in enumerate(qdot):
            ddq_lb[i] = q3ddot_m[i]  # Max deceleration to stop at lower bound
            if xdot >= 0:
                if q3ddot_M[i] > q1ddot_M[i]:
                    ddq_ub[i] = q3ddot_M[i]
                else:
                    ddq_ub[i] = min(q1ddot_M[i], q2ddot_M[i])
            else:
                ddq_ub[i] = q3ddot_M[i] # Max acceleration to stop at lower bound
                if q3ddot_m[i] < q1ddot_M[i]:
                    ddq_lb[i] = q3ddot_m[i]
                else:
                    ddq_lb[i] = max(q1ddot_M[i], q2ddot_m[i])

        return ddq_lb, ddq_ub

    def acceleration_bounds_from_viability(self, q, qdot):
        a = self.dt**2
        b = self.dt * (2 * qdot + self.ddq_max * self.dt)
        c = qdot**2 - 2*self.ddq_max*(self.q_max - q - self.dt*qdot)
        q1ddot = -qdot/self.dt
        delta = b**2 - 4*(a*c)

        ddq_lb = np.ones_like(q) * -self.ddq_max
        ddq_ub = np.ones_like(q) * self.ddq_max
        for i, d in enumerate(delta):
            if d > 0:
                ddq_ub[i] = max(q1ddot[i], (-b[i] + np.sqrt(delta[i]))/(2*a))
            else:
                ddq_ub[i] = q1ddot[i]

        b = self.dt * (2 * qdot - self.ddq_max * self.dt)
        c = qdot ** 2 - 2 * self.ddq_max * (q + self.dt * qdot - self.q_min)
        delta = b ** 2 - 4 * (a * c)

        for i, d in enumerate(delta):
            if d > 0:
                ddq_lb[i] = min(q1ddot[i], (-b[i] - np.sqrt(delta[i]))/(2*a))
            else:
                ddq_lb[i] = q1ddot[i]

        return ddq_lb, ddq_ub

    def batched_acceleration_bounds_from_pos_limits(self, q: torch.Tensor, qdot: torch.Tensor):

        assert q.ndim > 1 and qdot.ndim > 1, "Missing batch dimension in first position (batch, nj)"

        q1ddot_M = -qdot/self.dt
        q2ddot_M = -qdot**2 / (2*(self.q_max - q))
        q3ddot_M = 2*(self.q_max - q - self.dt*qdot) / (self.dt**2)

        q2ddot_m = qdot**2 / (2*(q - self.q_min))
        q3ddot_m = 2*(self.q_min - q - self.dt*qdot) / (self.dt**2)

        ddq_lb = np.zeros_like(q) * -self.ddq_max
        ddq_ub = np.zeros_like(q) * self.ddq_max

        pos_vel = qdot >= 0
        neg_vel = np.logical_not(pos_vel)

        # Lower Bound:
        # if q_dot >= 0
        ddq_lb = ddq_lb + pos_vel * q3ddot_m #
        # if q_dot < 0
            # if q3ddot_m < q1ddot_M
        lb_acc_constraint = q3ddot_m < q1ddot_M
        ddq_lb = ddq_lb + neg_vel * (lb_acc_constraint * q3ddot_m)
            # else (q3ddot_m > q1ddot_M)
            #   ddq_lb = max(q1ddot_M, q2ddot_m)
        lb_lim = np.max(np.stack([q1ddot_M, q2ddot_m], dim=2), dim=2)
        ddq_lb = ddq_lb + neg_vel * (np.logical_not(lb_acc_constraint) * lb_lim)

        # Upper Bound:
        # if q_dot >= 0
            # if q3ddot_M > q1ddot_M:
        up_acc_constraint = q3ddot_M > q1ddot_M
        ddq_ub = ddq_ub + pos_vel * (up_acc_constraint * q3ddot_M) #
            # else (q3ddot_M < q1ddot_M):
        ub_lim = np.min(np.stack([q1ddot_M, q2ddot_M], dim=2), dim=2).values
        ddq_ub = ddq_ub + pos_vel * (np.logical_not(up_acc_constraint) * ub_lim) #
        # if q_dot < 0
        ddq_ub = ddq_ub + neg_vel * q3ddot_M

        return ddq_lb, ddq_ub

    def batched_acceleration_bounds_from_viability(self,  q: np.ndarray, qdot: np.ndarray):
        assert q.ndim > 1 and qdot.ndim > 1, "Missing batch dimension in first position (batch, nj)"

        a = self.dt**2
        b = self.dt * (2 * qdot + torch.tensor(self.ddq_max) * self.dt)
        c = qdot**2 - 2*torch.tensor(self.ddq_max)*(torch.tensor(self.q_max) - q - self.dt*qdot)
        q1ddot = -qdot/self.dt
        delta = b**2 - 4*(a*c)

        ddq_lb = torch.zeros_like(q) * -torch.tensor(self.ddq_max)
        ddq_ub = torch.zeros_like(q) * torch.tensor(self.ddq_max)

        # Upper bound:
        #     if delta > 0:
        d_pos = delta > 0
        max_ub = torch.max(torch.stack([q1ddot, (-b + np.sqrt(delta * d_pos))/(2*a)], dim=2), dim=2).values
        ddq_ub = ddq_ub + d_pos * max_ub
        #     else:
        ddq_ub = ddq_ub + torch.logical_not(d_pos) * q1ddot

        # Lower Bound
        b = self.dt * (2 * qdot - torch.tensor(self.ddq_max) * self.dt)
        c = qdot ** 2 - 2 * torch.tensor(self.ddq_max) * (q + self.dt * qdot - self.q_min)
        delta = b ** 2 - 4 * (a * c)

        #     if delta > 0:
        d_pos = delta > 0
        min_lb = torch.min(torch.stack([q1ddot, (-b - torch.sqrt(delta * d_pos))/(2*a)], dim=2), dim=2).values
        ddq_lb = ddq_lb + d_pos * min_lb
        #     else:
        ddq_lb = ddq_lb + torch.logical_not(d_pos) * q1ddot

        return ddq_lb, ddq_ub


class BatchedContraintAccelerationBound:

    def __init__(self, q_max: Collection, q_min: Collection, qdot_max: Collection, ddq_max: Collection, dt,
                 dt_scaler=2.0, device='cuda'):
        self.nq = len(q_max)
        self.q_max = torch.tensor(q_max, device=device)
        self.q_min = torch.tensor(q_min, device=device)
        self.qdot_max = torch.tensor(qdot_max, device=device)
        self.ddq_max = torch.tensor(ddq_max, device=device)
        self.dt = copy.copy(dt * dt_scaler)
        self.device = device

    def batched_acceletation_bounds(self, q: torch.Tensor, dq: torch.Tensor) -> Tuple[Sequence, Sequence]:
        # assert q.get_device() == self.device
        # assert dq.get_device() == self.device

        ddq_ub = torch.zeros(((4,) + q.shape), device=self.device)
        ddq_lb = torch.zeros(((4,) + q.shape), device=self.device)
        ddq_ub[3, ...] = self.ddq_max
        ddq_lb[3, ...] = -self.ddq_max
        ddq_lb[0, ...], ddq_ub[0, ...] = self.batched_acceleration_bounds_from_pos_limits(q, dq)
        ddq_lb[1, ...] = (-self.qdot_max - dq) / self.dt
        ddq_ub[1, ...] = (self.qdot_max - dq) / self.dt
        ddq_lb[2, ...], ddq_ub[2, ...] = self.batched_acceleration_bounds_from_viability(q, dq)

        ddq_lb_f = torch.max(ddq_lb, dim=0).values
        ddq_ub_f = torch.min(ddq_ub, dim=0).values
        # Ensure that in cases of extreme accelerations the feasible original bounds are not violated.
        ddq_lb_f = torch.clip(ddq_lb_f, a_min=-self.ddq_max, a_max=self.ddq_max)
        ddq_ub_f = np.clip(ddq_ub_f, a_min=-self.ddq_max, a_max=self.ddq_max)

        return torch.max(ddq_lb, dim=0).values, torch.min(ddq_ub, dim=0).values

    def batched_acceleration_bounds_from_pos_limits(self, q: torch.Tensor, qdot: torch.Tensor):

        assert q.ndim > 1 and qdot.ndim > 1, "Missing batch dimension in first position (batch, nj)"

        q1ddot_M = -qdot / self.dt
        q2ddot_M = -qdot ** 2 / (2 * (self.q_max - q))
        q3ddot_M = 2 * (self.q_max - q - self.dt * qdot) / (self.dt ** 2)

        q2ddot_m = qdot ** 2 / (2 * (q - self.q_min))
        q3ddot_m = 2 * (self.q_min - q - self.dt * qdot) / (self.dt ** 2)

        ddq_lb = torch.zeros_like(q) * -self.ddq_max
        ddq_ub = torch.zeros_like(q) * self.ddq_max

        pos_vel = qdot >= 0
        neg_vel = torch.logical_not(pos_vel)

        # Lower Bound:
        # if q_dot >= 0
        ddq_lb = ddq_lb + pos_vel * q3ddot_m  #
        # if q_dot < 0
        # if q3ddot_m < q1ddot_M
        lb_acc_constraint = q3ddot_m < q1ddot_M
        ddq_lb = ddq_lb + neg_vel * (lb_acc_constraint * q3ddot_m)
        # else (q3ddot_m > q1ddot_M)
        #   ddq_lb = max(q1ddot_M, q2ddot_m)
        lb_lim = torch.max(torch.stack([q1ddot_M, q2ddot_m], dim=2), dim=2).values
        # action = torch.logical_not(lb_acc_constraint) * lb_lim.values
        ddq_lb = ddq_lb + neg_vel * (torch.logical_not(lb_acc_constraint) * lb_lim)

        # Upper Bound:
        # if q_dot >= 0
        # if q3ddot_M > q1ddot_M:
        up_acc_constraint = q3ddot_M > q1ddot_M
        ddq_ub = ddq_ub + pos_vel * (up_acc_constraint * q3ddot_M)  #
        # else (q3ddot_M < q1ddot_M):
        ub_lim = torch.min(torch.stack([q1ddot_M, q2ddot_M], dim=2), dim=2).values
        ddq_ub = ddq_ub + pos_vel * (torch.logical_not(up_acc_constraint) * ub_lim)  #
        # if q_dot < 0
        ddq_ub = ddq_ub + neg_vel * q3ddot_M

        return ddq_lb, ddq_ub

    def batched_acceleration_bounds_from_viability(self, q: torch.Tensor, qdot: torch.Tensor):

        assert q.ndim > 1 and qdot.ndim > 1, "Missing batch dimension in first position (batch, nj)"

        a = self.dt ** 2
        b = self.dt * (2 * qdot + self.ddq_max * self.dt)
        c = qdot ** 2 - 2 * self.ddq_max * (self.q_max - q - self.dt * qdot)
        q1ddot = -qdot / self.dt
        delta = b ** 2 - 4 * (a * c)

        ddq_lb = torch.zeros_like(q) * -self.ddq_max
        ddq_ub = torch.zeros_like(q) * self.ddq_max

        # Upper bound:
        #     if delta > 0:
        d_pos = delta > 0
        max_ub = torch.max(torch.stack([q1ddot, (-b + np.sqrt(delta * d_pos)) / (2 * a)], dim=2), dim=2).values
        ddq_ub = ddq_ub + d_pos * max_ub
        #     else:
        ddq_ub = ddq_ub + torch.logical_not(d_pos) * q1ddot

        # Lower Bound
        b = self.dt * (2 * qdot - self.ddq_max * self.dt)
        c = qdot ** 2 - 2 * self.ddq_max * (q + self.dt * qdot - self.q_min)
        delta = b ** 2 - 4 * (a * c)

        #     if delta > 0:
        d_pos = delta > 0
        min_lb = torch.min(torch.stack([q1ddot, (-b - torch.sqrt(delta * d_pos)) / (2 * a)], dim=2), dim=2).values
        ddq_lb = ddq_lb + d_pos * min_lb
        #     else:
        ddq_lb = ddq_lb + torch.logical_not(d_pos) * q1ddot

        return ddq_lb, ddq_ub
