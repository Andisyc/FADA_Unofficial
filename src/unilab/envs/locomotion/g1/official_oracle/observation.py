"""Explicit teacher layout and independent nonprivileged student view."""
from dataclasses import dataclass

import numpy as np

from unilab.envs.common.rotation import np_quat_apply_inverse, np_yaw_from_quat

from .config import EFFORT


@dataclass(frozen=True)
class PhysicalFrame:
    q: np.ndarray
    dq: np.ndarray
    root_pos: np.ndarray
    root_quat: np.ndarray
    root_vel: np.ndarray
    root_ang: np.ndarray
    foot_pos: np.ndarray
    foot_quat: np.ndarray
    foot_vel: np.ndarray
    foot_point_pos: np.ndarray
    foot_point_vel: np.ndarray
    forces: np.ndarray
    history: np.ndarray
    student_gyro: np.ndarray
    student_up: np.ndarray

    @property
    def lin_body(self):
        return np_quat_apply_inverse(self.root_quat, self.root_vel)

    @property
    def ang_body(self):
        return np_quat_apply_inverse(self.root_quat, self.root_ang)

    @property
    def gravity(self):
        down = np.broadcast_to(np.array([0, 0, -1], np.float32), self.root_vel.shape)
        return np_quat_apply_inverse(self.root_quat, down)

    @property
    def lin_yaw(self):
        yaw = np_yaw_from_quat(self.root_quat)
        c, s = np.cos(yaw), np.sin(yaw)
        v = self.root_vel
        return np.stack((c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1]), axis=1)


def build_teacher(frame, act, command, dr_features, foot_slots, rfi_enabled=True):
    n = len(frame.q)
    base = np.concatenate((frame.ang_body * .25, frame.lin_body * 2,
                           frame.q - act.default, frame.dq * .05, act.action, frame.gravity), axis=1)
    actuator = np.concatenate((act.previous, act.executed, act.delay[:, None],
                               np.clip(act.torque / EFFORT, -5, 5), act.kp_scale,
                               act.kd_scale, act.rfi_scale,
                               np.broadcast_to([float(rfi_enabled), .1], (n, 2))), axis=1)
    force = frame.forces[:, foot_slots]
    contact = np.concatenate((np.clip(force / 100, -5, 5).reshape(n, 6),
                              (np.linalg.norm(force, axis=-1) > 1)), axis=1)
    # Analytic flat plane z=0: local profile = base_z - ground_z - clearance.
    terrain = np.concatenate((frame.root_pos[:, 2:3], frame.foot_point_pos[:, :, 2],
                              np.zeros((n, 9)), np.zeros((n, 1))), axis=1)
    gait = np.concatenate((command.velocity, np.sin(command.phase), np.cos(command.phase),
                           command.frequency[:, None], command.phase_dt[:, None],
                           command.standing[:, None]), axis=1)
    obs = np.concatenate((base, actuator, contact, terrain, gait, dr_features), axis=1).astype(np.float32)
    if obs.shape != (n, 322):
        raise ValueError(f"Official Oracle observation must be (N,322), got {obs.shape}")
    return obs


def build_student(frame, act, command):
    # Deliberately retain the old student field meanings, not slices of teacher322.
    return np.concatenate((frame.student_gyro * .25, -frame.student_up,
                           frame.q - act.default, frame.dq * .05, command.phase), axis=1).astype(np.float32)
