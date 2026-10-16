"""FADA unitree's 19 active terms; no legacy curriculum or heading controller."""
import numpy as np

from unilab.envs.common.rotation import np_quat_apply_inverse, np_yaw_from_quat

from .config import LOWER, POSE_WEIGHTS, UPPER


def reward_terms(frame, act, command, last_dq, foot_slots, undesired_slots, dt=.02):
    square_sum = lambda x: np.sum(np.square(x), axis=-1)
    foot_contact = np.max(np.linalg.norm(frame.history[:, :, foot_slots], axis=-1), axis=1) > 1
    all_contact = np.max(np.linalg.norm(frame.history, axis=-1), axis=1) > 1
    soft_lo = (LOWER + UPPER) / 2 - .95 * (UPPER - LOWER) / 2
    soft_hi = (LOWER + UPPER) / 2 + .95 * (UPPER - LOWER) / 2
    relative = frame.foot_vel - frame.root_vel[:, None, :]
    foot_body_vel = np_quat_apply_inverse(
        np.repeat(frame.root_quat, 2, axis=0), relative.reshape(-1, 3)
    ).reshape(-1, 2, 3)
    down = np.broadcast_to([0., 0., -1.], (len(frame.q) * 2, 3))
    foot_gravity = np_quat_apply_inverse(frame.foot_quat.reshape(-1, 4), down).reshape(-1, 2, 3)
    yaw = np_yaw_from_quat(frame.root_quat)
    difference = frame.foot_pos[:, 0] - frame.foot_pos[:, 1]
    lateral_gap = np.abs(np.cos(yaw) * difference[:, 1] - np.sin(yaw) * difference[:, 0])
    stance = np.remainder(command.phase / (2 * np.pi), 1) < .55
    clearance_error = (frame.foot_point_pos[:, :, 2] - .05) ** 2
    foot_speed = np.linalg.norm(frame.foot_point_vel[:, :, :2], axis=-1)
    return {
        "track_xy": np.exp(-square_sum(command.velocity[:, :2] - frame.lin_yaw) / .25),
        "track_yaw": np.exp(-(command.velocity[:, 2] - frame.ang_body[:, 2]) ** 2 / .25),
        "alive": np.ones(len(frame.q), np.float32),
        "lin_z": frame.lin_body[:, 2] ** 2,
        "ang_xy": square_sum(frame.ang_body[:, :2]),
        "joint_vel": square_sum(frame.dq),
        "joint_acc": square_sum((last_dq - frame.dq) / dt),
        "action_rate": square_sum(act.previous - act.action),
        "pos_limits": np.sum(np.maximum(soft_lo - frame.q, 0) + np.maximum(frame.q - soft_hi, 0), axis=1),
        "energy": np.sum(np.abs(frame.dq) * np.abs(act.torque), axis=1),
        "pose": np.sum((frame.q - act.default) ** 2 * POSE_WEIGHTS, axis=1),
        "orientation": square_sum(frame.gravity[:, :2]),
        "world_base_height": (frame.root_pos[:, 2] - .78) ** 2,
        "gait": np.sum(stance == foot_contact, axis=1) * (np.linalg.norm(command.velocity, axis=1) > .1),
        "slide": np.sum(np.linalg.norm(foot_body_vel[:, :, :2], axis=-1) * foot_contact, axis=1),
        "clearance": np.exp(-np.sum(clearance_error * np.tanh(2 * foot_speed), axis=1) / .05),
        "feet_ori": np.sum(np.linalg.norm(foot_gravity[:, :, :2], axis=-1), axis=1),
        "close_feet": (lateral_gap < .2).astype(np.float32),
        "undesired": np.sum(all_contact[:, undesired_slots], axis=1),
    }


def total_reward(terms, scales, dt=.02):
    if terms.keys() != scales.keys():
        raise ValueError("Reward scale/term names differ")
    return (sum(scales[name] * value for name, value in terms.items()) * dt).astype(np.float32)
