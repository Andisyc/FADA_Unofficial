"""Explicit motor PD, evaluated with fresh joints at every 5ms substep."""
from dataclasses import dataclass

import numpy as np

from .config import DEFAULT_ANGLES, EFFORT, KD, KP


@dataclass
class ActuationState:
    default: np.ndarray
    action: np.ndarray
    previous: np.ndarray
    executed: np.ndarray
    queue: np.ndarray
    delay: np.ndarray
    kp_scale: np.ndarray
    kd_scale: np.ndarray
    rfi_scale: np.ndarray
    torque: np.ndarray
    target: np.ndarray

    @classmethod
    def create(cls, n, randomize=True):
        z = lambda: np.zeros((n, 29), np.float32)
        bias = np.random.uniform(-.05, .05, (n, 29)) if randomize else z()
        default = (DEFAULT_ANGLES + bias).astype(np.float32)
        return cls(default, z(), z(), z(), np.zeros((n, 2, 29), np.float32),
                   np.zeros(n, np.int64), np.ones((n, 29), np.float32),
                   np.ones((n, 29), np.float32), np.ones((n, 29), np.float32),
                   z(), default.copy())

    def reset(self, ids, randomize=True):
        for field in (self.action, self.previous, self.executed, self.queue, self.torque):
            field[ids] = 0
        self.target[ids] = self.default[ids]
        self.delay[ids] = np.random.randint(0, 2, len(ids)) if randomize else 0
        self.kp_scale[ids] = np.random.uniform(.8, 1.2, (len(ids), 29)) if randomize else 1
        self.kd_scale[ids] = np.random.uniform(.8, 1.2, (len(ids), 29)) if randomize else 1
        self.rfi_scale[ids] = 1

    def process(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != self.action.shape or not np.isfinite(actions).all():
            raise ValueError("Oracle actions must be finite (N,29)")
        self.previous[:] = self.action
        self.action[:] = actions
        self.queue[:, 1] = self.queue[:, 0]
        self.queue[:, 0] = np.clip(actions, -100, 100)
        self.executed[:] = self.queue[np.arange(len(actions)), self.delay]
        self.target[:] = self.default + .25 * self.executed
        return self.target

    def pd(self, q, dq, randomize=True):
        tau = self.kp_scale * KP * (self.target - q) - self.kd_scale * KD * dq
        if randomize:
            tau += np.random.uniform(-1, 1, tau.shape) * .1 * self.rfi_scale * EFFORT
        self.torque[:] = np.clip(tau, -EFFORT, EFFORT)
        return self.torque
