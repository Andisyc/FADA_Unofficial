"""Command and gait state; refresh never calls advance()."""
from dataclasses import dataclass

import numpy as np


@dataclass
class CommandState:
    velocity: np.ndarray
    frequency: np.ndarray
    phase_dt: np.ndarray
    offset: np.ndarray
    phase: np.ndarray
    age: np.ndarray

    @classmethod
    def create(cls, n):
        return cls(
            np.zeros((n, 3), np.float32), np.ones(n, np.float32),
            np.full(n, 2 * np.pi * .02, np.float32), np.zeros((n, 2), np.float32),
            np.zeros((n, 2), np.float32), np.zeros(n, np.int64),
        )

    @property
    def standing(self):
        return (np.linalg.norm(self.velocity[:, :2], axis=1) < .01) & (np.abs(self.velocity[:, 2]) < .01)

    def reset(self, ids, evaluating=False):
        n = len(ids)
        self.resample(ids, evaluating)
        left = np.zeros(n) if evaluating else np.random.uniform(-np.pi, np.pi, n)
        self.offset[ids, 0] = left
        self.offset[ids, 1] = np.fmod(left + 2 * np.pi, 2 * np.pi) - np.pi
        self.age[ids] = 0
        self.phase[ids] = self.offset[ids]

    def resample(self, ids, evaluating=False):
        n = len(ids)
        self.velocity[ids] = 0 if evaluating else np.random.uniform(-1, 1, (n, 3))
        if not evaluating:
            self.velocity[ids[np.random.random(n) < .2]] = 0
        self.frequency[ids] = 1 if evaluating else np.random.uniform(.8, 1.2, n)
        self.phase_dt[ids] = 2 * np.pi * .02 * self.frequency[ids]

    def update_phase(self, episode_steps, ids=None):
        if ids is None:
            ids = np.arange(len(self.age))
        phase = episode_steps[ids, None] * self.phase_dt[ids, None] + self.offset[ids]
        self.phase[ids] = (phase + np.pi) % (2 * np.pi) - np.pi
        self.phase[ids[self.standing[ids]]] = np.pi

    def advance(self, episode_steps, evaluating=False, *, external=False):
        self.age += 1
        self.update_phase(episode_steps)
        ids = np.flatnonzero(episode_steps % 500 == 0) if not (evaluating or external) else np.array([], dtype=int)
        if len(ids):
            self.resample(ids, evaluating)
