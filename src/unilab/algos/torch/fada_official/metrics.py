"""Raw episodic returns, matching official LoggingHelper's last 100 episodes."""
from collections import deque

import numpy as np

REWARD_STATISTIC = "episode_return_mean_last100"


class EpisodeMetrics:
    def __init__(self, num_envs):
        self.returns = np.zeros(num_envs, np.float64)
        self.lengths = np.zeros(num_envs, np.int64)
        self.recent_returns = deque(maxlen=100)
        self.recent_lengths = deque(maxlen=100)

    def consume(self, rewards, dones):
        rewards = rewards.detach().cpu().numpy()
        dones = dones.detach().cpu().numpy()
        if rewards.shape != dones.shape or rewards.shape[0] != self.returns.size:
            raise ValueError("Official episode metrics require matching (N,T) rewards/dones")
        completed = 0
        for t in range(rewards.shape[1]):
            self.returns += rewards[:, t]
            self.lengths += 1
            done = dones[:, t].astype(bool)
            self.recent_returns.extend(self.returns[done].tolist())
            self.recent_lengths.extend(self.lengths[done].tolist())
            completed += int(done.sum())
            self.returns[done] = 0
            self.lengths[done] = 0
        return {"Train/mean_reward": float(np.mean(self.recent_returns)) if self.recent_returns else None,
                "Train/mean_episode_length": float(np.mean(self.recent_lengths)) if self.recent_lengths else None,
                "episode_window_count": len(self.recent_returns), "episodes_completed": completed}
