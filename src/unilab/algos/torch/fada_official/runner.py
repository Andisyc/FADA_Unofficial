"""Synchronous heterogeneous PPO scheduling on the existing IPC lifecycle."""
from copy import deepcopy
import math
import time
from types import SimpleNamespace

import torch

from unilab.algos.torch.fada_official_ppo import OfficialFadaPPO
from unilab.ipc.async_runner import AsyncRunner
from unilab.ipc.memory_budget import (
    estimate_fada_ppo_bytes, raise_if_shared_memory_over_budget, warn_if_over_budget,
)
from unilab.ipc.rollout_ring_buffer import FADA_PPO_PROTOCOL, RolloutRingBuffer
from unilab.ipc.weight_sync import SharedWeightSync
from unilab.training.rsl_rl import normalize_ppo_train_cfg
from unilab.training.seed import apply_training_seed, derive_worker_seed

from .rollout import import_round_into_learner, model_observations, validate_round_payload
from .worker import official_fada_collector_fn
from .metrics import EpisodeMetrics


class OfficialFadaPPORunner(AsyncRunner):
    def __init__(self, env_name, env_cfg_overrides, rl_cfg, *, device=None,
                 sim_backend="mujoco", env_factory=None):
        cfg = normalize_ppo_train_cfg(deepcopy(rl_cfg))
        collection = cfg["collection"]
        if cfg.get("multi_gpu") is not None:
            raise ValueError("Official PPO synchronous v1 uses one learner device, not multi-GPU")
        if (env_name != "G1OfficialOracleFlat" or sim_backend != "mujoco"
                or collection["mode"] != "synchronous_heterogeneous"
                or collection["collector_device"] != "cpu" or collection["num_slots"] != 1):
            raise ValueError("Official PPO collector requires its MuJoCo task, CPU and one synchronous slot")
        super().__init__(env_name, env_cfg_overrides, cfg, device=device,
                         collector_device="cpu", sim_backend=sim_backend, num_envs=int(cfg["num_envs"]))
        self.collection = collection
        self.num_steps = int(cfg["num_steps_per_env"])
        self.obs_dim = int(collection["obs_dim"])
        self.critic_dim = int(collection["critic_dim"])
        self.action_dim = int(collection["action_dim"])
        if min(self.num_envs, self.num_steps, self.obs_dim, self.critic_dim, self.action_dim) < 1:
            raise ValueError("Official PPO collection dimensions must be positive")
        self.round_timeout_s = float(collection["round_timeout_s"])
        self.poll_interval_s = float(collection["poll_interval_s"])
        if not (math.isfinite(self.round_timeout_s) and math.isfinite(self.poll_interval_s)
                and 0 < self.poll_interval_s <= self.round_timeout_s):
            raise ValueError("Official PPO collection requires finite positive wait budgets")
        if int(collection["collector_torch_threads"]) < 1:
            raise ValueError("collector_torch_threads must be positive")
        self.env_factory = env_factory
        self.learner = None
        self.iteration = 0
        self.collector_exitcode = None
        self._closed = False
        self.episode_metrics = EpisodeMetrics(self.num_envs)

    def _get_default_device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _build_learner(self):
        apply_training_seed(self.rl_cfg.get("seed"), torch_runtime=True, cuda=torch.device(self.device).type == "cuda")
        cfg = deepcopy(self.rl_cfg)
        cfg["multi_gpu"] = None
        obs = model_observations(torch.zeros(self.num_envs, self.obs_dim, device=self.device),
                                 torch.zeros(self.num_envs, self.critic_dim, device=self.device))
        learner = OfficialFadaPPO.construct_algorithm(obs, SimpleNamespace(
            num_envs=self.num_envs, num_actions=self.action_dim), cfg, self.device)
        if not isinstance(learner, OfficialFadaPPO):
            raise ValueError("Synchronous Official FADA runner requires OfficialFadaPPO")
        return learner

    _collector_fn = staticmethod(official_fada_collector_fn)

    def _assert_collector_running(self):
        process = self._collector_process
        if self._stop_event.is_set() or (process is not None and not process.is_alive()):
            raise RuntimeError(self._read_collector_error())
        if not self._check_collector_alive():
            raise RuntimeError("Official PPO collector exited; see collector diagnostic")

    def _wait_for_round(self, ring):
        deadline = time.monotonic() + self.round_timeout_s
        while True:
            self._assert_collector_running()
            if ring.available():
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("Official PPO learner timed out waiting for a complete rollout")
            self._stop_event.wait(self.poll_interval_s)

    def learn(self, max_iterations, *, on_iteration=None):
        """Drive bounded rounds; file checkpoints/logging belong to the entrypoint."""
        if self._closed or max_iterations < 1:
            raise ValueError("Official PPO runner is closed or iteration budget is invalid")
        try:
            budget = estimate_fada_ppo_bytes(num_envs=self.num_envs, steps_per_env=self.num_steps,
                obs_dim=self.obs_dim, action_dim=self.action_dim, critic_dim=self.critic_dim, num_slots=1)
            warn_if_over_budget(budget, label="Official FADA PPO IPC")
            raise_if_shared_memory_over_budget(budget, label="Official FADA PPO IPC")
            if self.learner is None:
                self.learner = self._build_learner()
            if self.learner.storage.step or self.learner.transition.actions is not None:
                raise RuntimeError("Official PPO runner starts only from an empty rollout")
            ring = RolloutRingBuffer(self.num_envs, self.num_steps, self.obs_dim, self.action_dim,
                critic_dim=self.critic_dim, num_slots=1, protocol=FADA_PPO_PROTOCOL)
            self._shared_resources.append(ring)
            pair = torch.nn.ModuleDict({"actor": self.learner.actor, "critic": self.learner.critic})
            pair_state = pair.state_dict()
            if any(p.dtype != torch.float32 for p in pair_state.values()):
                raise ValueError("Official PPO weight transport requires float32 state")
            shapes = {name: p.shape for name, p in pair_state.items()}
            weights = SharedWeightSync(shapes)
            self._shared_resources.append(weights)
            weights.write_weights(pair_state)
            expected_version = weights.version
            self._start_collector(self._collector_fn, {
                "stop_event": self._stop_event, "env_name": self.env_name,
                "env_cfg_overrides": self.env_cfg_overrides, "rl_cfg": self.rl_cfg,
                "num_envs": self.num_envs, "num_steps": self.num_steps, "obs_dim": self.obs_dim,
                "action_dim": self.action_dim, "critic_dim": self.critic_dim,
                "ring_names": ring.name, "sync_primitives": ring.sync_primitives,
                "weight_name": weights.name, "weight_shapes": shapes, "weight_lock": weights.lock,
                "initial_version": expected_version, "round_timeout_s": self.round_timeout_s,
                "poll_interval_s": self.poll_interval_s,
                "collector_torch_threads": int(self.collection["collector_torch_threads"]),
                "sim_backend": self.sim_backend, "env_factory": self.env_factory,
                "seed": derive_worker_seed(self.rl_cfg.get("seed")),
            })
            last_metrics = None
            for index in range(max_iterations):
                self._wait_for_round(ring)
                validate_round_payload(ring.read_numpy_views(), expected_version=expected_version,
                    num_envs=self.num_envs, num_steps=self.num_steps, obs_dim=self.obs_dim,
                    action_dim=self.action_dim, critic_dim=self.critic_dim)
                payload = ring.read_torch(self.device)  # Blocking owned copy before ack.
                ring.advance_read()
                self._assert_collector_running()
                raw_mean_reward = float(payload["rewards"].mean())
                episode_metrics = self.episode_metrics.consume(payload["rewards"], payload["dones"])
                import_round_into_learner(self.learner, payload)
                metrics = self.learner.update()
                self._assert_collector_running()
                self.iteration += 1
                last_metrics = {**metrics, "iteration": self.iteration, "policy_version": expected_version,
                                "samples": self.num_envs * self.num_steps, "raw_mean_reward": raw_mean_reward,
                                **episode_metrics}
                if on_iteration is not None:
                    on_iteration(last_metrics)
                if index + 1 < max_iterations:
                    self._assert_collector_running()
                    weights.write_weights(pair_state)
                    expected_version = weights.version
            return last_metrics
        finally:
            self.close()

    def close(self):
        if getattr(self, "_closed", False):
            return
        try:
            super().close()
        finally:
            process = self._collector_process
            if process is not None and process.pid is not None and not process.is_alive():
                process.join(timeout=0)
                self.collector_exitcode = process.exitcode
                process.close()
                self._collector_process = None
            self._closed = True
