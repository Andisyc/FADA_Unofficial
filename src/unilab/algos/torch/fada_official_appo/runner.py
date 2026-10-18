"""Fresh bounded APPO batches on UniLab's existing collector/learner lifecycle."""
from copy import deepcopy
import multiprocessing as mp
import time
from queue import Empty

import numpy as np
import torch

from unilab.algos.torch.appo.runner import APPORunner
from unilab.algos.torch.appo.worker import appo_collector_fn
from unilab.algos.torch.appo.staging import RolloutStagingPool
from unilab.ipc.memory_budget import (
    estimate_fada_appo_bytes, raise_if_shared_memory_over_budget, warn_if_over_budget,
)
from unilab.ipc.rollout_ring_buffer import FADA_APPO_PROTOCOL, RolloutRingBuffer
from unilab.ipc.weight_sync import SharedWeightSync
from unilab.training.seed import derive_worker_seed
from unilab.logging import OffPolicyLogger

from .config import validate_collection_contract, validate_training_contract
from .rollout import HEADERS, split_rollout


class OfficialFadaAPPORunner(APPORunner):
    def __init__(self, env_name, env_cfg_overrides, rl_cfg, *, device=None,
                 collector_device="cpu", sim_backend="mujoco", num_envs=None,
                 seed=None, env_factory=None, replay_queue_size=3, nan_guard_cfg=None,
                 resume_path=None):
        self._closed = True
        cfg = deepcopy(rl_cfg)
        validate_collection_contract(cfg, task_name=env_name, sim_backend=sim_backend,
                                     collector_device=collector_device)
        count = cfg["num_envs"] if num_envs is None else num_envs
        if type(count) is not int or count < 1:
            raise ValueError("FADA APPO num_envs must be a positive integer")
        if resume_path is not None:
            raise ValueError("Use the formal FADA APPO training owner for explicit checkpoint resume")
        super().__init__(env_name, env_cfg_overrides, cfg, device=device,
                         collector_device=collector_device, sim_backend=sim_backend,
                         num_envs=count, steps_per_env=cfg["steps_per_env"],
                         replay_queue_size=replay_queue_size, nan_guard_cfg=nan_guard_cfg,
                         seed=cfg.get("seed") if seed is None else seed)
        self.collection = cfg["collection"]
        self.env_factory = env_factory
        self.ring = self.weights = self.metrics_queue = None
        self._last_rollout_id = 0
        self.collector_exitcode = None
        self.generated_samples = mp.get_context("spawn").Value("q", 0)
        self.learner = None
        self.latest_episode_return = None
        self.episode_window_count = 0
        self.latest_collector_timing = {}
        self.counters = {name: 0 for name in ("accepted_samples", "unique_used_samples",
            "reused_samples", "stale_dropped_samples", "learner_updates", "optimizer_steps",
            "sample_exposures", "abandoned_samples")}
        self._sample_base = {"generated_samples": 0, "published_samples": 0}
        self.last_run_summary = None
        self._closed = False

    def _detect_dims(self):
        # Frozen task contract: never construct a simulator in the parent.
        self.critic_dim = self.critic_input_dim = 322
        return 322, 29

    def _sample_counts(self):
        # Read publication before generation: producer increments generation first.
        published = (int(self.ring.sync_primitives[0].value) * self.num_envs * self.steps_per_env
                     if self.ring is not None else 0)
        generated = int(self.generated_samples.value) + self._sample_base["generated_samples"]
        published += self._sample_base["published_samples"]
        return {**self.counters, "generated_samples": generated, "published_samples": published,
                "shutdown_unused_samples": generated - self.counters["unique_used_samples"]
                                           - self.counters["stale_dropped_samples"]
                                           - self.counters["abandoned_samples"]}

    def restore_checkpoint(self, payload, **expected):
        """Restore before IPC exists; queues, episode windows and versions are fresh."""
        from .checkpoint import restore_oracle_checkpoint
        if self._closed or self.ring is not None or self.last_run_summary is not None:
            raise RuntimeError("FADA APPO restore must precede collection")
        self.learner = self.learner or self._build_learner()
        cursor = restore_oracle_checkpoint(payload, self.learner, **expected)
        self.counters.update({k: cursor[k] for k in self.counters})
        self.counters["abandoned_samples"] = (cursor["generated_samples"]
            - cursor["unique_used_samples"] - cursor["stale_dropped_samples"])
        self._sample_base = {k: cursor[k] for k in self._sample_base}

    def _read_metrics(self, logger=None):
        while self.metrics_queue is not None:
            try:
                message = self.metrics_queue.get_nowait()
            except Empty:
                break
            if "error" in message:
                raise RuntimeError(message["error"])
            if "episode_window_count" in message:
                if message.get("reward_statistic") != "raw_episode_return_last100":
                    raise ValueError("FADA APPO reward statistic mismatch")
                self.episode_window_count = int(message["episode_window_count"])
                self.latest_episode_return = float(message["mean_ep_reward"])
            if "collector_timing_ms" in message:
                self.latest_collector_timing = message["collector_timing_ms"]
                if logger is not None:
                    logger.update_collector_timing(self.latest_collector_timing)
            if logger is not None and "mean_ep_length" in message:
                logger.update_ep_length(message["mean_ep_length"])

    def learn(self, max_iterations=None, save_interval=0, log_dir=None,
              logger_type="tensorboard", *, on_update=None):
        """Finite cumulative APPO budgets; owner callback persists completed updates."""
        if self._closed or self.ring is not None:
            raise RuntimeError("FADA APPO collection is closed or already started")
        limit = self.rl_cfg["max_iterations"] if max_iterations is None else max_iterations
        validate_training_contract(self.rl_cfg, num_envs=self.num_envs, max_iterations=limit,
            save_interval=save_interval, replay_queue_size=self.staging_pool_size)
        logger = hook = None
        started = time.perf_counter()
        totals = {name: 0. for name in ("wait_host_s", "h2d_host_s", "process_host_s",
                  "update_host_s", "publish_host_s", "logging_host_s")}
        self.last_run_summary = {"status": "failed", "stop_reason": "error",
            "last_checkpoint": None, "persistence_available": False,
            "timing_scope": "host wall clock; not CUDA event timing"}
        try:
            self.learner = self.learner or self._build_learner()
            self.learner.train_mode()
            self.start_collection(self.learner.actor, self.learner.critic)
            self.staging_bytes = sum(int(np.prod(shape)) * 4 for shape in self.payload_shapes.values()) * self.staging_pool_size
            if torch.device(self.device).type == "cpu":
                warn_if_over_budget({"total": self.staging_bytes, "breakdown": "CPU staging only"},
                                    "FADA APPO additional staging")
            pool = RolloutStagingPool(capacity=self.staging_pool_size, num_envs=self.num_envs,
                                     slot_shapes=self.payload_shapes, device=self.device)
            if log_dir is not None:
                logger = OffPolicyLogger(algo_name="FADA APPO", max_iterations=limit,
                    num_envs=self.num_envs, env_name=self.env_name, obs_dim=self.obs_dim,
                    action_dim=self.action_dim, log_dir=str(log_dir), log_backend=logger_type)
                logger.set_collection_sync(False)
                logger.log_status(f"IPC {self.ipc_bytes} bytes; additional staging {self.staging_bytes} bytes.")
                logger.start(status="Training")

            def count_optimizer_step(optimizer, args, kwargs):
                self.counters["optimizer_steps"] += 1
                self.counters["sample_exposures"] += self._optimizer_batch_size
            hook = self.learner.optimizer.register_step_post_hook(count_optimizer_step)
            rollout_samples = self.num_envs * self.steps_per_env
            sample_limit = self.rl_cfg.get("max_unique_samples")
            for iteration in range(self.counters["learner_updates"] + 1, limit + 1):
                pool.reset()
                metadata = []
                capacity = self.staging_pool_size
                if sample_limit is not None:
                    capacity = min(capacity, (sample_limit - self.counters["unique_used_samples"]) // rollout_samples)
                if capacity == 0:
                    self.last_run_summary["stop_reason"] = "max_unique_samples"
                    break
                deadline = time.monotonic() + self.collection["wait_timeout_s"]
                h2d_seconds = 0.
                wait_started = time.perf_counter()

                def accept(meta):
                    if self.weights.version - meta["policy_version"] > self.collection["max_policy_lag"]:
                        self.counters["stale_dropped_samples"] += rollout_samples
                        return False
                    return True

                def copy(payload):
                    nonlocal h2d_seconds
                    copying = time.perf_counter()
                    slot = pool.stage_numpy_views(payload)
                    h2d_seconds += time.perf_counter() - copying
                    return slot

                def take():
                    slot, meta = self.consume_rollout(copy, accept_metadata=accept, deadline=deadline)
                    if slot is not None:
                        metadata.append(meta)
                        self.counters["accepted_samples"] += rollout_samples
                    return slot

                while take() is None:
                    pass  # All stale drops share the same first-valid-data deadline.
                # Bound draining by this snapshot; never chase a continuously writing producer.
                for _ in range(min(capacity - 1, self.ring.available())):
                    take()
                wait_seconds = time.perf_counter() - wait_started - h2d_seconds
                self._assert_running()
                if any(not 0 <= self.weights.version - meta["policy_version"] <= self.collection["max_policy_lag"]
                       for meta in metadata):
                    raise ValueError("FADA APPO batch changed behavior-version validity")
                self._read_metrics(logger)
                batch = pool.batch()  # new dict: derived V-trace fields never survive reset
                count = len(metadata) * rollout_samples
                self._optimizer_batch_size = count // self.learner.num_mini_batches
                processing = time.perf_counter()
                self.learner.process_batch(batch)  # timeout correction was already applied once
                process_seconds = time.perf_counter() - processing
                updating = time.perf_counter()
                metrics = self.learner.update(batch)
                update_seconds = time.perf_counter() - updating
                self.counters["learner_updates"] += 1
                self.counters["unique_used_samples"] += count
                publishing = time.perf_counter()
                self.publish_weights(self.learner.actor, self.learner.critic)
                publish_seconds = time.perf_counter() - publishing
                times = dict(wait_host_s=wait_seconds, h2d_host_s=h2d_seconds,
                    process_host_s=process_seconds, update_host_s=update_seconds, publish_host_s=publish_seconds)
                for name, value in times.items():
                    totals[name] += value
                self._read_metrics(logger)
                metrics.update({f"samples/{key}": value for key, value in self._sample_counts().items()})
                metrics.update({f"timing/{key}": value for key, value in times.items()})
                metrics["Train/raw_return_available"] = int(self.episode_window_count > 0)
                metrics["episode_window_count"] = self.episode_window_count
                if self.episode_window_count:
                    metrics["Train/mean_reward"] = self.latest_episode_return
                metrics["policy_lag_max"] = max(self.weights.version - 1 - meta["policy_version"] for meta in metadata)
                logging = time.perf_counter()
                if logger is not None:
                    logger.log_collector(self.generated_samples.value, 0)
                    logger.update_staging_pool(pool.active_count, pool.capacity)
                    logger.log_step(iteration=iteration, metrics=metrics, reward=self.latest_episode_return,
                        train_time=process_seconds + update_seconds, wait_time=wait_seconds,
                        learner_incremental_h2d_time=h2d_seconds, weight_sync_time=publish_seconds,
                        extra_info={"throughput_steps": count})
                totals["logging_host_s"] += time.perf_counter() - logging
                if on_update is not None:
                    on_update({**metrics, "iteration": iteration, "rollouts": metadata})
                if sample_limit is not None and self.counters["unique_used_samples"] == sample_limit:
                    self.last_run_summary["stop_reason"] = "max_unique_samples"
                    break
            else:
                self.last_run_summary["stop_reason"] = "max_iterations"
            self.last_run_summary["status"] = "completed"
        finally:
            try:
                self.close()
            finally:
                if hook is not None:
                    hook.remove()
                if logger is not None:
                    logger.finish()
                self.last_run_summary.update(self._sample_counts())
                self.last_run_summary.update(timing_totals=totals, collector_timing_ms=self.latest_collector_timing,
                    training_wall_time_sec=time.perf_counter() - started,
                    raw_mean_episode_return=self.latest_episode_return,
                    episode_window_count=self.episode_window_count)
        return self.last_run_summary

    @staticmethod
    def _pair_state(actor, critic):
        state = torch.nn.ModuleDict({"actor": actor, "critic": critic}).state_dict()
        if not state or any(value.dtype != torch.float32 for value in state.values()):
            raise ValueError("FADA APPO joint weight transport requires float32 state")
        return state

    def start_collection(self, actor, critic):
        if self._closed or self.ring is not None:
            raise RuntimeError("FADA APPO collection is closed or already started")
        try:
            state = self._pair_state(actor, critic)
            self._weight_shapes = {name: value.shape for name, value in state.items()}
            budget = estimate_fada_appo_bytes(num_envs=self.num_envs,
                steps_per_env=self.steps_per_env, obs_dim=self.obs_dim, action_dim=self.action_dim,
                critic_dim=self.critic_dim, num_slots=self.collection["num_slots"],
                weight_param_shapes=self._weight_shapes)
            self.ipc_bytes = int(budget["total"])
            warn_if_over_budget(budget, "FADA APPO IPC")
            raise_if_shared_memory_over_budget(budget, "FADA APPO IPC")
            self.ring = RolloutRingBuffer(self.num_envs, self.steps_per_env, self.obs_dim,
                self.action_dim, critic_dim=self.critic_dim, num_slots=self.collection["num_slots"],
                protocol=FADA_APPO_PROTOCOL)
            self._shared_resources.append(self.ring)
            self.weights = SharedWeightSync(self._weight_shapes)
            self._shared_resources.append(self.weights)
            self.weights.write_weights(state)
            self.metrics_queue = mp.get_context("spawn").Queue(maxsize=100)
            self._start_collector(appo_collector_fn, {
                "stop_event": self._stop_event, "env_name": self.env_name, "rl_cfg": self.rl_cfg,
                "num_envs": self.num_envs, "steps_per_env": self.steps_per_env,
                "shm_rollout_ring_buffer_name": self.ring.name,
                "sync_primitives": self.ring.sync_primitives, "obs_dim": self.obs_dim,
                "action_dim": self.action_dim, "critic_dim": self.critic_dim,
                "actor_weight_sync_name": "", "actor_weight_param_shapes": {},
                "critic_weight_sync_name": "", "critic_weight_param_shapes": {},
                "metrics_queue": self.metrics_queue, "collector_device": self.collector_device,
                "sim_backend": self.sim_backend, "env_cfg_override": self.env_cfg_overrides,
                "seed": derive_worker_seed(self.seed, worker_index=0),
                "nan_guard_cfg": self.nan_guard_cfg, "generated_samples": self.generated_samples,
                "protocol": FADA_APPO_PROTOCOL, "num_slots": self.collection["num_slots"],
                "joint_weight_sync": {"name": self.weights.name, "shapes": self._weight_shapes,
                                      "lock": self.weights.lock}, "env_factory": self.env_factory,
            })
        except BaseException:
            self.close()
            raise

    def publish_weights(self, actor, critic):
        """Called by the learner thread, never concurrently with its optimizer step."""
        try:
            self._assert_running()
            state = self._pair_state(actor, critic)
            if {name: value.shape for name, value in state.items()} != self._weight_shapes:
                raise ValueError("FADA APPO published weight schema changed")
            self.weights.write_weights(state)
            return self.weights.version
        except BaseException:
            self.close()
            raise

    @property
    def payload_shapes(self):
        """Only float fields go to staging; int64 headers stay on the CPU."""
        return {name: shape for name, shape in self.ring.slot_shapes.items() if name not in HEADERS}

    def _assert_running(self):
        if self._closed or self._collector_process is None:
            raise RuntimeError("FADA APPO collection is not running")
        if self._stop_event.is_set() or not self._collector_process.is_alive():
            raise RuntimeError(self._read_collector_error())

    def consume_rollout(self, copy_into_owned, *, accept_metadata=None, deadline=None):
        """Copy callback must finish synchronously before returning (including H2D).

        Return (callback result, CPU metadata). Callback failure is fatal: the
        borrowed shared views must not outlive this call or escape into staging.
        """
        try:
            if deadline is None:
                deadline = time.monotonic() + self.collection["wait_timeout_s"]
            while True:
                self._assert_running()
                if time.monotonic() >= deadline:
                    raise TimeoutError("FADA APPO timed out waiting for a complete rollout")
                if self.ring.available():
                    break
                self._stop_event.wait(self.collection["poll_interval_s"])
            payload, metadata = split_rollout(self.ring.read_numpy_views(),
                previous_rollout_id=self._last_rollout_id, published_version=self.weights.version,
                num_steps=self.steps_per_env)
            result = (copy_into_owned(payload) if accept_metadata is None or accept_metadata(metadata)
                      else None)  # An explicit stale drop is acknowledged, never staged.
            self.ring.advance_read()
            self._last_rollout_id = metadata["rollout_id"]
            return result, metadata
        except BaseException:
            self.close()
            raise

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            super().close()
        finally:
            process = self._collector_process
            if process is not None and process.pid is not None and not process.is_alive():
                process.join()
                self.collector_exitcode = process.exitcode
                process.close()
                self._collector_process = None
            if self.metrics_queue is not None:
                self.metrics_queue.cancel_join_thread()
                self.metrics_queue.close()
                self.metrics_queue = None
