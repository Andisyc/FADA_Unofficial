"""Formal APPO assembly with sealed checkpoints and explicit restart semantics."""
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import uuid

import torch

from unilab.algos.torch.fada_official_appo.config import (
    validate_collection_contract, validate_training_contract,
)
from unilab.algos.torch.fada_official_appo.runner import OfficialFadaAPPORunner
from unilab.algos.torch.fada_official_appo.checkpoint import (
    ALGORITHM, COUNTERS, ORACLE_SCHEMA, read_oracle_checkpoint, save_oracle_checkpoint,
    training_contract, validate_same_run,
)
from unilab.base.nan_guard import NanGuardCfg
from unilab.base.registry import apply_cfg_overrides
from unilab.envs.locomotion.g1.official_oracle.config import (
    G1OfficialOracleFlatCfg, build_environment_contract,
)
from unilab.training.backend_adapter import BackendAdapter
from unilab.training.experiment import ExperimentTracker
from unilab.training.run import get_log_root
from unilab.training.seed import apply_configured_training_seed


class APPOTrainingArtifacts:
    """Save only whole successful updates; final is sealed only after clean shutdown."""
    def __init__(self, directory, runner, identity, interval):
        self.directory, self.runner, self.identity = Path(directory), runner, identity
        self.interval, self.last_checkpoint = interval, None

    def save(self, role):
        runner = self.runner
        counts = runner._sample_counts()
        completed = counts["learner_updates"]
        identity = {**self.identity, "completed_updates": completed, "role": role,
            "reward": {"statistic": "raw_episode_return_last100", "value": runner.latest_episode_return,
                       "window_count": runner.episode_window_count, "session_id": self.identity["session_id"]}}
        prefix = "appo_final_update" if role == "final" else "appo_update"
        path = save_oracle_checkpoint(self.directory / "checkpoints" / f"{prefix}_{completed}.pt",
            runner.learner, identity=identity, cursor={k: counts[k] for k in COUNTERS})
        self.last_checkpoint = str(path)
        runner.last_run_summary.update(last_checkpoint=str(path), persistence_available=True)
        return path

    def __call__(self, event):
        if self.interval and event["iteration"] % self.interval == 0:
            self.save("intermediate")


def run_training(cfg, rl_cfg, *, root_dir, env_cfg_override=None, seed_info=None):
    """Own validation before logs/processes; never call the generic checkpoint loader."""
    validate_collection_contract(rl_cfg, task_name=cfg.training.task_name,
        sim_backend=cfg.training.sim_backend, collector_device=cfg.training.collector_device)
    if cfg.training.play_only:
        from unilab.algos.torch.fada_official_appo.playback import play_checkpoint
        return play_checkpoint(cfg, rl_cfg, root_dir=root_dir)
    capacity = cfg.training.replay_queue_size
    validate_training_contract(rl_cfg, num_envs=cfg.algo.num_envs,
        max_iterations=cfg.algo.max_iterations, save_interval=cfg.algo.save_interval,
        replay_queue_size=capacity)
    root_dir = Path(root_dir)
    if seed_info is None:
        seed_info = apply_configured_training_seed(cfg, torch_runtime=True, cuda=True)
    if env_cfg_override is None:
        env_cfg_override = BackendAdapter(cfg, root_dir=root_dir, algo_name="appo").build_task_env_cfg_override()
    env_cfg = G1OfficialOracleFlatCfg()
    apply_cfg_overrides(env_cfg, env_cfg_override)
    env_cfg.validate()
    if env_cfg.evaluating:
        raise ValueError("FADA APPO training must not use the evaluation command clock")
    environment = build_environment_contract(env_cfg)
    static = training_contract(rl_cfg, num_envs=cfg.algo.num_envs, replay_queue_size=capacity)
    resume = rl_cfg.get("resume_path")
    if resume is not None:
        resume = Path(resume)
        resume = (resume if resume.is_absolute() else root_dir / resume).resolve()
    payload = read_oracle_checkpoint(resume) if resume is not None else None
    lineage = cfg.training.oracle_lineage_id
    if lineage is not None and (not isinstance(lineage, str) or not lineage.strip()):
        raise ValueError("FADA APPO oracle_lineage_id must be a nonempty string")
    if payload is not None:
        validate_same_run(payload["official_appo_oracle"], environment_contract=environment,
                          static_contract=static, lineage_id=lineage)
        lineage = payload["official_appo_oracle"]["oracle_lineage_id"]
    lineage = lineage or uuid.uuid4().hex
    session = uuid.uuid4().hex
    identity = dict(schema=ORACLE_SCHEMA, algorithm=ALGORITHM, oracle_lineage_id=lineage,
        session_id=session, parent_checkpoint=str(resume) if resume else None,
        environment_contract=environment, training_contract=static,
        actor_config=rl_cfg["actor"], critic_config=rl_cfg["critic"])
    device = cfg.training.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if torch.device(device).type not in {"cpu", "cuda"}:
        raise ValueError("FADA APPO learner requires CPU or CUDA")
    log_dir = cfg.training.log_dir
    if log_dir is None:
        log_dir = Path(get_log_root(root_dir, cfg)) / cfg.training.task_name / "fada_official_appo" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    log_dir = Path(log_dir)
    if not log_dir.is_absolute():
        log_dir = root_dir / log_dir
    nan_guard_cfg = None
    if cfg.training.nan_guard.enabled:
        nan_guard_cfg = NanGuardCfg(enabled=True, buffer_size=cfg.training.nan_guard.buffer_size,
            max_envs_to_dump=cfg.training.nan_guard.max_envs_to_dump,
            output_dir=cfg.training.nan_guard.output_dir)
    with ExitStack() as resources:
        runner = OfficialFadaAPPORunner(cfg.training.task_name, env_cfg_override, rl_cfg,
            device=device, collector_device="cpu", sim_backend=cfg.training.sim_backend,
            num_envs=cfg.algo.num_envs, seed=rl_cfg.get("seed"),
            replay_queue_size=capacity, nan_guard_cfg=nan_guard_cfg)
        resources.callback(runner.close)
        if payload is not None:
            runner.restore_checkpoint(payload, environment_contract=environment,
                                      static_contract=static, lineage_id=lineage)
            cursor = payload["cursor"]
            if (cursor["learner_updates"] >= cfg.algo.max_iterations
                    or (rl_cfg.get("max_unique_samples") is not None
                        and cursor["unique_used_samples"] >= rl_cfg["max_unique_samples"])):
                return {"status": "already_complete", **cursor, "last_checkpoint": str(resume),
                        "persistence_available": True}
        # A resume always owns a new directory; never append to an earlier run.
        log_dir.mkdir(parents=True, exist_ok=False)
        tracker = ExperimentTracker(root_dir=root_dir, log_dir=str(log_dir), algo_name="appo",
            task_name=cfg.training.task_name, sim_backend=cfg.training.sim_backend,
            training_cfg=cfg.training, full_cfg=cfg, device=device,
            collector_device="cpu", seed_info=seed_info)
        resources.callback(tracker.finish)
        tracker.start()
        artifacts = APPOTrainingArtifacts(log_dir, runner, identity, cfg.algo.save_interval)
        tracker.update_summary({"status": "running", "session_id": session, "oracle_lineage_id": lineage,
            "parent_checkpoint": str(resume) if resume else None,
            "resume_semantics": "learner_and_known_sample_prefix; fresh environment/queues/episode window; "
                                "post-checkpoint crash sampling unknown; no trajectory/RNG replay"})
        try:
            summary = runner.learn(max_iterations=cfg.algo.max_iterations, save_interval=cfg.algo.save_interval,
                log_dir=str(log_dir), logger_type=cfg.training.logger, on_update=artifacts)
            artifacts.save("final")
            summary.update(log_dir=str(log_dir), session_id=session, oracle_lineage_id=lineage,
                           parent_checkpoint=str(resume) if resume else None)
            if not cfg.training.no_play:
                from unilab.algos.torch.fada_official_appo.playback import play_checkpoint
                play_cfg = deepcopy(cfg)
                from omegaconf import OmegaConf
                OmegaConf.update(play_cfg, "training.play_checkpoint_path", summary["last_checkpoint"], force_add=True)
                summary["play_video_path"] = play_checkpoint(play_cfg, rl_cfg, root_dir=root_dir)
            return summary
        except BaseException:
            tracker.update_summary({"status": "failed", "last_checkpoint": artifacts.last_checkpoint})
            if runner.last_run_summary is not None:
                runner.last_run_summary["status"] = "failed"
            raise
        finally:
            tracker.update_summary(runner.last_run_summary)
