"""Formal assembly for the synchronous CPU-collector/GPU-learner Oracle route."""
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import os
import tempfile
import uuid

import torch
from omegaconf import OmegaConf

from unilab.algos.torch.fada_official.checkpoint import (
    build_environment_contract, make_identity, read_oracle_checkpoint,
    restore_oracle_checkpoint, save_oracle_checkpoint,
)
from unilab.algos.torch.fada_official.metrics import REWARD_STATISTIC
from unilab.algos.torch.fada_official.runner import OfficialFadaPPORunner
from unilab.base.registry import apply_cfg_overrides
from unilab.envs.locomotion.g1.official_oracle.config import G1OfficialOracleFlatCfg
from unilab.training.backend_adapter import BackendAdapter
from unilab.training.experiment import ExperimentTracker
from unilab.training.run import get_log_root, parse_checkpoint_path
from unilab.training.seed import apply_configured_training_seed


def _atomic_curve(path, payload):
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


class OracleTrainingArtifacts:
    """Iteration callback: one completed-update cursor, checkpoint and reward owner."""
    def __init__(self, directory, learner, identity, *, target, save_interval,
                 completed=0, samples_seen=0, writer=None, tracker=None, curve=None):
        self.directory, self.learner = Path(directory), learner
        self.identity = deepcopy(identity)
        self.target, self.interval = int(target), int(save_interval)
        self.base, self.completed, self.samples = int(completed), int(completed), int(samples_seen)
        self.writer, self.tracker = writer, tracker
        self.last_checkpoint = None
        self.curve = deepcopy(curve) if curve is not None else {
            "schema": "fada_official_reward_curve_v1", "identity": identity,
            "statistic": REWARD_STATISTIC, "window": 100, "points": [], "checkpoints": []}

    def __call__(self, metrics):
        completed = self.base + int(metrics["iteration"])
        if completed != self.completed + 1 or completed > self.target:
            raise ValueError("Official training callback received a nonconsecutive iteration")
        self.completed = completed
        self.samples += int(metrics["samples"])
        value = metrics["Train/mean_reward"]
        self.curve["points"].append({"iteration": completed, "value": value,
            "episode_window_count": int(metrics["episode_window_count"])})
        scalars = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        scalars["Train/num_samples"] = self.samples
        for key, scalar in scalars.items():
            if self.writer is not None:
                self.writer.add_scalar(key, scalar, completed)
        if self.tracker is not None and self.tracker.run is not None:
            self.tracker.run.log(scalars, step=completed)
        if completed % self.interval == 0 or completed == self.target:
            identity = {**self.identity, "completed_iterations": completed,
                "role": "final" if completed == self.target else "intermediate",
                "reward": {"value": value, "statistic": REWARD_STATISTIC}}
            path = save_oracle_checkpoint(self.directory / f"model_{completed}.pt", self.learner,
                                         identity=identity, samples_seen=self.samples)
            self.last_checkpoint = path
            self.curve["identity"] = identity
            self.curve["checkpoints"].append({"path": str(path.resolve()), "identity": identity})
            _atomic_curve(self.directory / "reward_curve.pt", self.curve)
        if self.tracker is not None:
            self.tracker.update_summary({"completed_iterations": completed, "total_env_steps": self.samples,
                "final_mean_reward": value, "last_checkpoint": str(self.last_checkpoint) if self.last_checkpoint else None})
        print(f"[official PPO] iteration {completed}/{self.target}, samples={self.samples}, "
              f"episode_return={value}, episodes_in_window={metrics['episode_window_count']}")


def _target_iterations(cfg):
    target = int(cfg.algo.max_iterations)
    timesteps = cfg.training.num_timesteps
    if timesteps is not None:
        size = int(cfg.algo.num_envs) * int(cfg.algo.num_steps_per_env)
        if int(timesteps) != timesteps or int(timesteps) < size or int(timesteps) % size:
            raise ValueError("Official num_timesteps must be a positive whole N*T rollout budget")
        target = int(timesteps) // size
    if target < 1 or int(cfg.algo.save_interval) < 1:
        raise ValueError("Official iteration/save budgets must be positive")
    return target


def run_training(cfg, *, root_dir, runner_factory=OfficialFadaPPORunner, env_factory=None):
    from unilab.algos.torch.fada_official.runtime import resolve_runtime
    root_dir = Path(root_dir)
    rl_cfg = OmegaConf.to_container(cfg.algo, resolve=True)
    resolve_runtime(rl_cfg)
    if str(cfg.training.task_name) != "G1OfficialOracleFlat" or str(cfg.training.sim_backend) != "mujoco":
        raise ValueError("Official PPO runtime requires G1OfficialOracleFlat/mujoco")
    if rl_cfg.get("episode_statistics") != {"window": 100, "statistic": REWARD_STATISTIC}:
        raise ValueError("Official PPO requires last-100 raw episodic-return statistics")
    target = _target_iterations(cfg)
    overrides = BackendAdapter(cfg, root_dir=root_dir, algo_name="ppo").build_task_env_cfg_override()
    env_cfg = G1OfficialOracleFlatCfg()
    apply_cfg_overrides(env_cfg, overrides)
    env_cfg.validate()  # Cold config only; parent never creates a physics environment.
    if env_cfg.evaluating:
        raise ValueError("Official PPO training must not use the evaluation command clock")
    contract = build_environment_contract(env_cfg)
    resume = OmegaConf.select(cfg, "algo.resume_path")
    if resume:
        resume = Path(str(resume))
        resume = resume if resume.is_absolute() else root_dir / resume
    elif str(cfg.algo.load_run) != "-1" or bool(cfg.algo.resume):
        resume, _ = parse_checkpoint_path(cfg, root_dir=root_dir)
        if resume is None:
            raise FileNotFoundError("Official PPO resume checkpoint could not be resolved")
    restored = read_oracle_checkpoint(resume) if resume else None
    lineage = OmegaConf.select(cfg, "training.oracle_lineage_id")
    if restored is not None:
        saved_identity = restored["official_oracle"]
        if lineage is not None and lineage != saved_identity["oracle_lineage_id"]:
            raise ValueError("Official PPO resume lineage mismatch")
        lineage = saved_identity["oracle_lineage_id"]
    lineage = lineage or uuid.uuid4().hex
    identity = make_identity(environment_contract=contract, lineage_id=lineage,
        completed_iterations=1, role="intermediate", actor_config=rl_cfg["actor"],
        critic_config=rl_cfg["critic"], reward_statistic=REWARD_STATISTIC)
    configured_device = cfg.training.device
    device = str(configured_device) if configured_device else ("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.device(device).type not in {"cuda", "cpu"}:
        raise ValueError("Official heterogeneous training requires a CPU or CUDA learner")
    seed_info = apply_configured_training_seed(cfg, torch_runtime=True, cuda=torch.device(device).type == "cuda")
    with ExitStack() as resources:
        runner = runner_factory("G1OfficialOracleFlat", overrides, rl_cfg, device=device,
            sim_backend="mujoco", env_factory=env_factory)
        resources.callback(runner.close)
        runner.learner = runner._build_learner()
        cursor = restore_oracle_checkpoint(resume, runner.learner, expected_identity=identity) if resume else {
            "completed_iterations": 0, "samples_seen": 0}
        completed = cursor["completed_iterations"]
        if completed >= target:
            return {"status": "already_complete", **cursor, "last_checkpoint": str(resume)}
        curve = None
        if resume and (resume.parent / "reward_curve.pt").is_file():
            from unilab.algos.torch.fada_official.checkpoint_selection import read_reward_curve
            curve = read_reward_curve(resume.parent / "reward_curve.pt", identity)
            curve["points"] = [p for p in curve["points"] if p["iteration"] <= completed]
            curve["checkpoints"] = [p for p in curve["checkpoints"]
                                    if p["identity"]["completed_iterations"] <= completed]
        configured_dir = cfg.training.log_dir
        directory = Path(str(configured_dir)) if configured_dir else (
            get_log_root(root_dir, cfg) / str(cfg.training.task_name) /
            (datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f") + "_mujoco"))
        if not directory.is_absolute():
            directory = root_dir / directory
        directory.mkdir(parents=True, exist_ok=False)
        tracker = ExperimentTracker(root_dir=root_dir, log_dir=directory, algo_name="ppo",
            task_name=cfg.training.task_name, sim_backend="mujoco", training_cfg=cfg.training,
            full_cfg=cfg, device=device, collector_device="cpu", seed_info=seed_info)
        resources.callback(tracker.finish)
        tracker.start()
        writer = None
        if cfg.training.logger in {"tensorboard", "wandb"}:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(str(directory))
            resources.callback(writer.close)
        artifacts = OracleTrainingArtifacts(directory, runner.learner, identity, target=target,
            save_interval=int(cfg.algo.save_interval), completed=completed, samples_seen=cursor["samples_seen"],
            writer=writer, tracker=tracker, curve=curve)
        tracker.update_summary({"status": "running", "resume_semantics": "weights_and_optimizers_only",
                                "parent_checkpoint": str(resume) if resume else None})
        try:
            runner.learn(target - completed, on_iteration=artifacts)
        except BaseException:
            tracker.update_summary({"status": "failed"})
            raise
        summary = {"status": "completed", "completed_iterations": artifacts.completed,
            "samples_seen": artifacts.samples, "last_checkpoint": str(artifacts.last_checkpoint),
            "log_dir": str(directory)}
        tracker.update_summary(summary)
        return summary


def play_training_checkpoint(cfg, *, root_dir, device=None):
    """Noninteractive PPO play uses the same sealed actor as interactive play."""
    from unilab.training import create_env
    from unilab.training.rsl_rl import RslRlVecEnvWrapper
    from unilab.visualization.playback_policy_sessions import create_official_oracle_playback_session
    from unilab.visualization.playback_sessions import RslRlPlaybackConfig
    path, _ = parse_checkpoint_path(cfg, root_dir=root_dir)
    if path is None:
        raise FileNotFoundError("Official PPO playback checkpoint could not be resolved")
    adapter = BackendAdapter(cfg, root_dir=root_dir, algo_name="ppo")
    play_cfg = RslRlPlaybackConfig(task=str(cfg.training.task_name), load_run=str(cfg.algo.load_run),
        checkpoint=str(path), action_mode="policy", policy_obs_mode="actor",
        algo_log_name=str(cfg.algo.algo_log_name), log_root=None, num_envs=int(cfg.training.play_env_num))
    session, _, _ = create_official_oracle_playback_session(playback_cfg=play_cfg,
        env_factory=lambda n: create_env(cfg, num_envs=n, env_cfg_override=adapter.build_play_env_cfg_override()),
        checkpoint_resolver=lambda *a: str(path), wrapper_cls=RslRlVecEnvWrapper, device=device)
    try:
        return session.env.run_playback_mode(play_render_mode=cfg.training.play_render_mode,
            play_steps=cfg.training.play_steps, output_video=path.parent / "play_video.mp4",
            render_spacing=float(cfg.training.render_spacing),
            render_offset_mode=str(getattr(session.env.cfg, "render_offset_mode", "grid")),
            initialize=session.reset, step=lambda _: session.step_once())
    finally:
        session.env.close()
