"""HORA, generic distillation, and FADA playback session factories."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

from unilab.visualization.playback_policy_sessions import (
    _ensure_scripts_dir,
    _resolve_task_checkpoint_from_playback_cfg,
    select_torch_device,
)
from unilab.visualization.playback_sessions import (
    _HORA_DISTILL_CHECKPOINT_UNAVAILABLE,
    FADAPlaybackSession,
    RslRlPlaybackConfig,
    RslRlPlaybackSession,
)

LogFn = Callable[[str], None]

def _default_hora_distill_playback_deps(root_dir: str | Path) -> dict[str, Any]:
    _ensure_scripts_dir(root_dir)
    from train_hora_distill import (
        _apply_teacher_defaults,
        _build_play_env_cfg_override,
        _cfg_with_checkpoint_runtime,
        _format_stage2_play_checkpoint_error,
        _resolve_stage2_checkpoint_path,
        _student_policy,
    )

    from unilab.algos.torch.hora.distill import (
        build_student_actor_and_normalizer,
        load_distilled_checkpoint,
    )
    from unilab.algos.torch.hora.rsl_rl import HoraRslRlVecEnvWrapper
    from unilab.training import create_env, get_log_root

    return {
        "apply_teacher_defaults": _apply_teacher_defaults,
        "build_play_env_cfg_override": _build_play_env_cfg_override,
        "build_student_actor_and_normalizer": build_student_actor_and_normalizer,
        "cfg_with_checkpoint_runtime": _cfg_with_checkpoint_runtime,
        "create_env": create_env,
        "format_stage2_play_checkpoint_error": _format_stage2_play_checkpoint_error,
        "get_log_root": get_log_root,
        "load_distilled_checkpoint": load_distilled_checkpoint,
        "resolve_stage2_checkpoint_path": _resolve_stage2_checkpoint_path,
        "student_policy": _student_policy,
        "wrapper_cls": HoraRslRlVecEnvWrapper,
        "checkpoint_reader": torch.load,
    }


def create_hora_distill_playback_session(
    *,
    playback_cfg: RslRlPlaybackConfig,
    cfg: Any,
    root_dir: str | Path,
    device: str | None,
    deps: Mapping[str, Any] | None = None,
    log: LogFn = print,
) -> tuple[RslRlPlaybackSession, str, str | None]:
    """Create an interactive playback session for HORA stage-2 student checkpoints."""

    resolved_deps = dict(_default_hora_distill_playback_deps(root_dir) if deps is None else deps)
    device_name = select_torch_device() if device is None else str(device)
    load_path, load_path_dir = resolved_deps["resolve_stage2_checkpoint_path"](cfg)
    checkpoint_path = str(load_path) if load_path is not None else None
    policy: Callable[[Any], Any] | None = None

    if playback_cfg.action_mode == "policy":
        if load_path is None or load_path_dir is None or not Path(load_path).exists():
            task_log_root = resolved_deps["get_log_root"](Path(root_dir), cfg) / str(
                cfg.training.task_name
            )
            log(
                resolved_deps["format_stage2_play_checkpoint_error"](
                    cfg,
                    task_log_root=task_log_root,
                    load_path=load_path,
                    load_path_dir=load_path_dir,
                )
            )
            log("WARNING: falling back to zero actions.")
            runtime_cfg = resolved_deps["apply_teacher_defaults"](cfg)
        else:
            log(f"Loading distilled checkpoint: {load_path}")
            checkpoint = resolved_deps["checkpoint_reader"](
                load_path, map_location="cpu", weights_only=False
            )
            if "model_state_dict" not in checkpoint:
                raise ValueError(
                    f"Checkpoint at {load_path} is not a HORA distillation checkpoint "
                    f"(found keys: {set(checkpoint.keys())})."
                )
            runtime_cfg = resolved_deps["cfg_with_checkpoint_runtime"](cfg, checkpoint)
    else:
        runtime_cfg = resolved_deps["apply_teacher_defaults"](cfg)

    env_cfg_override = resolved_deps["build_play_env_cfg_override"](runtime_cfg)
    create_env = resolved_deps["create_env"]
    try:
        env = create_env(
            runtime_cfg,
            num_envs=int(playback_cfg.num_envs),
            env_cfg_override=env_cfg_override,
            sim_backend="mujoco",
            task_name=str(runtime_cfg.training.task_name),
        )
    except TypeError:
        if deps is None:
            raise
        env = create_env(
            runtime_cfg,
            num_envs=int(playback_cfg.num_envs),
            env_cfg_override=env_cfg_override,
        )
    if env is None:
        raise RuntimeError("Playback env factory did not return an environment.")

    policy_obs_mode = "actor"
    wrapper_cls = resolved_deps["wrapper_cls"]
    wrapped_env = wrapper_cls(env, device=device_name, policy_obs_mode=policy_obs_mode)
    torch_device = torch.device(device_name)

    if playback_cfg.action_mode == "policy" and load_path is not None and Path(load_path).exists():
        actor, hist_normalizer = resolved_deps["build_student_actor_and_normalizer"](
            wrapped_env,
            runtime_cfg,
            device=torch_device,
        )
        resolved_deps["load_distilled_checkpoint"](
            actor,
            hist_normalizer,
            load_path,
            device=torch_device,
        )
        actor.eval()
        hist_normalizer.eval()
        student_policy = resolved_deps["student_policy"]

        def policy(obs: Any) -> Any:
            return student_policy(actor, hist_normalizer, obs, device=torch_device)

    log(f"Policy obs mode: {policy_obs_mode}")
    log(f"Action mode: {playback_cfg.action_mode}")
    session = RslRlPlaybackSession(
        env=env,
        wrapped_env=wrapped_env,
        device=device_name,
        action_mode=playback_cfg.action_mode,
        policy=policy,
        num_envs=playback_cfg.num_envs,
    )
    return session, policy_obs_mode, checkpoint_path


def _apply_distill_playback_reset_contract(
    env_cfg_override: Mapping[str, Any] | None, task_name: str
) -> dict[str, Any] | None:
    """Force standing-only reset sampling for G1 distill playback owners."""

    task_key = str(task_name).lower().split("/", 1)[0].replace("-", "_")
    task_key = task_key.replace("_", "")
    if task_key not in {"g1walkflat", "g1walkheight"}:
        return dict(env_cfg_override) if env_cfg_override is not None else None
    merged = dict(env_cfg_override or {})
    commands_override = dict(merged.get("commands") or {})
    commands_override["rel_standing_envs"] = 1.0
    if "rel_transition_envs" in commands_override:
        commands_override["rel_transition_envs"] = 0.0
    merged["commands"] = commands_override
    if "standing_reset_base_qvel_limit" in merged:
        merged["standing_reset_base_qvel_limit"] = 0.0
    return merged


def _apply_keyboard_playback_reset_contract(
    env_cfg_override: Mapping[str, Any] | None,
    task_name: str,
    *,
    enabled: bool,
) -> dict[str, Any]:
    """Return deterministic standing reset overrides for keyboard-driven G1 playback."""

    # B1: OFF 路径只复制原配置, 保持训练和非键盘回放的 reset 分布不变.
    merged = dict(env_cfg_override or {})
    if not enabled:
        return merged

    task_key = str(task_name).lower().split("/", 1)[0].replace("-", "_")
    task_key = task_key.replace("_", "")
    if task_key not in {"g1walkflat", "g1walkheight"}:
        return merged

    # B2: ON 路径在 env 创建前原子化派生静止 command 与零根部速度 reset 合同.
    commands_override = dict(merged.get("commands") or {})
    commands_override["rel_standing_envs"] = 1.0
    commands_override["rel_transition_envs"] = 0.0
    merged["commands"] = commands_override
    merged["reset_base_qvel_limit"] = 0.0
    merged["standing_reset_base_qvel_limit"] = 0.0
    return merged


def _default_distill_playback_deps(root_dir: str | Path) -> dict[str, Any]:
    _ensure_scripts_dir(root_dir)
    from unilab.algos.torch.distill import load_distillation_student_policy
    from unilab.algos.torch.hora.rsl_rl import HoraRslRlVecEnvWrapper
    from unilab.training import BackendAdapter, create_env, ensure_registries

    ensure_registries()

    return {
        "build_env_cfg_override": lambda cfg: BackendAdapter(
            cfg,
            root_dir=root_dir,
            algo_name="distill",
        ).build_play_env_cfg_override(),
        "create_env": create_env,
        "load_student_policy": load_distillation_student_policy,
        "resolve_checkpoint": _resolve_task_checkpoint_from_playback_cfg,
        "wrapper_cls": HoraRslRlVecEnvWrapper,
    }


def _default_fada_playback_deps(root_dir: str | Path) -> dict[str, Any]:
    _ensure_scripts_dir(root_dir)
    from unilab.algos.torch.distill import load_fada_deployable_policy_checkpoint
    from unilab.algos.torch.hora.rsl_rl import HoraRslRlVecEnvWrapper
    from unilab.training import BackendAdapter, create_env, ensure_registries

    ensure_registries()
    return {
        "build_env_cfg_override": lambda cfg: BackendAdapter(
            cfg,
            root_dir=root_dir,
            algo_name="distill",
        ).build_play_env_cfg_override(),
        "create_env": create_env,
        "load_fada_policy": load_fada_deployable_policy_checkpoint,
        "resolve_checkpoint": _resolve_task_checkpoint_from_playback_cfg,
        "wrapper_cls": HoraRslRlVecEnvWrapper,
    }


def create_distill_playback_session(
    *,
    playback_cfg: RslRlPlaybackConfig,
    cfg: Any,
    root_dir: str | Path,
    device: str | None,
    deps: Mapping[str, Any] | None = None,
    log: LogFn = print,
) -> tuple[RslRlPlaybackSession, str, str | None]:
    """Create a playback session for generic distillation student checkpoints."""

    resolved_deps = dict(_default_distill_playback_deps(root_dir) if deps is None else deps)
    device_name = select_torch_device() if device is None else str(device)
    checkpoint = resolved_deps["resolve_checkpoint"](playback_cfg, cfg, root_dir)
    checkpoint_path = str(checkpoint) if checkpoint is not None else None
    policy_obs_mode = playback_cfg.policy_obs_mode
    if policy_obs_mode == "auto":
        policy_obs_mode = "actor"

    create_env = resolved_deps["create_env"]
    task_name = str(getattr(cfg.training, "task_name", playback_cfg.task))
    build_env_cfg_override = resolved_deps.get("build_env_cfg_override")
    env_cfg_override = build_env_cfg_override(cfg) if build_env_cfg_override is not None else {}
    env_cfg_override = _apply_distill_playback_reset_contract(env_cfg_override, task_name)
    try:
        env = create_env(
            cfg,
            num_envs=int(playback_cfg.num_envs),
            env_cfg_override=env_cfg_override,
            sim_backend="mujoco",
            task_name=task_name,
        )
    except TypeError:
        if deps is None:
            raise
        try:
            env = create_env(
                cfg,
                num_envs=int(playback_cfg.num_envs),
                env_cfg_override=env_cfg_override,
            )
        except TypeError:
            env = create_env(cfg, num_envs=int(playback_cfg.num_envs))
    if env is None:
        raise RuntimeError("Playback env factory did not return an environment.")

    wrapper_cls = resolved_deps["wrapper_cls"]
    wrapped_env = wrapper_cls(env, device=device_name, policy_obs_mode=policy_obs_mode)
    policy: Callable[[Any], Any] | None = None

    if playback_cfg.action_mode == "policy":
        if checkpoint is None or not Path(checkpoint).exists():
            log(
                "WARNING: no generic distillation student checkpoint found - "
                "falling back to zero actions."
            )
        else:
            log(f"Loading distillation student checkpoint: {checkpoint}")
            loaded_student = resolved_deps.get("load_student_policy")
            if loaded_student is None:
                from unilab.algos.torch.distill import load_distillation_student_policy

                loaded_student = load_distillation_student_policy
            from .playback_distill_policy import load_distill_playback_policy

            policy = load_distill_playback_policy(
                checkpoint=checkpoint,
                cfg=cfg,
                env=env,
                device=device_name,
                load_student_policy=loaded_student,
                log=log,
            )

    log(f"Policy obs mode: {policy_obs_mode}")
    log(f"Action mode: {playback_cfg.action_mode}")
    session = RslRlPlaybackSession(
        env=env,
        wrapped_env=wrapped_env,
        device=device_name,
        action_mode=playback_cfg.action_mode,
        policy=policy,
        num_envs=playback_cfg.num_envs,
    )
    return session, policy_obs_mode, checkpoint_path


def create_fada_playback_session(
    *,
    playback_cfg: RslRlPlaybackConfig,
    cfg: Any,
    root_dir: str | Path,
    device: str | None,
    deps: Mapping[str, Any] | None = None,
    log: LogFn = print,
) -> tuple[FADAPlaybackSession, str, str | None]:
    """Create a stateful FADA Planner-IDM playback session."""

    # B1: policy 模式先严格恢复 checkpoint, 避免加载失败后遗留已创建的环境资源.
    resolved_deps = dict(_default_fada_playback_deps(root_dir) if deps is None else deps)
    device_name = select_torch_device() if device is None else str(device)
    checkpoint = resolved_deps["resolve_checkpoint"](playback_cfg, cfg, root_dir)
    checkpoint_path = str(checkpoint) if checkpoint is not None else None
    policy_obs_mode = playback_cfg.policy_obs_mode
    if policy_obs_mode == "auto":
        policy_obs_mode = "actor"
    controller = None
    architecture = None
    if playback_cfg.action_mode == "policy":
        if checkpoint is None or not Path(checkpoint).is_file():
            raise FileNotFoundError(
                "FADA policy playback requires training.play_checkpoint_path or a resolvable run."
            )
        from unilab.algos.torch.distill.fada_playback import FADAPlaybackController

        loaded = resolved_deps["load_fada_policy"](checkpoint, device=device_name)
        controller = FADAPlaybackController(loaded.policy, device=device_name)
        architecture = loaded.policy.config
        log(f"Loading FADA Planner-IDM checkpoint: {checkpoint}")
        completed_name = (
            "completed_steps" if "completed_steps" in loaded.checkpoint else "completed_iterations"
        )
        log(
            "FADA checkpoint diagnostics: "
            f"{completed_name}={int(loaded.checkpoint[completed_name])}, "
            f"obs_dim={architecture.obs_dim}, action_dim={architecture.action_dim}, "
            f"command_dim={architecture.command_dim}, history={architecture.history_length}, "
            f"horizon={architecture.prediction_horizon}"
        )

    # B2: 通过 distill task config 构造环境, 键盘模式先收敛为静止零速 reset, 再核对 policy IO.
    create_env = resolved_deps["create_env"]
    task_name = str(getattr(cfg.training, "task_name", playback_cfg.task))
    build_env_cfg_override = resolved_deps.get("build_env_cfg_override")
    env_cfg_override = build_env_cfg_override(cfg) if build_env_cfg_override is not None else {}
    env_cfg_override = _apply_keyboard_playback_reset_contract(
        env_cfg_override,
        task_name,
        enabled=playback_cfg.keyboard,
    )
    if playback_cfg.keyboard:
        log("Keyboard playback reset: standing command with zero base velocity.")
    env = create_env(
        cfg,
        num_envs=int(playback_cfg.num_envs),
        env_cfg_override=env_cfg_override,
        sim_backend="mujoco",
        task_name=task_name,
    )
    if env is None:
        raise RuntimeError("Playback env factory did not return an environment.")
    try:
        if architecture is not None and architecture.observation_contract == "g1_official_oracle_student66_v1":
            from unilab.algos.torch.distill.fada.oracle import validate_fada_oracle_environment_contract
            validate_fada_oracle_environment_contract(
                loaded.checkpoint["runtime_config"]["official_oracle_identity"], env, cfg
            )
        wrapped_env = resolved_deps["wrapper_cls"](
            env, device=device_name, policy_obs_mode=policy_obs_mode
        )
    except BaseException:
        env.close()
        raise
    if architecture is not None:
        from unilab.algos.torch.distill.fada_observation import (
            raw_observation_dim_for_fada_contract,
        )

        observed_obs_dim = int(wrapped_env.num_obs)
        observed_action_dim = int(env.action_space.shape[0])
        expected_raw_obs_dim = raw_observation_dim_for_fada_contract(
            architecture.observation_contract,
            policy_observation_dim=architecture.obs_dim,
        )
        if (observed_obs_dim, observed_action_dim) != (
            expected_raw_obs_dim,
            architecture.action_dim,
        ):
            wrapped_env.close()
            raise ValueError(
                "FADA checkpoint/playback IO mismatch: "
                f"checkpoint_raw=({expected_raw_obs_dim}, {architecture.action_dim}) "
                f"environment=({observed_obs_dim}, {observed_action_dim})"
            )

    # B3: session 连接 done/reset lifecycle 与 history owner, 对 viewer 暴露标准 playback contract.
    log(f"Policy obs mode: {policy_obs_mode}")
    log(f"Action mode: {playback_cfg.action_mode}")
    session = FADAPlaybackSession(
        env=env,
        wrapped_env=wrapped_env,
        device=device_name,
        action_mode=playback_cfg.action_mode,
        controller=controller,
        num_envs=playback_cfg.num_envs,
    )
    return session, policy_obs_mode, checkpoint_path
