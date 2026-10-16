"""Checkpoint loading and policy-family playback session factories."""

from __future__ import annotations

import copy
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

from unilab.visualization.playback_sessions import (
    OffPolicyPlaybackSession,
    RslRlPlaybackConfig,
    RslRlPlaybackSession,
)

LogFn = Callable[[str], None]
_PRIVILEGED_CHECKPOINT_SCHEMAS = {
    "privileged_full_action_teacher": "unilab_privileged_full_action_teacher_v1",
    "privileged_residual_teacher": "unilab_privileged_residual_teacher_v1",
}
_LEGACY_TAR_WEIGHTS_ONLY_ERROR = (
    "Cannot use ``weights_only=True`` with files saved in the legacy .tar format"
)

def _ensure_scripts_dir(root_dir: str | Path) -> None:
    scripts_dir = Path(root_dir) / "scripts"
    if scripts_dir.is_dir() and str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))


def _actor_input_dim_from_state_dict(state_dict: Mapping[str, Any]) -> int | None:
    for key in ("net.0.weight", "actor.net.0.weight", "mlp.0.weight", "actor.mlp.0.weight"):
        weight = state_dict.get(key)
        if isinstance(weight, torch.Tensor) and weight.ndim == 2:
            return int(weight.shape[1])
    for key, weight in state_dict.items():
        if key.endswith(".0.weight") and isinstance(weight, torch.Tensor) and weight.ndim == 2:
            return int(weight.shape[1])
    return None


def _offpolicy_checkpoint_actor_input_dim(checkpoint: Mapping[str, Any]) -> int | None:
    fada_metadata = checkpoint.get("fada_privileged_oracle")
    if isinstance(fada_metadata, Mapping):
        from unilab.algos.torch.distill.fada_privileged_oracle import (
            FADA_ORACLE_CHECKPOINT_SCHEMA_VERSION,
        )

        dimensions = fada_metadata.get("dimensions")
        if fada_metadata.get("schema_version") in {
            2,
            FADA_ORACLE_CHECKPOINT_SCHEMA_VERSION,
        } and isinstance(dimensions, Mapping):
            obs_dim = dimensions.get("obs")
            if isinstance(obs_dim, int) and not isinstance(obs_dim, bool) and obs_dim > 0:
                return obs_dim

    for metadata_key, expected_schema in _PRIVILEGED_CHECKPOINT_SCHEMAS.items():
        metadata = checkpoint.get(metadata_key)
        if not isinstance(metadata, Mapping) or metadata.get("schema") != expected_schema:
            continue
        obs_dim = metadata.get("obs_dim")
        if isinstance(obs_dim, int) and not isinstance(obs_dim, bool) and obs_dim > 0:
            return obs_dim

    actor_state = checkpoint.get("actor")
    if not isinstance(actor_state, Mapping):
        return None
    return _actor_input_dim_from_state_dict(actor_state)


def _load_playback_checkpoint(checkpoint_path: str, *, device_name: str, log: LogFn) -> Any:
    try:
        return torch.load(checkpoint_path, map_location=device_name, weights_only=True)
    except RuntimeError as exc:
        if _LEGACY_TAR_WEIGHTS_ONLY_ERROR not in str(exc):
            raise
        log(
            "WARNING: checkpoint uses legacy PyTorch .tar serialization; "
            "reloading with weights_only=False. Only use trusted local checkpoints."
        )
        try:
            return torch.load(checkpoint_path, map_location=device_name, weights_only=False)
        except Exception as legacy_exc:
            raise RuntimeError(
                "Failed to load checkpoint after PyTorch legacy .tar fallback: "
                f"{checkpoint_path}. The file may be corrupted, incomplete, or not a "
                "PyTorch checkpoint; re-copy or re-download the checkpoint before playback."
            ) from legacy_exc


def select_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def create_rsl_rl_playback_session(
    *,
    playback_cfg: RslRlPlaybackConfig,
    env_factory: Callable[[int], Any],
    algo_config: dict[str, Any],
    root_dir: str | Path,
    device: str | None,
    checkpoint_resolver: Callable[[str, str, str | None, str, str | None], str | None],
    checkpoint_input_dim_reader: Callable[[str], int | None],
    entrypoint_log_root: Callable[..., Path],
    wrapper_cls: Any,
    runner_cls: Any,
    policy_obs_dims_getter: Callable[[Any], tuple[int, int]],
    train_cfg_normalizer: Callable[[dict[str, Any]], dict[str, Any]],
    log: LogFn = print,
) -> tuple[RslRlPlaybackSession, str, str | None]:
    """Create a playback session and load the selected policy checkpoint."""

    if algo_config.get("algorithm", {}).get("class_name") == "unilab.algos.torch.fada_official_ppo:OfficialFadaPPO":
        return create_official_oracle_playback_session(
            playback_cfg=playback_cfg, env_factory=env_factory,
            checkpoint_resolver=checkpoint_resolver, wrapper_cls=wrapper_cls,
            device=device, log=log)
    device_name = select_torch_device() if device is None else str(device)
    env = env_factory(int(playback_cfg.num_envs))
    if env is None:
        raise RuntimeError("Playback env factory did not return an environment.")
    actor_obs_dim, flat_obs_dim = policy_obs_dims_getter(env.obs_groups_spec)

    policy_obs_mode = playback_cfg.policy_obs_mode
    checkpoint_path: str | None = None
    if playback_cfg.action_mode == "policy":
        checkpoint_path = checkpoint_resolver(
            playback_cfg.task,
            playback_cfg.load_run,
            playback_cfg.checkpoint,
            playback_cfg.algo_log_name,
            playback_cfg.log_root,
        )
        if policy_obs_mode == "auto" and checkpoint_path is not None:
            ckpt_dim = checkpoint_input_dim_reader(checkpoint_path)
            if ckpt_dim == actor_obs_dim:
                policy_obs_mode = "actor"
            elif ckpt_dim == flat_obs_dim:
                policy_obs_mode = "flat"
            elif ckpt_dim is not None:
                raise RuntimeError(
                    "Checkpoint actor input dim mismatch: "
                    f"ckpt={ckpt_dim}, actor_obs={actor_obs_dim}, flat_obs={flat_obs_dim}. "
                    "Please pass --policy_obs_mode actor|flat explicitly if needed."
                )
            else:
                policy_obs_mode = "flat"

    wrapped_env = wrapper_cls(env, device=device_name, policy_obs_mode=policy_obs_mode)
    log(f"Policy obs mode: {policy_obs_mode} (actor_obs={actor_obs_dim}, flat_obs={flat_obs_dim})")

    train_cfg = train_cfg_normalizer(copy.deepcopy(algo_config))
    if "runner" not in train_cfg:
        train_cfg["runner"] = {}
    train_cfg["runner"]["logger"] = "none"

    policy = None
    if playback_cfg.action_mode == "policy":
        if checkpoint_path is None:
            log("WARNING: no checkpoint found - falling back to zero actions.")
        else:
            log_dir = str(
                entrypoint_log_root(
                    Path(root_dir),
                    algo_log_name=playback_cfg.algo_log_name,
                    log_root=playback_cfg.log_root,
                )
                / playback_cfg.task
                / "play_temp"
            )
            runner = runner_cls(wrapped_env, train_cfg, log_dir=log_dir, device=device_name)
            runner.load(
                checkpoint_path,
                load_cfg={
                    "actor": True,
                    "critic": False,
                    "optimizer": False,
                    "iteration": False,
                    "rnd": False,
                },
            )
            policy = runner.get_inference_policy(device=device_name)

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


def create_official_oracle_playback_session(
    *, playback_cfg, env_factory, checkpoint_resolver, wrapper_cls, device=None, log=print,
):
    """Load the same frozen raw-action actor used by the distillation teacher."""
    from unilab.algos.torch.fada_official.checkpoint import load_oracle_policy, validate_environment

    device_name = select_torch_device() if device is None else str(device)
    checkpoint = None
    policy = None
    if playback_cfg.action_mode == "policy":
        checkpoint = _normalize_checkpoint_value(playback_cfg.checkpoint_path)
        if checkpoint is None:
            checkpoint = checkpoint_resolver(playback_cfg.task, playback_cfg.load_run,
                playback_cfg.checkpoint, playback_cfg.algo_log_name, playback_cfg.log_root)
        if checkpoint is None:
            raise FileNotFoundError("Official Oracle playback requires a sealed checkpoint")
        policy = load_oracle_policy(checkpoint, device=device_name)
    env = env_factory(int(playback_cfg.num_envs))
    if env is None:
        raise RuntimeError("Playback env factory did not return an environment")
    try:
        if policy is not None:
            validate_environment(policy.checkpoint_identity, env)
        if playback_cfg.policy_obs_mode not in {"auto", "actor"}:
            raise ValueError("Official Oracle playback requires actor observations")
        wrapped = wrapper_cls(env, device=device_name, policy_obs_mode="actor")
        session = RslRlPlaybackSession(env=env, wrapped_env=wrapped, device=device_name,
            action_mode=playback_cfg.action_mode, policy=policy, num_envs=playback_cfg.num_envs)
    except BaseException:
        env.close()
        raise
    log(f"Loading Official PPO Oracle checkpoint: {checkpoint}")
    return session, "actor", checkpoint


def _normalize_checkpoint_value(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return None if text in {"", "-1", "None", "null"} else text


def _cfg_checkpoint_value(cfg: Any) -> str | None:
    from omegaconf import OmegaConf

    return _normalize_checkpoint_value(OmegaConf.select(cfg, "algo.checkpoint", default=None))


def _resolve_appo_checkpoint_from_cfg(
    cfg: Any,
    *,
    root_dir: str | Path,
) -> tuple[str | None, str | None]:
    _ensure_scripts_dir(root_dir)
    from unilab.training import get_log_root, resolve_task_checkpoint_path

    selected_checkpoint = _cfg_checkpoint_value(cfg)
    if selected_checkpoint is not None:
        checkpoint_path, checkpoint_dir = resolve_task_checkpoint_path(
            root_dir,
            task_name=str(cfg.training.task_name),
            load_run=str(cfg.algo.load_run),
            algo_log_name=str(cfg.algo.algo_log_name),
            checkpoint=selected_checkpoint,
            log_root=getattr(cfg.training, "log_root", None),
        )
        return (
            str(checkpoint_path) if checkpoint_path is not None else None,
            str(checkpoint_dir) if checkpoint_dir is not None else None,
        )

    from train_appo import resolve_appo_checkpoint_path

    base_log_dir = get_log_root(root_dir, cfg) / str(cfg.training.task_name)
    checkpoint_path, checkpoint_dir = resolve_appo_checkpoint_path(base_log_dir, cfg.algo.load_run)
    return (
        str(checkpoint_path) if checkpoint_path is not None else None,
        str(checkpoint_dir) if checkpoint_dir is not None else None,
    )


def _build_appo_actor(
    *,
    env: Any,
    wrapped_env: Any,
    cfg: Any,
    rl_cfg: dict[str, Any],
    device: str,
    is_hora: bool,
) -> Any:
    from copy import deepcopy

    from rsl_rl.utils import resolve_callable
    from tensordict import TensorDict

    from unilab.base.observations import get_obs_dims

    action_shape = env.action_space.shape
    if action_shape is None:
        raise ValueError("env.action_space.shape must be defined")
    action_dim = int(action_shape[0])
    rl_cfg_dict = deepcopy(rl_cfg)

    if is_hora:
        from unilab.algos.torch.hora.appo import _update_hora_obs_groups
        from unilab.algos.torch.hora.models import build_hora_shared_actor_critic
        from unilab.algos.torch.hora.rsl_rl_compat import (
            convert_config_v3_to_v4,
            is_rsl_rl_v4,
            is_rsl_rl_v5,
        )

        obs_td = wrapped_env.get_observations()
        num_envs = int(getattr(wrapped_env, "num_envs", getattr(env, "num_envs", 1)))
        obs_dim = int(obs_td["actor"].shape[-1])
        priv_info_dim = int(obs_td["priv_info"].shape[-1])
        if priv_info_dim <= 0:
            raise ValueError("HORA APPO interactive play requires privileged info.")
        _update_hora_obs_groups(rl_cfg_dict, obs_dim=obs_dim, priv_info_dim=priv_info_dim)
        if is_rsl_rl_v5():
            pass
        elif is_rsl_rl_v4():
            rl_cfg_dict = convert_config_v3_to_v4(rl_cfg_dict)

        actor_cfg = deepcopy(rl_cfg_dict["actor"])
        actor_cls = resolve_callable(actor_cfg.pop("class_name"))
        actor_cfg.pop("num_actions", None)
        critic_cfg = deepcopy(rl_cfg_dict.get("critic") or rl_cfg_dict.get("actor") or {})
        critic_cfg.pop("class_name", None)
        critic_cfg.pop("num_actions", None)
        critic_cfg.pop("distribution_cfg", None)
        shared_model = build_hora_shared_actor_critic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            priv_info_dim=priv_info_dim,
            actor_cfg=actor_cfg,
            critic_cfg=critic_cfg,
        ).to(device)
        td_example = TensorDict(
            {
                "actor": torch.zeros((num_envs, obs_dim), device=device),
                "priv_info": torch.zeros(
                    (num_envs, priv_info_dim),
                    device=device,
                ),
            },
            batch_size=num_envs,
        )
        actor = actor_cls(
            td_example,
            rl_cfg_dict["obs_groups"],
            "actor",
            action_dim,
            shared_model=shared_model,
            **actor_cfg,
        )
        return actor.to(device).eval()

    obs_dim, critic_dim = get_obs_dims(env.obs_groups_spec)
    num_envs = int(getattr(wrapped_env, "num_envs", getattr(env, "num_envs", 1)))
    obs_groups = rl_cfg_dict.setdefault("obs_groups", {})
    if "obs_groups" not in rl_cfg_dict or not isinstance(obs_groups, dict):
        obs_groups = {}
        rl_cfg_dict["obs_groups"] = obs_groups
    actor_group = obs_groups.get("actor", obs_groups.get("policy", {}))
    if isinstance(actor_group, dict) and "policy" in actor_group:
        actor_group["policy"] = obs_dim
        obs_groups["actor"] = actor_group
    else:
        obs_groups["actor"] = {"policy": obs_dim}
    critic_group = obs_groups.get("critic")
    if critic_group is None:
        obs_groups["critic"] = {"policy": critic_dim if critic_dim > 0 else obs_dim}
    elif isinstance(critic_group, dict) and "policy" in critic_group:
        critic_group["policy"] = critic_dim if critic_dim > 0 else obs_dim

    obs_example = torch.zeros((num_envs, obs_dim), device=device)
    td_example = TensorDict({"policy": obs_example}, batch_size=num_envs)
    actor_cfg = deepcopy(rl_cfg_dict["actor"])
    actor_cls = resolve_callable(actor_cfg.pop("class_name"))
    actor_cfg.pop("num_actions", None)
    actor = actor_cls(td_example, rl_cfg_dict["obs_groups"], "actor", action_dim, **actor_cfg)
    return actor.to(device).eval()


def create_appo_playback_session(
    *,
    playback_cfg: RslRlPlaybackConfig,
    cfg: Any,
    rl_cfg: dict[str, Any],
    env_factory: Callable[[int], Any],
    root_dir: str | Path,
    device: str | None,
    wrapper_cls: Any,
    log: LogFn = print,
) -> tuple[RslRlPlaybackSession, str, str | None]:
    """Create an APPO interactive playback session."""

    device_name = select_torch_device() if device is None else str(device)
    env = env_factory(int(playback_cfg.num_envs))
    if env is None:
        raise RuntimeError("Playback env factory did not return an environment.")

    from unilab.algos.torch.hora.runtime import is_hora_appo_runtime

    is_hora = is_hora_appo_runtime(rl_cfg)
    selected_wrapper_cls = wrapper_cls
    policy_obs_mode = playback_cfg.policy_obs_mode
    if is_hora:
        from unilab.algos.torch.hora.rsl_rl import HoraRslRlVecEnvWrapper

        selected_wrapper_cls = HoraRslRlVecEnvWrapper
        policy_obs_mode = "actor"

    wrapped_env = selected_wrapper_cls(env, device=device_name, policy_obs_mode=policy_obs_mode)
    policy = None
    checkpoint_path: str | None = None
    if playback_cfg.action_mode == "policy":
        checkpoint_path, _checkpoint_dir = _resolve_appo_checkpoint_from_cfg(cfg, root_dir=root_dir)
        if checkpoint_path is None or not Path(checkpoint_path).exists():
            log(
                "WARNING: no APPO checkpoint found for "
                f"load_run={cfg.algo.load_run} - falling back to zero actions."
            )
        else:
            actor = _build_appo_actor(
                env=env,
                wrapped_env=wrapped_env,
                cfg=cfg,
                rl_cfg=rl_cfg,
                device=device_name,
                is_hora=is_hora,
            )
            checkpoint = _load_playback_checkpoint(
                checkpoint_path,
                device_name=device_name,
                log=log,
            )
            actor.load_state_dict(checkpoint["actor"])
            policy = actor
            log(f"Loading APPO checkpoint: {checkpoint_path}")

    log(f"Action mode: {playback_cfg.action_mode}")
    return (
        RslRlPlaybackSession(
            env=env,
            wrapped_env=wrapped_env,
            device=device_name,
            action_mode=playback_cfg.action_mode,
            policy=policy,
            num_envs=playback_cfg.num_envs,
        ),
        policy_obs_mode,
        checkpoint_path,
    )


def create_sac_playback_session(
    *,
    playback_cfg: RslRlPlaybackConfig,
    cfg: Any,
    env_factory: Callable[[int], Any],
    root_dir: str | Path,
    device: str | None,
    algo_name: str = "sac",
    log: LogFn = print,
) -> tuple[OffPolicyPlaybackSession, str, str | None]:
    """Create an interactive playback session for off-policy actors."""


    _ensure_scripts_dir(root_dir)

    from train_offpolicy import (
        default_device,
        extract_play_obs,
        resolve_checkpoint_path,
        resolve_play_actor_spec,
        resolve_play_obs_dims,
    )

    from unilab.algos.torch.common.actor_factory import build_actor
    from unilab.algos.torch.offpolicy.worker import resolve_offpolicy_actor_priv_info

    device_name = default_device(torch, str(device) if device is not None else None)
    env = env_factory(int(playback_cfg.num_envs))
    if env is None:
        raise RuntimeError("Playback env factory did not return an environment.")

    obs_dim, critic_obs_dim = resolve_play_obs_dims(env.obs_groups_spec)
    action_shape = env.action_space.shape
    if action_shape is None:
        raise ValueError("env.action_space.shape must be defined")
    action_dim = int(action_shape[0])
    actor_algo_type, actor_kwargs = resolve_play_actor_spec(
        algo_name,
        cfg,
        obs_dim=obs_dim,
        critic_obs_dim=critic_obs_dim,
    )
    if algo_name == "flashsac":
        actor_kwargs.update(
            {
                "actor_num_blocks": cfg.algo.algo_params.actor_num_blocks,
                "actor_noise_zeta_mu": cfg.algo.algo_params.actor_noise_zeta_mu,
                "actor_noise_zeta_max": cfg.algo.algo_params.actor_noise_zeta_max,
            }
        )

    actor = None
    checkpoint_path: str | None = None
    normalizer = None
    if bool(getattr(cfg.algo, "obs_normalization", False)):
        from unilab.algos.torch.common.normalization import EmpiricalNormalization

        normalizer = EmpiricalNormalization(shape=obs_dim, device=device_name)
    if playback_cfg.action_mode == "policy":
        actor = build_actor(
            actor_algo_type,
            obs_dim,
            action_dim,
            cfg.algo.actor_hidden_dim,
            cfg.algo.use_layer_norm,
            device_name,
            **actor_kwargs,
        )
        actor.eval()
        if playback_cfg.checkpoint_path not in (None, ""):
            checkpoint = _resolve_task_checkpoint_from_playback_cfg(
                playback_cfg,
                cfg,
                root_dir,
            )
            checkpoint_path = str(checkpoint) if checkpoint is not None else None
        else:
            checkpoint_path, _checkpoint_dir = resolve_checkpoint_path(
                Path(root_dir),
                cfg.algo.algo_log_name,
                cfg.training.task_name,
                cfg.algo.load_run,
            )
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            log(
                f"WARNING: no {algo_name} checkpoint found for "
                f"load_run={cfg.algo.load_run} - falling back to zero actions."
            )
            actor = None
        else:
            checkpoint = _load_playback_checkpoint(
                checkpoint_path,
                device_name=device_name,
                log=log,
            )
            checkpoint_actor = checkpoint["actor"]
            checkpoint_obs_dim = _offpolicy_checkpoint_actor_input_dim(checkpoint)
            if checkpoint_obs_dim is not None and checkpoint_obs_dim != obs_dim:
                raise RuntimeError(
                    "Off-policy checkpoint actor input dim mismatch: "
                    f"checkpoint={checkpoint_obs_dim}, playback_env_obs={obs_dim}. "
                    "The playback env contract does not match the selected run_config. "
                    "For G1 mode-conditioned policies, ensure env.mode_observation is restored "
                    "from the checkpoint run_config or pass the matching Hydra overrides."
                )
            actor.load_state_dict(checkpoint_actor)
            if normalizer is not None and checkpoint.get("obs_normalizer"):
                normalizer.load_state_dict(checkpoint["obs_normalizer"])
                normalizer.eval()
            log(f"Loading {algo_name} checkpoint: {checkpoint_path}")

    log(f"Action mode: {playback_cfg.action_mode}")
    return (
        OffPolicyPlaybackSession(
            env=env,
            device=device_name,
            action_mode=playback_cfg.action_mode,
            actor=actor,
            actor_algo_type=actor_algo_type,
            normalizer=normalizer,
            num_envs=playback_cfg.num_envs,
            obs_extractor=extract_play_obs,
            priv_info_resolver=resolve_offpolicy_actor_priv_info,
        ),
        "actor",
        checkpoint_path,
    )


def _resolve_task_checkpoint_from_playback_cfg(
    playback_cfg: RslRlPlaybackConfig,
    cfg: Any,
    root_dir: str | Path,
) -> Path | None:
    from unilab.training.run import resolve_task_checkpoint_path

    if playback_cfg.checkpoint_path not in (None, ""):
        path = Path(str(playback_cfg.checkpoint_path))
        resolved_path = path if path.is_absolute() else Path(root_dir) / path
        if not resolved_path.is_file():
            raise FileNotFoundError(
                f"training.play_checkpoint_path does not exist: {resolved_path}"
            )
        return resolved_path

    checkpoint_path, _run_dir = resolve_task_checkpoint_path(
        root_dir,
        task_name=str(getattr(cfg.training, "task_name", playback_cfg.task)),
        load_run=playback_cfg.load_run,
        algo_log_name=playback_cfg.algo_log_name,
        checkpoint=playback_cfg.checkpoint,
        log_root=playback_cfg.log_root,
    )
    return checkpoint_path
