from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.distill.fada.model import FADA_COMMAND_SCENARIOS, FADAArchitectureConfig
from unilab.algos.torch.distill.learning.teacher import DistillationTeacherSpec

FADA_TRAINING_SCHEDULES = {
    "idm_pretrain",
    "alternating_idm_then_planner",
    "planner_from_idm",
}


def validate_fada_training_schedule(value: object) -> str:
    resolved = str(value)
    if resolved not in FADA_TRAINING_SCHEDULES:
        raise ValueError(
            "training.fada.training_schedule must be one of "
            f"{sorted(FADA_TRAINING_SCHEDULES)}, got {resolved!r}"
        )
    return resolved


def fada_training_schedule(fada_cfg: DictConfig) -> str:
    return validate_fada_training_schedule(
        OmegaConf.select(
            fada_cfg,
            "training_schedule",
            default="alternating_idm_then_planner",
        )
    )


def fada_runtime_device(cfg: DictConfig) -> str:
    configured = OmegaConf.select(cfg, "training.device", default="cpu")
    return "cpu" if configured in (None, "") else str(configured)


def allocate_fada_command_scenarios(
    total_windows: int,
    ratios: Mapping[str, float],
) -> tuple[tuple[str, int], ...]:
    """Allocate an exact scenario budget using stable largest remainder."""

    if int(total_windows) <= 0:
        raise ValueError(f"total_windows must be positive, got {total_windows}")
    unknown = set(ratios) - set(FADA_COMMAND_SCENARIOS)
    if unknown:
        raise ValueError(f"unknown FADA command scenarios: {sorted(unknown)}")
    values = [float(ratios.get(name, 0.0)) for name in FADA_COMMAND_SCENARIOS]
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("FADA command scenario ratios must be finite and non-negative")
    if abs(sum(values) - 1.0) > 1.0e-6:
        raise ValueError(f"FADA command scenario ratios must sum to 1, got {sum(values)}")
    positive_count = sum(value > 0.0 for value in values)
    if int(total_windows) < positive_count:
        raise ValueError(
            "total_windows must give every positive-ratio scenario at least one window: "
            f"total={total_windows} positive_scenarios={positive_count}"
        )

    raw = [int(total_windows) * value for value in values]
    counts = [int(value) for value in raw]
    for index, value in enumerate(values):
        if value > 0.0 and counts[index] == 0:
            counts[index] = 1
    while sum(counts) > int(total_windows):
        candidates = [index for index, count in enumerate(counts) if count > 1]
        if not candidates:
            raise ValueError("unable to preserve positive FADA scenario allocations")
        index = min(candidates, key=lambda item: (raw[item] - counts[item], -item))
        counts[index] -= 1
    while sum(counts) < int(total_windows):
        index = max(range(len(counts)), key=lambda item: (raw[item] - counts[item], -item))
        counts[index] += 1
    return tuple(
        (name, count)
        for name, count in zip(FADA_COMMAND_SCENARIOS, counts, strict=True)
        if count > 0
    )


def teacher_spec(cfg: DictConfig) -> DistillationTeacherSpec:
    algo_type = str(cfg.teacher.algo_type)
    if algo_type not in {"sac", "privileged_locomotion_sac", "fada_official_ppo", "fada_official_appo"}:
        raise ValueError(f"Unsupported FADA teacher algo_type: {algo_type!r}")
    if algo_type in {"fada_official_ppo", "fada_official_appo"}:
        return DistillationTeacherSpec(obs_dim=int(cfg.teacher.obs_dim),
            action_dim=int(cfg.teacher.action_dim), algo_type=cast(Any, algo_type),
            critic_obs_dim=int(cfg.teacher.critic_obs_dim))
    return DistillationTeacherSpec(
        obs_dim=int(cfg.teacher.obs_dim),
        action_dim=int(cfg.teacher.action_dim),
        algo_type=cast(Any, algo_type),
        actor_hidden_dim=int(cfg.teacher.actor_hidden_dim),
        use_layer_norm=bool(cfg.teacher.use_layer_norm),
        obs_normalization=bool(cfg.teacher.obs_normalization),
        critic_obs_dim=OmegaConf.select(cfg, "teacher.critic_obs_dim"),
        priv_info_embed_dim=int(OmegaConf.select(cfg, "teacher.priv_info_embed_dim", default=32)),
        priv_mlp_hidden_dims=tuple(
            int(value)
            for value in OmegaConf.select(
                cfg, "teacher.priv_mlp_hidden_dims", default=[256, 128, 32]
            )
        ),
        priv_info_normalization=bool(
            OmegaConf.select(cfg, "teacher.priv_info_normalization", default=True)
        ),
    )


def stand_transition_curriculum_cfg(fada_cfg: DictConfig) -> DictConfig:
    configured = OmegaConf.select(fada_cfg, "stand_transition_curriculum")
    defaults = OmegaConf.create(
        {
            "enabled": False,
            "standing_task": "g1_stand_still/mujoco",
            "walk_ratio": 1.0,
            "static_stand_ratio": 0.0,
            "walk_to_stand_ratio": 0.0,
            "walk_command": [0.4, 0.0, 0.0],
            "pre_switch_steps": 30,
            "post_switch_steps": 36,
        }
    )
    if configured is None:
        return defaults
    return cast(DictConfig, OmegaConf.merge(defaults, configured))


def v005_replay_cfg(fada_cfg: DictConfig) -> DictConfig:
    configured = OmegaConf.select(fada_cfg, "v005_replay")
    defaults = OmegaConf.create(
        {
            "enabled": False,
            "walk_cold_start_ratio": 0.2,
            "static_cold_start_ratio": 0.5,
            "planner_scenario_ratios": {
                "walk": 0.5,
                "static_stand": 0.25,
                "walk_to_stand": 0.25,
            },
            "walk_steady_speed_thresholds": [0.25, 0.6],
            "walk_steady_speed_ratios": {"slow": 0.1, "medium": 0.3, "high": 0.6},
            "min_high_speed_replay_passes": 8,
        }
    )
    return (
        defaults if configured is None else cast(DictConfig, OmegaConf.merge(defaults, configured))
    )


def v005_collection_profile_ratios(fada_cfg: DictConfig) -> dict[str, float]:
    """Resolve the Collector profile identity independently from replay resampling."""

    configured = OmegaConf.select(fada_cfg, "v005_collection_profile")
    defaults = OmegaConf.create(
        {"walk_cold_start_ratio": 0.5, "static_cold_start_ratio": 0.5}
    )
    resolved = (
        defaults if configured is None else cast(DictConfig, OmegaConf.merge(defaults, configured))
    )
    ratios = {
        "walk": float(resolved.walk_cold_start_ratio),
        "static_stand": float(resolved.static_cold_start_ratio),
    }
    if bool(v005_replay_cfg(fada_cfg).enabled) and any(
        not math.isclose(value, 0.5, rel_tol=0.0, abs_tol=1.0e-12)
        for value in ratios.values()
    ):
        raise ValueError("v005 collection cold-start ratios are fixed at walk/static=0.5/0.5")
    return ratios


def standing_owner_cfg(
    *,
    root_dir: Path,
    cfg: DictConfig,
    task_selector: str,
) -> DictConfig:
    """Compose the dedicated static-standing environment owner configuration."""

    task_path = Path(task_selector)
    if not task_selector or task_path.is_absolute() or ".." in task_path.parts:
        raise ValueError(
            f"standing curriculum task must be a relative owner selector, got {task_selector!r}"
        )
    owner_path = root_dir / "conf" / "distill" / "task" / task_path.with_suffix(".yaml")
    if not owner_path.is_file():
        raise FileNotFoundError(f"standing curriculum task owner does not exist: {owner_path}")

    base = cast(DictConfig, OmegaConf.load(root_dir / "conf" / "distill" / "config.yaml"))
    if "defaults" in base:
        del base["defaults"]
    standing_cfg = cast(
        DictConfig,
        OmegaConf.merge(
            base,
            OmegaConf.load(owner_path),
            {
                "algo": OmegaConf.to_container(cfg.algo, resolve=True),
                "student": OmegaConf.to_container(cfg.student, resolve=True),
                "training": {
                    "device": OmegaConf.select(cfg, "training.device"),
                    "fada": OmegaConf.to_container(cfg.training.fada, resolve=True),
                },
            },
        ),
    )
    if str(standing_cfg.training.task_name) != "G1StandStill":
        raise ValueError(
            "standing curriculum task owner must resolve to G1StandStill, got "
            f"{standing_cfg.training.task_name!r}"
        )
    if str(standing_cfg.training.sim_backend) != str(cfg.training.sim_backend):
        raise ValueError(
            "standing and walking FADA environments must use the same simulation backend"
        )
    return standing_cfg


def curriculum_and_allocations(
    fada_cfg: DictConfig,
    config: FADAArchitectureConfig,
) -> tuple[DictConfig, tuple[tuple[str, int], ...]]:
    curriculum = stand_transition_curriculum_cfg(fada_cfg)
    if not bool(curriculum.enabled):
        return curriculum, (("walk", int(fada_cfg.windows_per_iteration)),)
    if config.command_dim != 3:
        raise ValueError("standing curriculum requires FADA command_dim=3")
    command_keys = OmegaConf.to_container(fada_cfg.command_info_keys, resolve=True)
    if command_keys != ["commands"]:
        raise ValueError("standing curriculum requires command_info_keys=['commands']")
    walk_command = [float(value) for value in curriculum.walk_command]
    if (
        len(walk_command) != 3
        or not all(math.isfinite(value) for value in walk_command)
        or not any(abs(value) > 1.0e-6 for value in walk_command)
    ):
        raise ValueError("standing curriculum walk_command must be finite, active, and 3-D")
    if int(curriculum.pre_switch_steps) < config.history_length:
        raise ValueError("standing curriculum pre_switch_steps must be at least history_length")
    if int(curriculum.post_switch_steps) < config.prediction_horizon:
        raise ValueError(
            "standing curriculum post_switch_steps must be at least prediction_horizon"
        )
    allocations = allocate_fada_command_scenarios(
        int(fada_cfg.windows_per_iteration),
        {
            "walk": float(curriculum.walk_ratio),
            "static_stand": float(curriculum.static_stand_ratio),
            "walk_to_stand": float(curriculum.walk_to_stand_ratio),
        },
    )
    return curriculum, allocations
