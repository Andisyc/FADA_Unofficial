"""Production owner extracted from the generic distillation entrypoint: entry_collection.py."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.distill.collection.standard import (
    collect_distillation_dataset_from_env,
)
from unilab.algos.torch.distill.datasets.io import save_distillation_dataset
from unilab.algos.torch.distill.learning.dagger import run_iterative_dagger_updates
from unilab.algos.torch.distill.learning.playback import load_distillation_student_policy
from unilab.algos.torch.distill.learning.teacher import load_sac_teacher_policy
from unilab.algos.torch.distill.observability.performance import (
    COLLECTOR_REQUEST_STAGE_NAMES,
    DISTILLATION_METRICS_SCHEMA_VERSION,
    DistillationStageObservation,
)
from unilab.algos.torch.distill.workflows.entry_training import (
    _distill_runtime_cfg,
    _teacher_metadata,
    build_distillation_trainer,
    build_teacher_spec,
    resolve_teacher_checkpoint,
)
from unilab.training import BackendAdapter, create_env, ensure_registries

ROOT_DIR = Path(__file__).resolve().parents[6]

_OWNER_COMMAND_SAMPLE_FILTERS = {"G1WalkFlat": "active", "G1StandStill": "inactive"}
_HEIGHT_OWNER_COMMAND_SAMPLE_FILTERS = {"G1WalkHeight": "active", "G1StandHeight": "inactive"}
_DISTILL_TASK_NAME_HINTS = frozenset(
    {*_OWNER_COMMAND_SAMPLE_FILTERS, *_HEIGHT_OWNER_COMMAND_SAMPLE_FILTERS}
)


def _collect_action_mode(cfg: DictConfig) -> str:
    return str(OmegaConf.select(cfg, "training.collect_action_mode", default="zero"))


def _distill_device(cfg: DictConfig) -> str:
    device = OmegaConf.select(cfg, "training.device", default="cpu")
    return "cpu" if device in (None, "") else str(device)


def _teacher_task_name_for_collection(cfg: DictConfig) -> str:
    teacher_task_name = str(
        OmegaConf.select(
            cfg,
            "teacher.task_name",
            default=str(OmegaConf.select(cfg, "training.task_name")),
        )
    )
    checkpoint_path = OmegaConf.select(cfg, "teacher.checkpoint_path")
    if checkpoint_path in (None, ""):
        return teacher_task_name
    checkpoint_parts = set(Path(str(checkpoint_path)).parts)
    hinted_task_names = sorted(_DISTILL_TASK_NAME_HINTS & checkpoint_parts)
    if not hinted_task_names:
        return teacher_task_name
    if len(hinted_task_names) > 1:
        raise ValueError(
            f"teacher.checkpoint_path contains multiple distill task hints: {hinted_task_names}"
        )
    hinted_task_name = hinted_task_names[0]
    if teacher_task_name != hinted_task_name:
        default_task_name = str(OmegaConf.select(cfg, "training.task_name"))
        if teacher_task_name != default_task_name:
            raise ValueError(
                "teacher.task_name conflicts with teacher.checkpoint_path task hint: "
                f"teacher.task_name={teacher_task_name!r}, checkpoint_hint={hinted_task_name!r}"
            )
        teacher_task_name = hinted_task_name
    return teacher_task_name


def _expected_owner_command_sample_filter(cfg: DictConfig) -> str | None:
    task_name = str(OmegaConf.select(cfg, "training.task_name"))
    teacher_task_name = _teacher_task_name_for_collection(cfg)
    if task_name == "G1WalkFlat" and teacher_task_name == "G1StandStill":
        return "inactive"
    target_height_info_key = OmegaConf.select(cfg, "training.collect_target_height_info_key")
    if target_height_info_key not in (None, ""):
        return _HEIGHT_OWNER_COMMAND_SAMPLE_FILTERS.get(task_name)
    return _OWNER_COMMAND_SAMPLE_FILTERS.get(task_name)


def _require_owner_command_sample_filter(cfg: DictConfig) -> None:
    expected_filter = _expected_owner_command_sample_filter(cfg)
    if expected_filter is None:
        return
    actual_filter = str(
        OmegaConf.select(cfg, "training.collect_command_sample_filter", default="none")
    )
    if actual_filter != expected_filter:
        task_name = str(OmegaConf.select(cfg, "training.task_name"))
        raise ValueError(
            f"{task_name} requires training.collect_command_sample_filter={expected_filter} "
            f"for command-intent distillation collection; got {actual_filter!r}"
        )


def _require_collected_command_intent_contract(cfg: DictConfig, dataset: Any) -> None:
    expected_filter = _expected_owner_command_sample_filter(cfg)
    if expected_filter is None:
        return
    expected_intent = "active" if expected_filter == "active" else "inactive"
    actual_filter = str(dataset.metadata.get("command_sample_filter", "none"))
    if actual_filter != expected_filter:
        raise ValueError(
            "collected dataset command filter mismatch: "
            f"expected {expected_filter!r}, got {actual_filter!r}"
        )
    if dataset.commands is None:
        raise ValueError("owner command-intent collection must persist dataset.commands")
    if dataset.command_intents is None:
        raise ValueError("owner command-intent collection must persist dataset.command_intents")
    intent_counts = dict(dataset.metadata.get("command_intent_counts") or {})
    expected_counts = {expected_intent: int(dataset.num_samples)}
    if intent_counts != expected_counts:
        raise ValueError(
            "collected dataset command intent mismatch: "
            f"expected {expected_counts}, got {intent_counts}"
        )
    seen_samples = dataset.metadata.get("command_seen_samples")
    selected_samples = dataset.metadata.get("command_selected_samples")
    if seen_samples is None or selected_samples is None:
        raise ValueError(
            "owner command-intent collection must record command_seen/selected samples"
        )
    if int(selected_samples) < int(dataset.num_samples):
        raise ValueError(
            "owner command-intent collection selected too few samples: "
            f"selected={selected_samples}, dataset_num_samples={dataset.num_samples}"
        )


def _require_collected_target_height_contract(cfg: DictConfig, dataset: Any) -> None:
    info_key = OmegaConf.select(cfg, "training.collect_target_height_info_key")
    if info_key in (None, ""):
        return
    if str(info_key) != "height_commands":
        raise ValueError(
            "99-D height collection requires "
            "training.collect_target_height_info_key=height_commands"
        )
    target_height = dataset.target_height
    if target_height is None:
        raise ValueError("99-D height collection must persist dataset.target_height")
    if tuple(target_height.shape) != (int(dataset.num_samples), 1):
        raise ValueError(
            "collected target_height shape mismatch: "
            f"expected={(int(dataset.num_samples), 1)} got={tuple(target_height.shape)}"
        )
    if int(dataset.student_obs_dim) != 99 or int(dataset.teacher_obs_dim) != 99:
        raise ValueError("height-aware role data requires 99-D student and teacher observations")
    if not torch.equal(dataset.student_obs[:, 96:97], target_height):
        raise ValueError(
            "student observation target-height column does not match dataset.target_height"
        )
    if not torch.equal(dataset.teacher_obs[:, 96:97], target_height):
        raise ValueError(
            "teacher observation target-height column does not match dataset.target_height"
        )
    if str(OmegaConf.select(cfg, "training.task_name")) == "G1WalkHeight":
        if bool(OmegaConf.select(cfg, "env.commands.random_height_during_walking")):
            raise ValueError("Walk role must use fixed nominal target height")
        nominal_height = float(OmegaConf.select(cfg, "env.commands.default_height"))
        if not torch.equal(target_height, torch.full_like(target_height, nominal_height)):
            raise ValueError("Walk role target height must stay at its nominal owner-config value")


def _collect_command_distribution_overrides(cfg: DictConfig) -> dict[str, Any]:
    expected_filter = _expected_owner_command_sample_filter(cfg)
    actual_filter = str(
        OmegaConf.select(cfg, "training.collect_command_sample_filter", default="none")
    )
    task_name = str(OmegaConf.select(cfg, "training.task_name"))
    teacher_task_name = _teacher_task_name_for_collection(cfg)
    if (
        task_name == "G1WalkFlat"
        and teacher_task_name == "G1StandStill"
        and expected_filter == "inactive"
        and actual_filter == "inactive"
    ):
        return {
            "env.commands.vel_limit": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            "env.commands.transition_vel_limit": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            "env.commands.rel_standing_envs": 1.0,
            "env.commands.rel_transition_envs": 0.0,
            "env.commands.small_xy_threshold": 0.0,
        }
    return {}


def _apply_collect_command_distribution_overrides(cfg: DictConfig) -> dict[str, Any]:
    overrides = _collect_command_distribution_overrides(cfg)
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, merge=False, force_add=True)
    return overrides


def _require_teacher_policy_collection_route(cfg: DictConfig) -> None:
    """Require an explicit legacy-98-D or height-aware-99-D owner route."""

    task_name = str(OmegaConf.select(cfg, "training.task_name"))
    teacher_task_name = _teacher_task_name_for_collection(cfg)
    if task_name == "G1OfficialOracleFlat":
        if (teacher_task_name != task_name or str(cfg.teacher.algo_type) != "fada_official_ppo"
                or not bool(OmegaConf.select(cfg, "training.fada.enabled", default=False))
                or int(cfg.teacher.obs_dim) != 322 or int(cfg.teacher.critic_obs_dim) != 322
                or int(cfg.student.obs_dim) != 66 or int(cfg.teacher.action_dim) != 29
                or OmegaConf.select(cfg, "training.fada.observation_contract") != "g1_official_oracle_student66_v1"):
            raise ValueError("Official teacher collection requires its dedicated PPO/Planner-IDM route")
        return
    legacy_tasks = {"G1WalkFlat", "G1StandStill"}
    height_tasks = {"G1WalkHeight", "G1StandHeight"}
    if task_name not in legacy_tasks | height_tasks:
        raise ValueError(
            "teacher target collection only supports explicit G1 flat/stand or "
            "height-aware owner routes"
        )
    cross_stand_teacher = task_name == "G1WalkFlat" and teacher_task_name == "G1StandStill"
    if teacher_task_name != task_name and not cross_stand_teacher:
        raise ValueError(
            "teacher target collection requires teacher.task_name to match training.task_name, "
            "except G1WalkFlat inactive collection may use a G1StandStill teacher"
        )
    if cross_stand_teacher:
        actual_filter = str(
            OmegaConf.select(cfg, "training.collect_command_sample_filter", default="none")
        )
        if actual_filter != "inactive":
            raise ValueError(
                "G1WalkFlat collection with a G1StandStill teacher requires "
                "training.collect_command_sample_filter=inactive"
            )
    expected_obs_dim = 98 if task_name in legacy_tasks else 99
    if int(cfg.teacher.obs_dim) != expected_obs_dim or int(cfg.student.obs_dim) != expected_obs_dim:
        raise ValueError(
            f"teacher target collection requires {expected_obs_dim}-D teacher and student obs"
        )
    if str(OmegaConf.select(cfg, "training.collect_teacher_obs_key", default="obs")) != "obs":
        raise ValueError("teacher target collection requires training.collect_teacher_obs_key=obs")
    if (
        str(OmegaConf.select(cfg, "training.collect_teacher_projection", default="identity"))
        != "identity"
    ):
        raise ValueError("teacher target collection requires identity teacher projection")
    if (
        str(OmegaConf.select(cfg, "training.collect_student_projection", default="identity"))
        != "identity"
    ):
        raise ValueError("teacher target collection requires identity student projection")
    if OmegaConf.select(cfg, "training.collect_student_drop_index") is not None:
        raise ValueError("teacher target collection does not support collect_student_drop_index")
    if OmegaConf.select(cfg, "training.collect_action_seed") is not None:
        raise ValueError("teacher target collection does not use training.collect_action_seed")
    observes_height = bool(
        OmegaConf.select(cfg, "env.commands.observe_height_command", default=False)
    )
    target_height_info_key = OmegaConf.select(cfg, "training.collect_target_height_info_key")
    if task_name in legacy_tasks:
        if observes_height or target_height_info_key not in (None, ""):
            raise ValueError("98-D teacher target collection must not use height commands")
    elif not observes_height or str(target_height_info_key) != "height_commands":
        raise ValueError(
            "99-D teacher target collection requires observed height_commands and "
            "training.collect_target_height_info_key=height_commands"
        )


def _resolve_collect_rollout_checkpoint(cfg: DictConfig) -> Path:
    checkpoint_path = OmegaConf.select(cfg, "training.collect_rollout_checkpoint_path")
    if checkpoint_path in (None, ""):
        raise ValueError(
            "training.collect_rollout_checkpoint_path must be set when "
            "training.collect_action_mode=student_policy"
        )
    path = Path(str(checkpoint_path))
    resolved_path = path if path.is_absolute() else ROOT_DIR / path
    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"training.collect_rollout_checkpoint_path does not exist: {resolved_path}"
        )
    return resolved_path


def _collection_metadata(
    cfg: DictConfig,
    *,
    teacher_checkpoint: Path | None,
    rollout_checkpoint: Path | None,
    command_distribution_overrides: dict[str, Any],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "task_name": str(OmegaConf.select(cfg, "training.task_name")),
        "sim_backend": str(OmegaConf.select(cfg, "training.sim_backend", default="mujoco")),
    }
    scenario = OmegaConf.select(cfg, "training.collect_workflow_scenario")
    if scenario not in (None, ""):
        metadata["workflow_scenario"] = str(scenario)
    if teacher_checkpoint is not None:
        metadata["teacher_policy_checkpoint_path"] = str(teacher_checkpoint)
    if rollout_checkpoint is not None:
        metadata["rollout_policy_checkpoint_path"] = str(rollout_checkpoint)
    if command_distribution_overrides:
        metadata["command_distribution_overrides"] = command_distribution_overrides
    return metadata


def _collection_result(
    cfg: DictConfig,
    *,
    dataset_path: str | Path,
    dataset: Any,
    action_mode: str,
    drop_index: Any,
    teacher_checkpoint: Path | None,
    rollout_checkpoint: Path | None,
    request_observations: tuple[DistillationStageObservation, ...] | None,
) -> dict[str, Any]:
    result = {
        "distill_source": "live_env_rollout",
        "dataset_path": str(dataset_path),
        "dataset_num_samples": dataset.num_samples,
        "dataset_student_obs_dim": dataset.student_obs_dim,
        "dataset_teacher_obs_dim": dataset.teacher_obs_dim,
        "dataset_metadata": dict(dataset.metadata),
        "student_obs_shape": tuple(dataset.student_obs.shape),
        "teacher_obs_shape": tuple(dataset.teacher_obs.shape),
        "collect_num_envs": int(OmegaConf.select(cfg, "training.collect_num_envs", default=1)),
        "collect_action_mode": action_mode,
        "collect_action_seed": OmegaConf.select(cfg, "training.collect_action_seed"),
        "collect_action_abs_max": float(dataset.metadata.get("action_abs_max", 0.0)),
        "teacher_policy_checkpoint_path": (
            str(teacher_checkpoint) if teacher_checkpoint is not None else None
        ),
        "rollout_policy_checkpoint_path": (
            str(rollout_checkpoint) if rollout_checkpoint is not None else None
        ),
        "teacher_obs_key": str(
            OmegaConf.select(cfg, "training.collect_teacher_obs_key", default="obs")
        ),
        "teacher_projection": str(
            OmegaConf.select(cfg, "training.collect_teacher_projection", default="identity")
        ),
        "student_projection": str(
            OmegaConf.select(cfg, "training.collect_student_projection", default="identity")
        ),
        "student_drop_index": None if drop_index is None else int(drop_index),
        "collect_command_sample_filter": str(
            dataset.metadata.get("command_sample_filter", "none")
        ),
        "collect_command_seen_samples": dataset.metadata.get("command_seen_samples"),
        "collect_command_selected_samples": dataset.metadata.get("command_selected_samples"),
        "collect_command_intent_counts": dataset.metadata.get("command_intent_counts"),
        "collect_target_height_shape": (
            None if dataset.target_height is None else tuple(dataset.target_height.shape)
        ),
        "collect_command_distribution_overrides": dataset.metadata.get(
            "command_distribution_overrides"
        ),
    }
    if request_observations is not None:
        result["performance_metrics_schema_version"] = DISTILLATION_METRICS_SCHEMA_VERSION
        result["performance_stage_observations"] = [
            observation.as_dict() for observation in request_observations
        ]
    return result


@dataclass(frozen=True)
class _PreparedCollectionPolicies:
    action_mode: str
    teacher_policy: Any | None
    rollout_policy: Any | None
    teacher_checkpoint: Path | None
    rollout_checkpoint: Path | None
    command_distribution_overrides: dict[str, Any]


def _prepare_collection_policies(cfg: DictConfig) -> _PreparedCollectionPolicies:
    action_mode = _collect_action_mode(cfg)
    _require_owner_command_sample_filter(cfg)
    overrides = _apply_collect_command_distribution_overrides(cfg)
    teacher_policy = None
    rollout_policy = None
    teacher_checkpoint = None
    rollout_checkpoint = None
    if action_mode in {"teacher_policy", "student_policy"}:
        _require_teacher_policy_collection_route(cfg)
        teacher_checkpoint, _run_dir = resolve_teacher_checkpoint(cfg, root_dir=ROOT_DIR)
        if teacher_checkpoint is None:
            raise FileNotFoundError(
                "No SAC teacher checkpoint resolved for teacher target collection. "
                "Set teacher.load_run/teacher.checkpoint or training.log_root."
            )
        teacher_policy = load_sac_teacher_policy(
            teacher_checkpoint, build_teacher_spec(cfg), device=_distill_device(cfg)
        )
    if action_mode == "student_policy":
        rollout_checkpoint = _resolve_collect_rollout_checkpoint(cfg)
        loaded = load_distillation_student_policy(
            rollout_checkpoint, device=_distill_device(cfg)
        )
        if int(loaded.obs_dim) != int(cfg.student.obs_dim):
            raise ValueError(
                "student_policy rollout obs dim mismatch: "
                f"checkpoint={loaded.obs_dim} cfg.student.obs_dim={int(cfg.student.obs_dim)}"
            )
        if int(loaded.action_dim) != int(cfg.student.action_dim):
            raise ValueError(
                "student_policy rollout action dim mismatch: "
                f"checkpoint={loaded.action_dim} "
                f"cfg.student.action_dim={int(cfg.student.action_dim)}"
            )
        rollout_policy = loaded.policy
    return _PreparedCollectionPolicies(
        action_mode=action_mode,
        teacher_policy=teacher_policy,
        rollout_policy=rollout_policy,
        teacher_checkpoint=teacher_checkpoint,
        rollout_checkpoint=rollout_checkpoint,
        command_distribution_overrides=overrides,
    )


def _execute_collect_dataset(
    cfg: DictConfig,
    *,
    dataset_path: str | Path | None = None,
    create_env_fn: Any | None = None,
    env_cfg_override_fn: Any | None = None,
    performance_clock: Callable[[], float] | None = None,
) -> dict[str, Any]:
    """Collect and save a small live-env distillation observation dataset."""

    request_start = None if performance_clock is None else float(performance_clock())
    resolved_dataset_path = dataset_path or OmegaConf.select(cfg, "training.collect_dataset_path")
    if resolved_dataset_path in (None, ""):
        raise ValueError("training.collect_dataset_path must be set for live dataset collection")

    policies = _prepare_collection_policies(cfg)
    action_mode = policies.action_mode
    teacher_policy_checkpoint_path = policies.teacher_checkpoint
    rollout_policy_checkpoint_path = policies.rollout_checkpoint

    if create_env_fn is None:
        ensure_registries()
        create_env_fn = create_env
    if env_cfg_override_fn is None:
        env_cfg_override_fn = lambda cfg: BackendAdapter(  # noqa: E731
            cfg,
            root_dir=ROOT_DIR,
            algo_name="distill",
        ).build_task_env_cfg_override()

    env = create_env_fn(
        cfg,
        num_envs=int(OmegaConf.select(cfg, "training.collect_num_envs", default=1)),
        env_cfg_override=env_cfg_override_fn(cfg),
        sim_backend=str(OmegaConf.select(cfg, "training.sim_backend", default="mujoco")),
        task_name=str(OmegaConf.select(cfg, "training.task_name")),
    )
    cold_start_seconds = (
        None
        if performance_clock is None or request_start is None
        else float(performance_clock()) - request_start
    )
    request_observations: tuple[DistillationStageObservation, ...] | None = None
    try:
        drop_index = OmegaConf.select(cfg, "training.collect_student_drop_index")
        collect_max_env_steps = OmegaConf.select(cfg, "training.collect_max_env_steps")
        metadata = _collection_metadata(
            cfg,
            teacher_checkpoint=teacher_policy_checkpoint_path,
            rollout_checkpoint=rollout_policy_checkpoint_path,
            command_distribution_overrides=policies.command_distribution_overrides,
        )
        dataset = collect_distillation_dataset_from_env(
            env,
            num_samples=int(OmegaConf.select(cfg, "training.collect_num_samples", default=1024)),
            expected_student_obs_dim=int(cfg.student.obs_dim),
            expected_teacher_obs_dim=int(cfg.teacher.obs_dim),
            teacher_obs_key=str(
                OmegaConf.select(cfg, "training.collect_teacher_obs_key", default="obs")
            ),
            teacher_projection=str(
                OmegaConf.select(cfg, "training.collect_teacher_projection", default="identity")
            ),
            student_projection=str(
                OmegaConf.select(cfg, "training.collect_student_projection", default="identity")
            ),
            student_drop_index=None if drop_index is None else int(drop_index),
            action_mode=action_mode,
            action_seed=OmegaConf.select(cfg, "training.collect_action_seed"),
            teacher_policy=policies.teacher_policy,
            rollout_policy=policies.rollout_policy,
            command_sample_filter=str(
                OmegaConf.select(cfg, "training.collect_command_sample_filter", default="none")
            ),
            command_info_key=str(
                OmegaConf.select(cfg, "training.collect_command_info_key", default="commands")
            ),
            target_height_info_key=OmegaConf.select(cfg, "training.collect_target_height_info_key"),
            command_xy_threshold=float(
                OmegaConf.select(cfg, "training.collect_command_xy_threshold", default=0.05)
            ),
            command_yaw_threshold=float(
                OmegaConf.select(cfg, "training.collect_command_yaw_threshold", default=0.05)
            ),
            max_env_steps=None if collect_max_env_steps is None else int(collect_max_env_steps),
            role_label=OmegaConf.select(cfg, "training.collect_role_label"),
            metadata=metadata,
            performance_clock=performance_clock,
        )
        _require_collected_command_intent_contract(cfg, dataset)
        _require_collected_target_height_contract(cfg, dataset)
        write_start = None if performance_clock is None else float(performance_clock())
        save_distillation_dataset(resolved_dataset_path, dataset)
        if performance_clock is not None:
            assert request_start is not None
            assert cold_start_seconds is not None
            assert write_start is not None
            artifact_write_seconds = float(performance_clock()) - write_start
            collector_payloads = dataset.metadata.get("performance_stage_observations")
            if not isinstance(collector_payloads, list):
                raise ValueError("legacy collector performance observations are missing")
            collector_observations = tuple(
                DistillationStageObservation.from_dict(payload) for payload in collector_payloads
            )
            collector_stages = tuple(item.stage for item in collector_observations)
            if collector_stages != COLLECTOR_REQUEST_STAGE_NAMES:
                raise ValueError(
                    "legacy collector performance stage order mismatch: "
                    f"expected={COLLECTOR_REQUEST_STAGE_NAMES} "
                    f"observed={collector_stages}"
                )
            total_elapsed_seconds = float(performance_clock()) - request_start
            env_steps = int(dataset.metadata.get("env_steps", 0))
            request_observations = (
                DistillationStageObservation(
                    stage="cold_start",
                    duration_seconds=cold_start_seconds,
                    row_count=0,
                    env_step_count=0,
                    success=True,
                    error=None,
                    cleanup_state="not_applicable",
                ),
                *collector_observations,
                DistillationStageObservation(
                    stage="artifact_write",
                    duration_seconds=artifact_write_seconds,
                    row_count=dataset.num_samples,
                    env_step_count=0,
                    success=True,
                    error=None,
                    cleanup_state="not_applicable",
                ),
                DistillationStageObservation(
                    stage="total_elapsed",
                    duration_seconds=total_elapsed_seconds,
                    row_count=dataset.num_samples,
                    env_step_count=env_steps,
                    success=True,
                    error=None,
                    cleanup_state="pending",
                ),
            )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    return _collection_result(
        cfg,
        dataset_path=resolved_dataset_path,
        dataset=dataset,
        action_mode=action_mode,
        drop_index=drop_index,
        teacher_checkpoint=teacher_policy_checkpoint_path,
        rollout_checkpoint=rollout_policy_checkpoint_path,
        request_observations=request_observations,
    )


@dataclass(frozen=True)
class _CollectionEntryContext:
    """Own one collection request and its injected environment dependencies."""

    cfg: DictConfig
    dataset_path: str | Path | None = None
    create_env_fn: Any | None = None
    env_cfg_override_fn: Any | None = None
    performance_clock: Callable[[], float] | None = None

    def collect(self) -> dict[str, Any]:
        return _execute_collect_dataset(
            self.cfg,
            dataset_path=self.dataset_path,
            create_env_fn=self.create_env_fn,
            env_cfg_override_fn=self.env_cfg_override_fn,
            performance_clock=self.performance_clock,
        )


def run_collect_dataset(
    cfg: DictConfig,
    *,
    dataset_path: str | Path | None = None,
    create_env_fn: Any | None = None,
    env_cfg_override_fn: Any | None = None,
    performance_clock: Callable[[], float] | None = None,
) -> dict[str, Any]:
    """Collect and persist one live-env distillation dataset transaction."""

    return _CollectionEntryContext(
        cfg=cfg,
        dataset_path=dataset_path,
        create_env_fn=create_env_fn,
        env_cfg_override_fn=env_cfg_override_fn,
        performance_clock=performance_clock,
    ).collect()


def run_online_dagger_update(
    cfg: DictConfig,
    *,
    teacher_checkpoint: str | Path,
    create_env_fn: Any | None = None,
    env_cfg_override_fn: Any | None = None,
) -> dict[str, Any]:
    """Assemble the iterative student-rollout DAgger owner loop."""

    _require_owner_command_sample_filter(cfg)
    _require_teacher_policy_collection_route(cfg)
    command_distribution_overrides = _apply_collect_command_distribution_overrides(cfg)
    init_checkpoint = OmegaConf.select(cfg, "training.offline_init_checkpoint")
    if init_checkpoint in (None, ""):
        raise ValueError("training.offline_init_checkpoint must be set for online DAgger")
    output_checkpoint = OmegaConf.select(cfg, "training.dagger_checkpoint")
    if output_checkpoint in (None, ""):
        raise ValueError("training.dagger_checkpoint must be set for online DAgger")
    output_checkpoint = Path(str(output_checkpoint))

    role_label = OmegaConf.select(cfg, "training.dagger_role_label")
    if float(OmegaConf.select(cfg, "algo.role_loss_coef", default=0.0)) > 0.0 and role_label in (
        None,
        "",
    ):
        raise ValueError("training.dagger_role_label is required when algo.role_loss_coef > 0")

    device = _distill_device(cfg)
    trainer = build_distillation_trainer(
        cfg,
        teacher_checkpoint=teacher_checkpoint,
        student_init_checkpoint=init_checkpoint,
        device=device,
    )
    if create_env_fn is None:
        ensure_registries()
        create_env_fn = create_env
    if env_cfg_override_fn is None:
        env_cfg_override_fn = lambda cfg: BackendAdapter(  # noqa: E731
            cfg,
            root_dir=ROOT_DIR,
            algo_name="distill",
        ).build_task_env_cfg_override()

    env = create_env_fn(
        cfg,
        num_envs=int(OmegaConf.select(cfg, "training.collect_num_envs", default=1)),
        env_cfg_override=env_cfg_override_fn(cfg),
        sim_backend=str(OmegaConf.select(cfg, "training.sim_backend", default="mujoco")),
        task_name=str(OmegaConf.select(cfg, "training.task_name")),
    )
    try:
        drop_index = OmegaConf.select(cfg, "training.collect_student_drop_index")
        max_env_steps = OmegaConf.select(cfg, "training.collect_max_env_steps")
        result = run_iterative_dagger_updates(
            env,
            trainer=trainer,
            num_iterations=int(OmegaConf.select(cfg, "training.dagger_iterations", default=8)),
            samples_per_iteration=int(
                OmegaConf.select(cfg, "training.dagger_samples_per_iteration", default=65536)
            ),
            batch_size=int(OmegaConf.select(cfg, "training.dagger_batch_size", default=512)),
            updates_per_iteration=int(
                OmegaConf.select(cfg, "training.dagger_updates_per_iteration", default=128)
            ),
            expected_student_obs_dim=int(cfg.student.obs_dim),
            expected_teacher_obs_dim=int(cfg.teacher.obs_dim),
            teacher_obs_key=str(
                OmegaConf.select(cfg, "training.collect_teacher_obs_key", default="obs")
            ),
            teacher_projection=str(
                OmegaConf.select(cfg, "training.collect_teacher_projection", default="identity")
            ),
            student_projection=str(
                OmegaConf.select(cfg, "training.collect_student_projection", default="identity")
            ),
            student_drop_index=None if drop_index is None else int(drop_index),
            command_sample_filter=str(
                OmegaConf.select(cfg, "training.collect_command_sample_filter", default="none")
            ),
            command_info_key=str(
                OmegaConf.select(cfg, "training.collect_command_info_key", default="commands")
            ),
            target_height_info_key=OmegaConf.select(cfg, "training.collect_target_height_info_key"),
            command_xy_threshold=float(
                OmegaConf.select(cfg, "training.collect_command_xy_threshold", default=0.05)
            ),
            command_yaw_threshold=float(
                OmegaConf.select(cfg, "training.collect_command_yaw_threshold", default=0.05)
            ),
            max_env_steps=None if max_env_steps is None else int(max_env_steps),
            role_label=None if role_label in (None, "") else str(role_label),
            shuffle=bool(OmegaConf.select(cfg, "training.dagger_shuffle", default=True)),
            seed=int(cfg.algo.seed),
            balance_key=str(OmegaConf.select(cfg, "training.dagger_balance_key", default="none")),
            balanced_labels=list(
                OmegaConf.select(cfg, "training.dagger_balanced_labels", default=[])
            ),
            checkpoint_path=output_checkpoint,
            teacher_metadata=_teacher_metadata(cfg, teacher_checkpoint),
            distill_runtime_cfg=_distill_runtime_cfg(
                cfg,
                distill_source="iterative_dagger",
                student_init_metadata=dict(getattr(trainer, "student_init_metadata", {})),
            ),
        )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    last_result = result.iteration_results[-1]
    return {
        "distill_source": "iterative_dagger",
        "iteration_count": result.iteration_count,
        "update_count": result.update_count,
        "samples_collected": result.samples_collected,
        "samples_seen": result.samples_seen,
        "loss": last_result.last_loss,
        "checkpoint_path": str(result.checkpoint_path),
        "role_label": role_label,
        "command_sample_filter": str(
            OmegaConf.select(cfg, "training.collect_command_sample_filter", default="none")
        ),
        "command_distribution_overrides": command_distribution_overrides,
    }
