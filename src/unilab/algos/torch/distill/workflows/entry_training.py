"""Production owner extracted from the generic distillation entrypoint: entry_training.py."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, cast

import torch
from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.distill.contracts.checkpoint import load_distillation_checkpoint
from unilab.algos.torch.distill.datasets.dataset import make_fake_distillation_dataset
from unilab.algos.torch.distill.datasets.io import (
    load_distillation_dataset,
    save_distillation_dataset,
)
from unilab.algos.torch.distill.datasets.merge import build_multitask_distillation_dataset
from unilab.algos.torch.distill.learning.models import MLPStudentPolicy
from unilab.algos.torch.distill.learning.moe_student import MoEStudentPolicy
from unilab.algos.torch.distill.learning.offline import (
    required_balanced_replay_updates,
    run_offline_distillation_updates,
)
from unilab.algos.torch.distill.learning.playback import load_distillation_student_policy
from unilab.algos.torch.distill.learning.teacher import (
    DistillationTeacherSpec,
    load_sac_teacher_policy,
    validate_sac_teacher_checkpoint_contract,
)
from unilab.algos.torch.distill.learning.trainer import BehaviorDistillationTrainer
from unilab.training import ExperimentTracker
from unilab.training.run import resolve_task_checkpoint_path

ROOT_DIR = Path(__file__).resolve().parents[6]
_ROLE_DATA_ASSEMBLY_DEVICE = "cpu"


def _int_tuple(values: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in values)


def _student_model_type(cfg: DictConfig) -> str:
    return str(OmegaConf.select(cfg, "student.model_type", default="mlp"))


def build_teacher_spec(cfg: DictConfig) -> DistillationTeacherSpec:
    """Build the frozen teacher load contract from owner config."""

    if str(cfg.teacher.algo_type) in {"fada_official_ppo", "fada_official_appo"}:
        from unilab.algos.torch.distill.fada.async_config import teacher_spec
        return teacher_spec(cfg)
    return DistillationTeacherSpec(
        obs_dim=int(cfg.teacher.obs_dim),
        action_dim=int(cfg.teacher.action_dim),
        algo_type=str(cfg.teacher.algo_type),
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


def build_student_policy(
    cfg: DictConfig,
    *,
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    """Build the student actor from owner config."""

    model_type = _student_model_type(cfg)
    if model_type == "mlp":
        student = MLPStudentPolicy(
            obs_dim=int(cfg.student.obs_dim),
            action_dim=int(cfg.student.action_dim),
            hidden_dims=_int_tuple(cfg.student.hidden_dims),
            activation=str(cfg.student.activation),
            squash_action=bool(cfg.student.squash_action),
        )
    elif model_type == "moe":
        student = MoEStudentPolicy(
            obs_dim=int(cfg.student.obs_dim),
            action_dim=int(cfg.student.action_dim),
            num_experts=int(cfg.student.num_experts),
            expert_hidden_dims=_int_tuple(cfg.student.expert_hidden_dims),
            router_hidden_dims=_int_tuple(cfg.student.router_hidden_dims),
            activation=str(cfg.student.activation),
            squash_action=bool(cfg.student.squash_action),
            routing_mode=str(cfg.student.routing_mode),
            router_temperature=float(cfg.student.router_temperature),
        )
    else:
        raise ValueError(f"Unsupported distillation student.model_type: {model_type!r}")
    return cast(torch.nn.Module, student.to(device))


def _student_runtime_cfg(cfg: DictConfig) -> dict[str, Any]:
    model_type = _student_model_type(cfg)
    payload: dict[str, Any] = {
        "student_model_type": model_type,
        "student_obs_dim": int(cfg.student.obs_dim),
        "student_action_dim": int(cfg.student.action_dim),
        "student_activation": str(cfg.student.activation),
        "student_squash_action": bool(cfg.student.squash_action),
    }
    if model_type == "mlp":
        payload["student_hidden_dims"] = [int(dim) for dim in cfg.student.hidden_dims]
    elif model_type == "moe":
        payload.update(
            {
                "student_num_experts": int(cfg.student.num_experts),
                "student_expert_hidden_dims": [int(dim) for dim in cfg.student.expert_hidden_dims],
                "student_router_hidden_dims": [int(dim) for dim in cfg.student.router_hidden_dims],
                "student_routing_mode": str(cfg.student.routing_mode),
                "student_router_temperature": float(cfg.student.router_temperature),
            }
        )
    else:
        raise ValueError(f"Unsupported distillation student.model_type: {model_type!r}")
    return payload


def _resolve_optional_checkpoint_path(
    checkpoint_path: str | Path | None,
    *,
    root_dir: str | Path = ROOT_DIR,
    field_name: str,
) -> Path | None:
    if checkpoint_path in (None, ""):
        return None
    path = Path(str(checkpoint_path))
    resolved_path = path if path.is_absolute() else Path(root_dir) / path
    if not resolved_path.is_file():
        raise FileNotFoundError(f"{field_name} does not exist: {resolved_path}")
    return resolved_path


def _runtime_cfg_subset_for_student(cfg: DictConfig) -> dict[str, Any]:
    runtime_cfg = _student_runtime_cfg(cfg)
    if runtime_cfg["student_model_type"] == "moe":
        return {
            key: runtime_cfg[key]
            for key in (
                "student_model_type",
                "student_obs_dim",
                "student_action_dim",
                "student_activation",
                "student_squash_action",
                "student_num_experts",
                "student_expert_hidden_dims",
                "student_router_hidden_dims",
                "student_routing_mode",
            )
        }
    return {
        key: runtime_cfg[key]
        for key in (
            "student_model_type",
            "student_obs_dim",
            "student_action_dim",
            "student_activation",
            "student_squash_action",
            "student_hidden_dims",
        )
    }


def _validate_student_init_runtime_cfg(
    cfg: DictConfig,
    *,
    runtime_cfg: dict[str, Any],
    checkpoint_path: Path,
) -> None:
    expected = _runtime_cfg_subset_for_student(cfg)
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        actual_value = runtime_cfg.get(key)
        if actual_value != expected_value:
            mismatches.append(f"{key}: expected {expected_value!r}, got {actual_value!r}")
    if mismatches:
        raise ValueError(
            "training.offline_init_checkpoint student runtime config mismatch for "
            f"{checkpoint_path}: " + "; ".join(mismatches)
        )


def _load_student_init_checkpoint(
    student: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path,
    *,
    cfg: DictConfig,
    device: str | torch.device,
    resume_optimizer: bool,
) -> dict[str, Any]:
    loaded_student = load_distillation_student_policy(checkpoint_path, device=device)
    runtime_cfg = dict(loaded_student.distill_runtime_cfg)
    _validate_student_init_runtime_cfg(
        cfg,
        runtime_cfg=runtime_cfg,
        checkpoint_path=checkpoint_path,
    )
    checkpoint = load_distillation_checkpoint(
        student,
        checkpoint_path,
        optimizer=optimizer if resume_optimizer else None,
        device=device,
    )
    return {
        "path": str(checkpoint_path),
        "agent_steps": int(checkpoint.get("agent_steps", loaded_student.agent_steps)),
        "optimizer_requested": bool(resume_optimizer),
        "optimizer_loaded": bool(resume_optimizer)
        and checkpoint.get("optimizer_state_dict") is not None,
        "student_model_type": runtime_cfg.get("student_model_type"),
        "student_obs_dim": runtime_cfg.get("student_obs_dim"),
        "student_action_dim": runtime_cfg.get("student_action_dim"),
    }


def _teacher_metadata(cfg: DictConfig, teacher_checkpoint: str | Path) -> dict[str, Any]:
    metadata = {
        "algo_family": str(cfg.teacher.algo_family),
        "algo_type": str(cfg.teacher.algo_type),
        "task": str(cfg.teacher.task),
        "task_name": str(cfg.teacher.task_name),
        "checkpoint_path": str(teacher_checkpoint),
    }
    info = validate_sac_teacher_checkpoint_contract(
        teacher_checkpoint,
        build_teacher_spec(cfg),
        device="cpu",
    )
    metadata.update(
        {
            "checkpoint_actor_input_dim": info.actor_input_dim,
            "checkpoint_first_weight_key": info.first_weight_key,
        }
    )
    return metadata


def _distill_runtime_cfg(
    cfg: DictConfig,
    *,
    distill_source: str,
    dataset_path: str | Path | None = None,
    student_init_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "distill_source": str(distill_source),
        "loss_type": str(cfg.algo.loss_type),
        "learning_rate": float(cfg.algo.learning_rate),
        "aux_loss_coef": float(OmegaConf.select(cfg, "algo.aux_loss_coef", default=0.0)),
        "role_loss_coef": float(OmegaConf.select(cfg, "algo.role_loss_coef", default=0.0)),
        "role_expert_targets": dict(
            OmegaConf.to_container(
                OmegaConf.select(cfg, "algo.role_expert_targets", default={}),
                resolve=True,
            )
        ),
        "command_intent_loss_coef": float(
            OmegaConf.select(cfg, "algo.command_intent_loss_coef", default=0.0)
        ),
        "command_intent_expert_targets": dict(
            OmegaConf.to_container(
                OmegaConf.select(cfg, "algo.command_intent_expert_targets", default={}),
                resolve=True,
            )
        ),
        "expert_behavior_loss_source": str(
            OmegaConf.select(cfg, "algo.expert_behavior_loss_source", default="auto")
        ),
        "offline_repeat_dataset": bool(
            OmegaConf.select(cfg, "training.offline_repeat_dataset", default=False)
        ),
        "offline_shuffle": bool(OmegaConf.select(cfg, "training.offline_shuffle", default=False)),
        "offline_balance_key": str(
            OmegaConf.select(cfg, "training.offline_balance_key", default="none")
        ),
        "offline_balanced_labels": list(
            OmegaConf.select(cfg, "training.offline_balanced_labels", default=[])
        ),
        "offline_balance_quotas": dict(
            OmegaConf.to_container(
                OmegaConf.select(cfg, "training.offline_balance_quotas", default={}),
                resolve=True,
            )
        ),
        "offline_min_balanced_replay_passes": int(
            OmegaConf.select(
                cfg,
                "training.offline_min_balanced_replay_passes",
                default=0,
            )
        ),
        "offline_min_balanced_replay_labels": list(
            OmegaConf.select(
                cfg,
                "training.offline_min_balanced_replay_labels",
                default=[],
            )
        ),
        **_student_runtime_cfg(cfg),
        "teacher_obs_dim": int(cfg.teacher.obs_dim),
    }
    if dataset_path is not None:
        payload["dataset_path"] = str(dataset_path)
    if student_init_metadata:
        payload["student_init_checkpoint_path"] = str(student_init_metadata["path"])
        payload["student_init_agent_steps"] = int(student_init_metadata["agent_steps"])
        payload["student_init_optimizer_requested"] = bool(
            student_init_metadata.get("optimizer_requested", False)
        )
        payload["student_init_optimizer_loaded"] = bool(student_init_metadata["optimizer_loaded"])
    return payload


def _probe_result(
    cfg: DictConfig,
    *,
    dataset: Any,
    result: Any,
    distill_source: str,
    dataset_path: str | Path | None = None,
    student_init_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    probe = {
        "distill_source": str(distill_source),
        "student_model_type": _student_model_type(cfg),
        "student_obs_shape": result.student_obs_shape,
        "teacher_obs_shape": result.teacher_obs_shape,
        "dataset_num_samples": dataset.num_samples,
        "dataset_student_obs_dim": dataset.student_obs_dim,
        "dataset_teacher_obs_dim": dataset.teacher_obs_dim,
        "dataset_metadata": dict(getattr(dataset, "metadata", {})),
        "student_action_shape": result.student_action_shape,
        "teacher_action_shape": result.teacher_action_shape,
        "teacher_action_requires_grad": result.teacher_action_requires_grad,
        "teacher_action_source": result.last_teacher_action_source,
        "student_grad_norm": result.last_student_grad_norm,
        "loss": result.last_loss,
        "behavior_loss": result.last_behavior_loss,
        "behavior_action_shape": result.last_behavior_action_shape,
        "behavior_action_source": result.last_behavior_action_source,
        "behavior_target_count": result.last_behavior_target_count,
        "aux_loss": result.last_aux_loss,
        "role_loss": result.last_role_loss,
        "role_target_count": result.last_role_target_count,
        "command_intent_loss": result.last_command_intent_loss,
        "command_intent_target_count": result.last_command_intent_target_count,
        "expert_usage": result.last_expert_usage,
        "route_entropy": result.last_route_entropy,
        "offline_balance_key": result.balance_key,
        "offline_batch_label_counts": result.batch_label_counts,
        "offline_last_balance_label_counts": result.last_balance_label_counts,
        "update_count": result.update_count,
        "samples_seen": result.samples_seen,
        "checkpoint_path": str(result.checkpoint_path) if result.checkpoint_path else None,
        "performance_stage_observations": [
            observation.as_dict() for observation in result.performance_stage_observations
        ],
    }
    if isinstance(student_init_metadata, dict):
        probe["student_init_checkpoint_path"] = student_init_metadata.get("path")
        probe["student_init_agent_steps"] = student_init_metadata.get("agent_steps")
        probe["student_init_optimizer_requested"] = student_init_metadata.get("optimizer_requested")
        probe["student_init_optimizer_loaded"] = student_init_metadata.get("optimizer_loaded")
    if dataset_path is not None:
        probe["dataset_path"] = str(dataset_path)
    return probe


def _normalize_checkpoint_selector(selector: Any) -> str | None:
    if selector in (None, "", -1, "-1"):
        return None
    return str(selector)


def resolve_teacher_checkpoint(
    cfg: DictConfig,
    *,
    root_dir: str | Path = ROOT_DIR,
) -> tuple[Path | None, Path | None]:
    """Resolve the teacher checkpoint through shared training path semantics."""

    explicit_checkpoint_path = OmegaConf.select(cfg, "teacher.checkpoint_path")
    if explicit_checkpoint_path not in (None, ""):
        path = Path(str(explicit_checkpoint_path))
        resolved_path = path if path.is_absolute() else Path(root_dir) / path
        if not resolved_path.is_file():
            raise FileNotFoundError(f"teacher.checkpoint_path does not exist: {resolved_path}")
        return resolved_path, resolved_path.parent

    log_root = OmegaConf.select(
        cfg,
        "teacher.log_root",
        default=OmegaConf.select(cfg, "training.log_root"),
    )
    return resolve_task_checkpoint_path(
        root_dir,
        task_name=str(cfg.teacher.task_name),
        load_run=str(cfg.teacher.load_run),
        algo_log_name=str(cfg.teacher.algo_log_name),
        checkpoint=_normalize_checkpoint_selector(cfg.teacher.checkpoint),
        suffix=".pt",
        log_root=log_root,
    )


def build_distillation_trainer(
    cfg: DictConfig,
    *,
    teacher_checkpoint: str | Path,
    student_init_checkpoint: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> BehaviorDistillationTrainer:
    """Load teacher, build student, and assemble one behavior distillation trainer."""

    validate_sac_teacher_checkpoint_contract(
        teacher_checkpoint,
        build_teacher_spec(cfg),
        device=device,
    )
    teacher = load_sac_teacher_policy(
        teacher_checkpoint,
        build_teacher_spec(cfg),
        device=device,
    )
    student = build_student_policy(cfg, device=device)
    optimizer = torch.optim.Adam(student.parameters(), lr=float(cfg.algo.learning_rate))
    student_init_metadata: dict[str, Any] = {}
    resolved_student_init_checkpoint = _resolve_optional_checkpoint_path(
        student_init_checkpoint,
        field_name="training.offline_init_checkpoint",
    )
    if resolved_student_init_checkpoint is not None:
        student_init_metadata = _load_student_init_checkpoint(
            student,
            optimizer,
            resolved_student_init_checkpoint,
            cfg=cfg,
            device=device,
            resume_optimizer=bool(
                OmegaConf.select(cfg, "training.offline_resume_optimizer", default=True)
            ),
        )
    return BehaviorDistillationTrainer(
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        loss_type=str(cfg.algo.loss_type),
        max_grad_norm=float(cfg.algo.max_grad_norm),
        aux_loss_coef=float(OmegaConf.select(cfg, "algo.aux_loss_coef", default=0.0)),
        role_loss_coef=float(OmegaConf.select(cfg, "algo.role_loss_coef", default=0.0)),
        role_expert_targets=dict(
            OmegaConf.to_container(
                OmegaConf.select(cfg, "algo.role_expert_targets", default={}),
                resolve=True,
            )
        ),
        command_intent_loss_coef=float(
            OmegaConf.select(cfg, "algo.command_intent_loss_coef", default=0.0)
        ),
        command_intent_expert_targets=dict(
            OmegaConf.to_container(
                OmegaConf.select(cfg, "algo.command_intent_expert_targets", default={}),
                resolve=True,
            )
        ),
        student_init_metadata=student_init_metadata,
        expert_behavior_loss_source=str(
            OmegaConf.select(cfg, "algo.expert_behavior_loss_source", default="auto")
        ),
    )


def run_fake_batch_update(
    cfg: DictConfig,
    *,
    teacher_checkpoint: str | Path,
    batch_size: int = 8,
    max_updates: int = 1,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Run a bounded shape-valid offline distillation probe for the entrypoint."""

    torch.manual_seed(int(cfg.algo.seed))
    trainer = build_distillation_trainer(
        cfg,
        teacher_checkpoint=teacher_checkpoint,
        device=device,
    )
    dataset = make_fake_distillation_dataset(
        num_samples=int(batch_size) * int(max_updates),
        student_obs_dim=int(cfg.student.obs_dim),
        teacher_obs_dim=int(cfg.teacher.obs_dim),
        seed=int(cfg.algo.seed),
        device=device,
    )
    result = run_offline_distillation_updates(
        trainer,
        dataset,
        batch_size=int(batch_size),
        max_updates=int(max_updates),
        checkpoint_path=checkpoint_path,
        teacher_metadata=_teacher_metadata(cfg, teacher_checkpoint),
        distill_runtime_cfg=_distill_runtime_cfg(cfg, distill_source="fake_probe"),
    )

    return _probe_result(cfg, dataset=dataset, result=result, distill_source="fake_probe")


def run_offline_dataset_update(
    cfg: DictConfig,
    *,
    teacher_checkpoint: str | Path,
    dataset_path: str | Path,
    batch_size: int | None = None,
    max_updates: int | None = None,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    auto_expand_replay_budget: bool = False,
    progress_callback: Callable[[int, int, Any], None] | None = None,
    performance_clock: Callable[[], float] | None = None,
) -> dict[str, Any]:
    """Run bounded offline updates from a saved distillation tensor dataset."""

    torch.manual_seed(int(cfg.algo.seed))
    resolved_batch_size = int(
        batch_size
        if batch_size is not None
        else OmegaConf.select(cfg, "training.offline_batch_size", default=256)
    )
    resolved_max_updates = int(
        max_updates
        if max_updates is not None
        else OmegaConf.select(cfg, "training.offline_max_updates", default=1)
    )
    trainer = build_distillation_trainer(
        cfg,
        teacher_checkpoint=teacher_checkpoint,
        student_init_checkpoint=OmegaConf.select(
            cfg,
            "training.offline_init_checkpoint",
            default=None,
        ),
        device=device,
    )
    student_init_metadata = dict(getattr(trainer, "student_init_metadata", {}))
    dataset = load_distillation_dataset(
        dataset_path,
        expected_student_obs_dim=int(cfg.student.obs_dim),
        expected_teacher_obs_dim=int(cfg.teacher.obs_dim),
        expected_teacher_action_dim=int(cfg.teacher.action_dim),
        device=device,
    )
    offline_balance_key = str(OmegaConf.select(cfg, "training.offline_balance_key", default="none"))
    offline_balanced_labels = list(
        OmegaConf.select(cfg, "training.offline_balanced_labels", default=[])
    )
    offline_balance_quotas = dict(
        OmegaConf.to_container(
            OmegaConf.select(cfg, "training.offline_balance_quotas", default={}),
            resolve=True,
        )
    )
    offline_replay_passes = int(
        OmegaConf.select(cfg, "training.offline_min_balanced_replay_passes", default=0)
    )
    offline_replay_labels = list(
        OmegaConf.select(cfg, "training.offline_min_balanced_replay_labels", default=[])
    )
    if auto_expand_replay_budget:
        resolved_max_updates = max(
            resolved_max_updates,
            required_balanced_replay_updates(
                dataset,
                balance_key=offline_balance_key,
                batch_size=resolved_batch_size,
                balanced_labels=offline_balanced_labels,
                balance_quotas=offline_balance_quotas,
                replay_labels=offline_replay_labels,
                replay_passes=offline_replay_passes,
            ),
        )
    progress_enabled = os.environ.get("UNILAB_DISTILL_PROGRESS", "0").lower() not in {
        "",
        "0",
        "false",
        "off",
    }
    progress_interval = int(os.environ.get("UNILAB_DISTILL_PROGRESS_INTERVAL", "0") or 0)
    if progress_callback is not None and progress_interval <= 0:
        progress_interval = max(1, resolved_max_updates // 20)
    if progress_enabled:
        print(
            "[distill-progress] "
            f"dataset={dataset_path} samples={dataset.num_samples} "
            f"updates={resolved_max_updates} batch_size={resolved_batch_size}",
            flush=True,
        )
    result = run_offline_distillation_updates(
        trainer,
        dataset,
        batch_size=resolved_batch_size,
        max_updates=resolved_max_updates,
        checkpoint_path=checkpoint_path,
        teacher_metadata=_teacher_metadata(cfg, teacher_checkpoint),
        distill_runtime_cfg=_distill_runtime_cfg(
            cfg,
            distill_source="offline_dataset",
            dataset_path=dataset_path,
            student_init_metadata=student_init_metadata,
        ),
        repeat_dataset=bool(
            OmegaConf.select(cfg, "training.offline_repeat_dataset", default=False)
        ),
        shuffle=bool(OmegaConf.select(cfg, "training.offline_shuffle", default=False)),
        seed=int(cfg.algo.seed),
        balance_key=offline_balance_key,
        balanced_labels=offline_balanced_labels,
        balance_quotas=offline_balance_quotas,
        min_balanced_replay_passes=offline_replay_passes,
        min_balanced_replay_labels=offline_replay_labels,
        save_optimizer_state=bool(
            OmegaConf.select(cfg, "training.offline_save_optimizer", default=True)
        ),
        progress_interval=(
            progress_interval if progress_enabled or progress_callback is not None else 0
        ),
        progress_callback=progress_callback,
        performance_clock=performance_clock,
    )
    return _probe_result(
        cfg,
        dataset=dataset,
        result=result,
        distill_source="offline_dataset",
        dataset_path=dataset_path,
        student_init_metadata=student_init_metadata,
    )


def _multitask_sources(cfg: DictConfig) -> list[dict[str, Any]]:
    sources = OmegaConf.to_container(
        OmegaConf.select(cfg, "training.multitask_sources", default=[]),
        resolve=True,
    )
    if not isinstance(sources, list):
        raise ValueError("training.multitask_sources must be a list")
    return [dict(cast(dict[str, Any], source)) for source in sources]


def _optional_int_cfg(cfg: DictConfig, path: str) -> int | None:
    value = OmegaConf.select(cfg, path)
    if value in (None, ""):
        return None
    return int(value)


def run_multitask_dataset_assembly(
    cfg: DictConfig,
    *,
    dataset_path: str | Path | None = None,
) -> dict[str, Any]:
    """Merge saved role-specific datasets into one CPU-owned cached dataset."""

    resolved_dataset_path = dataset_path or OmegaConf.select(
        cfg,
        "training.multitask_dataset_path",
    )
    if resolved_dataset_path in (None, ""):
        raise ValueError("training.multitask_dataset_path must be set")
    dataset = build_multitask_distillation_dataset(
        _multitask_sources(cfg),
        expected_student_obs_dim=_optional_int_cfg(
            cfg,
            "training.multitask_expected_student_obs_dim",
        ),
        expected_teacher_obs_dim=_optional_int_cfg(
            cfg,
            "training.multitask_expected_teacher_obs_dim",
        ),
        expected_teacher_action_dim=_optional_int_cfg(
            cfg,
            "training.multitask_expected_teacher_action_dim",
        ),
        device=_ROLE_DATA_ASSEMBLY_DEVICE,
    )
    save_distillation_dataset(resolved_dataset_path, dataset)
    return {
        "distill_source": "multitask_adapter",
        "dataset_path": str(resolved_dataset_path),
        "aggregate_assembly_device": _ROLE_DATA_ASSEMBLY_DEVICE,
        "dataset_num_samples": dataset.num_samples,
        "dataset_student_obs_dim": dataset.student_obs_dim,
        "dataset_teacher_obs_dim": dataset.teacher_obs_dim,
        "dataset_teacher_action_dim": dataset.teacher_action_dim,
        "dataset_metadata": dict(dataset.metadata),
        "student_obs_shape": tuple(dataset.student_obs.shape),
        "teacher_obs_shape": tuple(dataset.teacher_obs.shape),
        "teacher_actions_shape": (
            None if dataset.teacher_actions is None else tuple(dataset.teacher_actions.shape)
        ),
        "source_roles": list(dataset.metadata["source_roles"]),
        "source_sample_counts": list(dataset.metadata["source_sample_counts"]),
    }


def _resolve_formal_run_dir(
    cfg: DictConfig,
    *,
    root_dir: str | Path = ROOT_DIR,
) -> Path:
    explicit = OmegaConf.select(cfg, "training.formal_run_dir")
    if explicit not in (None, ""):
        path = Path(str(explicit))
        return path if path.is_absolute() else Path(root_dir) / path

    log_dir = OmegaConf.select(cfg, "training.log_dir")
    if log_dir not in (None, ""):
        path = Path(str(log_dir))
        return path if path.is_absolute() else Path(root_dir) / path

    log_root = OmegaConf.select(cfg, "training.log_root")
    root = Path(str(log_root)) if log_root not in (None, "") else Path(root_dir) / "logs"
    if not root.is_absolute():
        root = Path(root_dir) / root
    run_name = OmegaConf.select(cfg, "training.formal_run_name")
    if run_name in (None, ""):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{timestamp}_{OmegaConf.select(cfg, 'training.sim_backend', default='mujoco')}"
    return root / str(cfg.algo.algo_log_name) / str(cfg.training.task_name) / str(run_name)


def _expected_samples_seen_for_offline_run(
    cfg: DictConfig,
    *,
    dataset_path: str | Path,
    batch_size: int,
    max_updates: int,
    device: str | torch.device = "cpu",
) -> int:
    dataset = load_distillation_dataset(
        dataset_path,
        expected_student_obs_dim=int(cfg.student.obs_dim),
        expected_teacher_obs_dim=int(cfg.teacher.obs_dim),
        expected_teacher_action_dim=int(cfg.teacher.action_dim),
        device=device,
    )
    if str(OmegaConf.select(cfg, "training.offline_balance_key", default="none")) != "none":
        return int(batch_size) * int(max_updates)
    if bool(OmegaConf.select(cfg, "training.offline_repeat_dataset", default=False)):
        samples_seen = 0
        cursor = 0
        for _ in range(int(max_updates)):
            if cursor >= dataset.num_samples:
                cursor = 0
            end = min(dataset.num_samples, cursor + int(batch_size))
            samples_seen += end - cursor
            cursor = end
        return samples_seen
    return min(int(dataset.num_samples), int(batch_size) * int(max_updates))


def run_formal_offline_dataset_update(
    cfg: DictConfig,
    *,
    teacher_checkpoint: str | Path,
    dataset_path: str | Path | None = None,
    run_dir: str | Path | None = None,
    batch_size: int | None = None,
    max_updates: int | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Run the saved-dataset distillation path with run metadata and checkpoint layout."""

    resolved_dataset_path = dataset_path or OmegaConf.select(cfg, "training.offline_dataset_path")
    if resolved_dataset_path in (None, ""):
        raise ValueError("training.offline_dataset_path must be set for formal distill runs")
    resolved_batch_size = int(
        batch_size
        if batch_size is not None
        else OmegaConf.select(cfg, "training.offline_batch_size", default=256)
    )
    resolved_max_updates = int(
        max_updates
        if max_updates is not None
        else OmegaConf.select(cfg, "training.offline_max_updates", default=1)
    )
    resolved_run_dir = (
        Path(run_dir) if run_dir is not None else _resolve_formal_run_dir(cfg, root_dir=ROOT_DIR)
    )
    samples_seen = _expected_samples_seen_for_offline_run(
        cfg,
        dataset_path=resolved_dataset_path,
        batch_size=resolved_batch_size,
        max_updates=resolved_max_updates,
        device=device,
    )
    checkpoint_path = resolved_run_dir / f"model_{samples_seen}.pt"

    tracker = ExperimentTracker(
        root_dir=ROOT_DIR,
        log_dir=resolved_run_dir,
        algo_name=str(cfg.algo.algo_log_name),
        task_name=str(cfg.training.task_name),
        sim_backend=str(OmegaConf.select(cfg, "training.sim_backend", default="mujoco")),
        training_cfg=cfg.training,
        full_cfg=cfg,
        device=str(device),
    )
    tracker.start()
    try:
        probe = run_offline_dataset_update(
            cfg,
            teacher_checkpoint=teacher_checkpoint,
            dataset_path=resolved_dataset_path,
            batch_size=resolved_batch_size,
            max_updates=resolved_max_updates,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        tracker.update_summary(
            {
                "status": "completed",
                "distill_source": "formal_offline_dataset",
                "dataset_path": str(resolved_dataset_path),
                "checkpoint_path": str(checkpoint_path),
                "update_count": int(probe["update_count"]),
                "samples_seen": int(probe["samples_seen"]),
                "loss": float(probe["loss"]),
                "behavior_loss": float(probe["behavior_loss"]),
                "aux_loss": float(probe["aux_loss"]),
                "student_grad_norm": float(probe["student_grad_norm"]),
            }
        )
    finally:
        tracker.finish()

    probe.update(
        {
            "distill_source": "formal_offline_dataset",
            "run_dir": str(resolved_run_dir),
            "run_config_path": str(resolved_run_dir / "run_config.json"),
            "run_summary_path": str(resolved_run_dir / "run_summary.json"),
            "checkpoint_path": str(checkpoint_path),
        }
    )
    return probe
