"""Paired FADA policy and optimizer checkpoint owner."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from unilab.algos.torch.distill.fada.async_config import validate_fada_training_schedule
from unilab.algos.torch.distill.fada.model import FADAPlannerIDMPolicy
from unilab.algos.torch.distill.fada.source_artifact import load_architecture_config
from unilab.algos.torch.distill.fada.trainer import FADATrainer
from unilab.algos.torch.distill.fada.official_oracle import validate_student_binding

FADA_CHECKPOINT_SCHEMA_VERSION = 5
FADA_TRAINING_SCHEDULE = "alternating_idm_then_planner"


FADA_V005_REQUIRED_QUALITY_METRICS = (
    "scenario/walk/planner_idm_oracle_action_mse",
    "scenario/walk/cold_start_fraction",
    "scenario/walk/cold_start_planner_mse",
    "scenario/walk/steady_state_planner_mse",
    "scenario/static_stand/planner_idm_oracle_action_mse",
    "scenario/walk_to_stand/planner_idm_oracle_action_mse",
    "scenario/static_stand/cold_start_fraction",
    "scenario/static_stand/cold_start_planner_mse",
    "scenario/static_stand/steady_state_planner_mse",
)


@dataclass(frozen=True)
class LoadedFADAPlannerIDMPolicy:
    """Inference-ready FADA policy reconstructed from one strict checkpoint."""

    policy: FADAPlannerIDMPolicy
    checkpoint: Mapping[str, Any]


def _canonical_state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().to(device="cpu").contiguous()
        identity = f"{name}\0{tensor.dtype}\0{tuple(tensor.shape)}\0".encode("ascii")
        digest.update(len(identity).to_bytes(8, "big"))
        digest.update(identity)
        raw = tensor.view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _validated_idm_state(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    state = payload.get("idm_state_dict")
    if not isinstance(state, dict) or payload.get("idm_sha256") != _canonical_state_dict_sha256(
        state
    ):
        raise ValueError("FADA checkpoint IDM identity mismatch")
    return state


def _validate_schema5_training_state(payload: Mapping[str, Any]) -> None:
    validate_fada_training_schedule(payload.get("training_schedule"))
    required = ("idm_optimizer_state_dict", "planner_optimizer_state_dict")
    if any(not isinstance(payload.get(name), dict) for name in required):
        raise ValueError("schema-5 FADA checkpoint requires both optimizer states")


def save_fada_checkpoint(
    path: str | Path,
    policy: FADAPlannerIDMPolicy,
    trainer: FADATrainer,
    *,
    completed_iterations: int,
    samples_seen: int,
    runtime_config: Mapping[str, Any],
    quality_metrics: Mapping[str, float] | None = None,
) -> Path:
    """Atomically persist the paired FADA module and optimizer identity."""

    validate_student_binding(policy.config, runtime_config)
    metrics = dict(quality_metrics or {})
    v005_enabled = bool(dict(runtime_config.get("v005_replay") or {}).get("enabled", False))
    if v005_enabled and int(completed_iterations) > 0:
        missing = [name for name in FADA_V005_REQUIRED_QUALITY_METRICS if name not in metrics]
        if missing:
            raise ValueError(f"v005 FADA checkpoint quality metrics are missing: {missing}")
        invalid = [
            name
            for name, value in metrics.items()
            if not isinstance(value, (int, float)) or not math.isfinite(float(value))
        ]
        if invalid:
            raise ValueError(f"v005 FADA checkpoint quality metrics are non-finite: {invalid}")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    idm_sha256 = _canonical_state_dict_sha256(policy.idm.state_dict())
    training_schedule = validate_fada_training_schedule(
        runtime_config.get("training_schedule", FADA_TRAINING_SCHEDULE)
    )
    payload = {
        "schema_version": FADA_CHECKPOINT_SCHEMA_VERSION,
        "architecture": asdict(policy.config),
        "planner_state_dict": policy.planner.state_dict(),
        "idm_state_dict": policy.idm.state_dict(),
        "training_schedule": training_schedule,
        "planner_optimizer_state_dict": trainer.planner_optimizer.state_dict(),
        "idm_optimizer_state_dict": trainer.idm_optimizer.state_dict(),
        "idm_sha256": idm_sha256,
        "completed_iterations": int(completed_iterations),
        "samples_seen": int(samples_seen),
        "runtime_config": dict(runtime_config),
        "quality_metrics": metrics,
    }
    torch.save(payload, temporary)
    temporary.replace(target)
    return target


def load_fada_checkpoint(
    path: str | Path,
    policy: FADAPlannerIDMPolicy,
    trainer: FADATrainer | None = None,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Restore one exact FADA architecture and optionally its paired optimizers."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") not in {1, 2, 3, 4, 5}:
        raise ValueError("unsupported or malformed FADA checkpoint schema")
    if trainer is not None:
        raise ValueError(
            "v011 training resume is disabled; load without a trainer for state inspection"
        )
    observed = load_architecture_config(
        payload.get("architecture"),
        schema_version=payload.get("schema_version"),
        contract_schema_version=FADA_CHECKPOINT_SCHEMA_VERSION,
        context="FADA checkpoint",
    )
    if observed != policy.config:
        raise ValueError(
            f"FADA checkpoint architecture mismatch: expected={policy.config} observed={observed}"
        )
    validate_student_binding(observed, payload.get("runtime_config"))
    if payload.get("schema_version") in {4, 5}:
        _validated_idm_state(payload)
    if payload.get("schema_version") == 5:
        _validate_schema5_training_state(payload)
    policy.planner.load_state_dict(payload["planner_state_dict"], strict=True)
    policy.idm.load_state_dict(payload["idm_state_dict"], strict=True)
    return payload


def load_fada_policy_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> LoadedFADAPlannerIDMPolicy:
    """Construct an inference-only Planner-IDM policy from checkpoint-owned architecture."""

    # B1: 只读取 tensor 与基础类型, 并在模型构造前关闭 schema/architecture 不匹配.
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") not in {1, 2, 3, 4, 5}:
        raise ValueError("unsupported or malformed FADA checkpoint schema")
    config = load_architecture_config(
        payload.get("architecture"),
        schema_version=payload.get("schema_version"),
        contract_schema_version=FADA_CHECKPOINT_SCHEMA_VERSION,
        context="FADA checkpoint",
    )
    validate_student_binding(config, payload.get("runtime_config"))
    if payload.get("schema_version") in {4, 5}:
        _validated_idm_state(payload)
    if payload.get("schema_version") == 5:
        _validate_schema5_training_state(payload)

    # B2: 严格恢复 Planner 与 IDM, 产出 eval 模式的唯一 playback policy owner.
    policy = FADAPlannerIDMPolicy(config).to(device)
    policy.planner.load_state_dict(payload["planner_state_dict"], strict=True)
    policy.idm.load_state_dict(payload["idm_state_dict"], strict=True)
    policy.eval()
    return LoadedFADAPlannerIDMPolicy(policy=policy, checkpoint=payload)
