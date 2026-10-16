from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from unilab.algos.torch.distill.fada.adaptation import (
    FADALoRAConfig,
    _inject_fada_idm_legacy_linear_lora,
    assert_fada_adaptation_parameter_ownership,
    fada_adapter_named_parameters,
    inject_fada_idm_lora,
)
from unilab.algos.torch.distill.fada.checkpoint import (
    FADA_CHECKPOINT_SCHEMA_VERSION,
    LoadedFADAPlannerIDMPolicy,
    load_fada_policy_checkpoint,
)
from unilab.algos.torch.distill.fada.model import FADAArchitectureConfig, FADAPlannerIDMPolicy
from unilab.algos.torch.distill.fada.privileged_oracle import (
    FADA_ORACLE_BEHAVIOR_PROFILES,
    FADA_ORACLE_PHASE_NEUTRAL_PROFILE,
)
from unilab.algos.torch.distill.fada.target_data import (
    FADA_ACTUATOR_TARGET_ARTIFACT_SCHEMA_VERSION,
    FADA_REAL_TARGET_ARTIFACT_SCHEMA_VERSION,
    FADA_TARGET_ARTIFACT_SCHEMA_VERSION,
)

FADA_ADAPTED_CHECKPOINT_SCHEMA_VERSION = "fada-adapted/v3"
FADA_LEGACY_ADAPTED_CHECKPOINT_SCHEMA_VERSION = "fada-adapted/v1"
FADA_LEGACY_LINEAR_ADAPTED_CHECKPOINT_SCHEMA_VERSION = "fada-adapted/v2"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def assert_fada_adaptation_source_checkpoint(
    loaded: LoadedFADAPlannerIDMPolicy,
    *,
    expected_behavior_profile: str | None = None,
) -> LoadedFADAPlannerIDMPolicy:
    """Keep Stage-C/D pinned to the current complete FADA source schema."""

    checkpoint = getattr(loaded, "checkpoint", None)
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("schema_version") != FADA_CHECKPOINT_SCHEMA_VERSION
    ):
        raise ValueError(
            "FADA adaptation requires current schema-"
            f"{FADA_CHECKPOINT_SCHEMA_VERSION} source checkpoint"
        )
    _assert_checkpoint_behavior_profile(
        checkpoint,
        expected_behavior_profile=expected_behavior_profile,
    )
    return loaded


def assert_fada_target_collection_checkpoint(
    loaded: LoadedFADAPlannerIDMPolicy,
    *,
    expected_behavior_profile: str | None = None,
) -> LoadedFADAPlannerIDMPolicy:
    """Admit current source and adapted policies to post-training collection."""

    checkpoint = getattr(loaded, "checkpoint", None)
    schema = checkpoint.get("schema_version") if isinstance(checkpoint, Mapping) else None
    if schema not in {
        FADA_CHECKPOINT_SCHEMA_VERSION,
        FADA_ADAPTED_CHECKPOINT_SCHEMA_VERSION,
    }:
        raise ValueError(
            "FADA target collection requires a schema-5 source or fada-adapted/v3 checkpoint"
        )
    assert isinstance(checkpoint, Mapping)
    _assert_checkpoint_behavior_profile(
        checkpoint,
        expected_behavior_profile=expected_behavior_profile,
    )
    return loaded


def fada_checkpoint_behavior_profile(checkpoint: Mapping[str, Any]) -> str:
    """Read source behavior identity; pre-v023 artifacts are phase-neutral."""

    runtime_config = checkpoint.get("runtime_config")
    if not isinstance(runtime_config, Mapping):
        return FADA_ORACLE_PHASE_NEUTRAL_PROFILE
    profile = runtime_config.get(
        "source_behavior_profile", FADA_ORACLE_PHASE_NEUTRAL_PROFILE
    )
    if profile not in FADA_ORACLE_BEHAVIOR_PROFILES:
        raise ValueError(f"unsupported FADA source behavior profile: {profile!r}")
    return str(profile)


def _assert_checkpoint_behavior_profile(
    checkpoint: Mapping[str, Any],
    *,
    expected_behavior_profile: str | None,
) -> None:
    observed = fada_checkpoint_behavior_profile(checkpoint)
    if expected_behavior_profile is not None and observed != expected_behavior_profile:
        raise ValueError(
            "FADA checkpoint behavior profile mismatch: "
            f"expected={expected_behavior_profile!r} observed={observed!r}"
        )


def _validate_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _lora_payload(config: FADALoRAConfig) -> dict[str, Any]:
    return {
        "rank": int(config.rank),
        "alpha": float(config.alpha),
        "dropout": float(config.dropout),
        "target_modules": list(config.target_modules),
        "adapter_type": config.adapter_type,
        "target_projections": list(config.target_projections),
    }


def save_fada_adapted_checkpoint(
    path: str | Path,
    policy: FADAPlannerIDMPolicy,
    optimizer: torch.optim.Optimizer,
    *,
    lora_config: FADALoRAConfig,
    source_checkpoint_sha256: str,
    target_artifact_sha256: str,
    target_artifact_schema_version: str,
    completed_steps: int,
    samples_seen: int,
    runtime_config: Mapping[str, Any],
) -> Path:
    """Atomically persist a self-contained frozen-base plus adapter checkpoint."""

    assert_fada_adaptation_parameter_ownership(policy, lora_config)
    if lora_config.adapter_type != "qv_attention":
        raise ValueError("new adapted checkpoints require Q/V attention LoRA")
    if (
        isinstance(completed_steps, bool)
        or not isinstance(completed_steps, int)
        or completed_steps < 0
    ):
        raise ValueError("completed_steps must be a non-negative integer")
    if isinstance(samples_seen, bool) or not isinstance(samples_seen, int) or samples_seen < 0:
        raise ValueError("samples_seen must be a non-negative integer")
    adapter_ids = {id(parameter) for _name, parameter in fada_adapter_named_parameters(policy)}
    optimizer_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    optimizer_ids = [id(parameter) for parameter in optimizer_parameters]
    if len(optimizer_ids) != len(set(optimizer_ids)) or set(optimizer_ids) != adapter_ids:
        raise ValueError(
            "adapted checkpoint optimizer must own only adapter parameters exactly once"
        )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    payload = {
        "schema_version": FADA_ADAPTED_CHECKPOINT_SCHEMA_VERSION,
        "architecture": asdict(policy.config),
        "lora_config": _lora_payload(lora_config),
        "policy_state_dict": policy.state_dict(),
        "trainable_parameter_names": list(dict(fada_adapter_named_parameters(policy))),
        "optimizer_state_dict": optimizer.state_dict(),
        "source_checkpoint_sha256": _validate_sha256(
            "source_checkpoint_sha256", source_checkpoint_sha256
        ),
        "target_artifact_sha256": _validate_sha256(
            "target_artifact_sha256", target_artifact_sha256
        ),
        "target_artifact_schema_version": target_artifact_schema_version,
        "completed_steps": completed_steps,
        "samples_seen": samples_seen,
        "runtime_config": dict(runtime_config),
    }
    target_domain = runtime_config.get("target_domain")
    if isinstance(target_domain, Mapping):
        target_domain_id = target_domain.get("target_domain_id")
        if not isinstance(target_domain_id, str) or not target_domain_id.strip():
            raise ValueError("adapted checkpoint target_domain_id must be a non-empty string")
        payload["target_domain_id"] = target_domain_id
    if target_artifact_schema_version not in {
        FADA_ACTUATOR_TARGET_ARTIFACT_SCHEMA_VERSION,
        FADA_TARGET_ARTIFACT_SCHEMA_VERSION,
        FADA_REAL_TARGET_ARTIFACT_SCHEMA_VERSION,
    }:
        raise ValueError("adapted checkpoint target artifact schema is unsupported")
    torch.save(payload, temporary)
    temporary.replace(target)
    return target


def _parse_lora_config(payload: Any, *, schema_version: str) -> FADALoRAConfig:
    common = {"rank", "alpha", "dropout", "target_modules"}
    current = common | {"adapter_type", "target_projections"}
    expected = current if schema_version == FADA_ADAPTED_CHECKPOINT_SCHEMA_VERSION else common
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("adapted checkpoint LoRA config or manifest is malformed")
    targets = payload.get("target_modules")
    if not isinstance(targets, list) or not all(isinstance(name, str) for name in targets):
        raise ValueError("adapted checkpoint LoRA target manifest is malformed")
    kwargs = {
        "rank": payload["rank"],
        "alpha": payload["alpha"],
        "dropout": payload["dropout"],
        "target_modules": tuple(targets),
    }
    if schema_version == FADA_ADAPTED_CHECKPOINT_SCHEMA_VERSION:
        projections = payload.get("target_projections")
        adapter_type = payload.get("adapter_type")
        if not isinstance(projections, list) or not all(
            isinstance(name, str) for name in projections
        ):
            raise ValueError("adapted checkpoint LoRA projection manifest is malformed")
        if not isinstance(adapter_type, str):
            raise ValueError("adapted checkpoint LoRA adapter type is malformed")
        return FADALoRAConfig(
            **kwargs,
            adapter_type=adapter_type,
            target_projections=tuple(projections),
        )
    return FADALoRAConfig.legacy(**kwargs)


def load_fada_adapted_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> LoadedFADAPlannerIDMPolicy:
    """Fresh-construct and strict-load one self-contained adapted policy."""

    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") not in {
        FADA_LEGACY_ADAPTED_CHECKPOINT_SCHEMA_VERSION,
        FADA_LEGACY_LINEAR_ADAPTED_CHECKPOINT_SCHEMA_VERSION,
        FADA_ADAPTED_CHECKPOINT_SCHEMA_VERSION,
    }:
        raise ValueError("unsupported or malformed FADA adapted checkpoint schema")
    architecture = payload.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError("adapted checkpoint architecture must be a mapping")
    if (
        payload.get("schema_version")
        in {
            FADA_LEGACY_LINEAR_ADAPTED_CHECKPOINT_SCHEMA_VERSION,
            FADA_ADAPTED_CHECKPOINT_SCHEMA_VERSION,
        }
        and "observation_contract" not in architecture
    ):
        raise ValueError("adapted checkpoint architecture must contain observation_contract")
    try:
        config = FADAArchitectureConfig(**architecture)
    except (TypeError, ValueError) as exc:
        raise ValueError("adapted checkpoint architecture is invalid") from exc
    schema_version = str(payload["schema_version"])
    lora_config = _parse_lora_config(payload.get("lora_config"), schema_version=schema_version)
    policy = FADAPlannerIDMPolicy(config).to(device)
    adapted = (
        inject_fada_idm_lora(policy, lora_config)
        if schema_version == FADA_ADAPTED_CHECKPOINT_SCHEMA_VERSION
        else _inject_fada_idm_legacy_linear_lora(policy, lora_config)
    )
    expected_trainable = list(dict(fada_adapter_named_parameters(adapted.policy)))
    if payload.get("trainable_parameter_names") != expected_trainable:
        raise ValueError("adapted checkpoint trainable parameter manifest is incompatible")
    state = payload.get("policy_state_dict")
    if not isinstance(state, dict):
        raise ValueError("adapted checkpoint policy state must be a mapping")
    try:
        adapted.policy.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError("adapted checkpoint architecture or policy state is incompatible") from exc
    _validate_sha256("source_checkpoint_sha256", payload.get("source_checkpoint_sha256"))
    _validate_sha256("target_artifact_sha256", payload.get("target_artifact_sha256"))
    if schema_version == FADA_ADAPTED_CHECKPOINT_SCHEMA_VERSION:
        target_schema = payload.get("target_artifact_schema_version")
        if target_schema not in {
            FADA_ACTUATOR_TARGET_ARTIFACT_SCHEMA_VERSION,
            FADA_TARGET_ARTIFACT_SCHEMA_VERSION,
            FADA_REAL_TARGET_ARTIFACT_SCHEMA_VERSION,
        }:
            raise ValueError("adapted checkpoint target artifact schema is unsupported")
        runtime_target = (
            payload["runtime_config"].get("target_domain")
            if isinstance(payload.get("runtime_config"), dict)
            else None
        )
        if isinstance(runtime_target, Mapping):
            if payload.get("target_domain_id") != runtime_target.get("target_domain_id"):
                raise ValueError("adapted checkpoint target-domain identity is inconsistent")
    for name in ("completed_steps", "samples_seen"):
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"adapted checkpoint {name} must be a non-negative integer")
    if not isinstance(payload.get("optimizer_state_dict"), dict):
        raise ValueError("adapted checkpoint optimizer state must be a mapping")
    if not isinstance(payload.get("runtime_config"), dict):
        raise ValueError("adapted checkpoint runtime_config must be a mapping")
    assert_fada_adaptation_parameter_ownership(adapted.policy, adapted.lora_config)
    adapted.policy.eval()
    return LoadedFADAPlannerIDMPolicy(policy=adapted.policy, checkpoint=payload)


def load_fada_deployable_policy_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> LoadedFADAPlannerIDMPolicy:
    """Dispatch supported source and adapted schemas for the official playback consumer."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("unsupported or malformed FADA deployable checkpoint schema")
    schema = payload.get("schema_version")
    if schema in {1, 2, 3, FADA_CHECKPOINT_SCHEMA_VERSION}:
        return load_fada_policy_checkpoint(path, device=device)
    if schema in {
        FADA_LEGACY_ADAPTED_CHECKPOINT_SCHEMA_VERSION,
        FADA_LEGACY_LINEAR_ADAPTED_CHECKPOINT_SCHEMA_VERSION,
        FADA_ADAPTED_CHECKPOINT_SCHEMA_VERSION,
    }:
        return load_fada_adapted_checkpoint(path, device=device)
    raise ValueError(f"unsupported FADA deployable checkpoint schema: {schema!r}")
