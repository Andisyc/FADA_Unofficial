from __future__ import annotations

import importlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import torch

from unilab.algos.torch.distill import (
    FADAArchitectureConfig,
    FADAPlannerIDMPolicy,
    load_fada_policy_checkpoint,
)
from unilab.algos.torch.distill.fada_adaptation_checkpoint import (
    assert_fada_adaptation_source_checkpoint,
)


@pytest.mark.parametrize("schema_version", [1, 2, 3, 4, "fada-adapted/v1", "fada-adapted/v2"])
def test_adaptation_source_contract_rejects_retired_source_schema(
    schema_version: int,
) -> None:
    with pytest.raises(ValueError, match="requires current schema-5"):
        assert_fada_adaptation_source_checkpoint(
            type("Loaded", (), {"checkpoint": {"schema_version": schema_version}})()
        )


def test_adaptation_source_contract_accepts_current_schema5() -> None:
    loaded = type("Loaded", (), {"checkpoint": {"schema_version": 5}})()
    assert assert_fada_adaptation_source_checkpoint(loaded) is loaded


@pytest.mark.parametrize("schema_version", [1, 2, 3, 4])
def test_target_collection_contract_rejects_retired_source_schema(
    schema_version: int,
) -> None:
    _adaptation, checkpoint_owner = _owners()
    assert hasattr(checkpoint_owner, "assert_fada_target_collection_checkpoint")
    loaded = type("Loaded", (), {"checkpoint": {"schema_version": schema_version}})()
    with pytest.raises(ValueError, match="schema-5 source or fada-adapted/v3"):
        checkpoint_owner.assert_fada_target_collection_checkpoint(loaded)


@pytest.mark.parametrize("schema_version", [5, "fada-adapted/v3"])
def test_target_collection_contract_accepts_current_deployable_schemas(
    schema_version: int | str,
) -> None:
    _adaptation, checkpoint_owner = _owners()
    assert hasattr(checkpoint_owner, "assert_fada_target_collection_checkpoint")
    loaded = type("Loaded", (), {"checkpoint": {"schema_version": schema_version}})()
    assert checkpoint_owner.assert_fada_target_collection_checkpoint(loaded) is loaded


def _owners() -> tuple[Any, Any]:
    try:
        adaptation = importlib.import_module("unilab.algos.torch.distill.fada_adaptation")
        checkpoint = importlib.import_module(
            "unilab.algos.torch.distill.fada_adaptation_checkpoint"
        )
    except ModuleNotFoundError as exc:
        pytest.fail(f"Stage-D checkpoint owner is missing: {exc}")
    return adaptation, checkpoint


def _config() -> FADAArchitectureConfig:
    return FADAArchitectureConfig(
        obs_dim=4,
        action_dim=2,
        command_dim=3,
        history_length=3,
        prediction_horizon=2,
        hidden_dim=8,
        num_heads=2,
        planner_layers=1,
        idm_encoder_layers=1,
        idm_decoder_layers=1,
        feedforward_dim=16,
    )


def _source_checkpoint(path: Path, policy: FADAPlannerIDMPolicy) -> Path:
    torch.save(
        {
            "schema_version": 2,
            "architecture": asdict(policy.config),
            "planner_state_dict": policy.planner.state_dict(),
            "idm_state_dict": policy.idm.state_dict(),
            "planner_optimizer_state_dict": {},
            "idm_optimizer_state_dict": {},
            "completed_iterations": 9,
            "samples_seen": 123,
            "runtime_config": {},
            "quality_metrics": {},
        },
        path,
    )
    return path


def _inputs(config: FADAArchitectureConfig) -> tuple[torch.Tensor, ...]:
    return (
        torch.arange(config.history_length * config.obs_dim, dtype=torch.float32).reshape(
            1, config.history_length, config.obs_dim
        )
        / 13.0,
        torch.arange(config.history_length * config.action_dim, dtype=torch.float32).reshape(
            1, config.history_length, config.action_dim
        )
        / 7.0,
        torch.tensor([[0.4, -0.2, 0.1]]),
    )


def test_adapted_checkpoint_round_trip_is_exact_and_self_contained(tmp_path: Path) -> None:
    adaptation, checkpoint_owner = _owners()
    torch.manual_seed(23)
    adapted = adaptation.inject_fada_idm_lora(
        FADAPlannerIDMPolicy(_config()), adaptation.FADALoRAConfig(dropout=0.0)
    )
    optimizer = torch.optim.AdamW(adaptation.fada_adapter_parameters(adapted.policy), lr=3e-4)
    with torch.no_grad():
        next(
            parameter
            for name, parameter in adaptation.fada_adapter_named_parameters(adapted.policy)
            if name.endswith("lora_q_B.weight")
        ).fill_(0.125)
    path = tmp_path / "adapted.pt"

    checkpoint_owner.save_fada_adapted_checkpoint(
        path,
        adapted.policy,
        optimizer,
        lora_config=adapted.lora_config,
        source_checkpoint_sha256="a" * 64,
        target_artifact_sha256="b" * 64,
        target_artifact_schema_version="fada-target-batch/v3",
        completed_steps=3,
        samples_seen=96,
        runtime_config={
            "batch_size": 32,
            "seed": 7,
            "target_domain": {"target_domain_id": "g1_slope_15_mujoco"},
        },
    )
    adapted.policy.eval()
    expected = adapted.policy(*_inputs(adapted.policy.config))

    loaded = checkpoint_owner.load_fada_adapted_checkpoint(path, device="cpu")
    observed = loaded.policy(*_inputs(loaded.policy.config))

    assert loaded.policy.training is False
    assert loaded.checkpoint["schema_version"] == "fada-adapted/v3"
    assert loaded.checkpoint["source_checkpoint_sha256"] == "a" * 64
    assert loaded.checkpoint["target_artifact_sha256"] == "b" * 64
    assert loaded.checkpoint["target_artifact_schema_version"] == "fada-target-batch/v3"
    assert loaded.checkpoint["target_domain_id"] == "g1_slope_15_mujoco"
    assert loaded.checkpoint["lora_config"]["target_modules"] == list(
        adapted.lora_config.target_modules
    )
    assert loaded.checkpoint["lora_config"]["adapter_type"] == "qv_attention"
    assert loaded.checkpoint["lora_config"]["target_projections"] == ["q", "v"]
    assert loaded.checkpoint["optimizer_state_dict"] == optimizer.state_dict()
    torch.testing.assert_close(observed.action_chunk, expected.action_chunk, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("completed_steps", [1.5, True, -1])
def test_adapted_checkpoint_rejects_invalid_step_identity_and_optimizer_ownership(
    tmp_path: Path, completed_steps: Any
) -> None:
    adaptation, checkpoint_owner = _owners()
    adapted = adaptation.inject_fada_idm_lora(
        FADAPlannerIDMPolicy(_config()), adaptation.FADALoRAConfig(dropout=0.0)
    )
    wrong_optimizer = torch.optim.SGD(
        list(adaptation.fada_adapter_parameters(adapted.policy))
        + [next(adapted.policy.planner.parameters())],
        lr=0.01,
    )

    with pytest.raises(ValueError, match="adapter|completed_steps"):
        checkpoint_owner.save_fada_adapted_checkpoint(
            tmp_path / "invalid.pt",
            adapted.policy,
            wrong_optimizer,
            lora_config=adapted.lora_config,
            source_checkpoint_sha256="a" * 64,
            target_artifact_sha256="b" * 64,
            target_artifact_schema_version="fada-target-batch/v3",
            completed_steps=completed_steps,
            samples_seen=0,
            runtime_config={},
        )


def test_deployable_loader_accepts_old_and_new_while_source_reader_rejects_new(
    tmp_path: Path,
) -> None:
    adaptation, checkpoint_owner = _owners()
    source_policy = FADAPlannerIDMPolicy(_config())
    source_path = _source_checkpoint(tmp_path / "source.pt", source_policy)
    adapted = adaptation.inject_fada_idm_lora(
        FADAPlannerIDMPolicy(_config()), adaptation.FADALoRAConfig(dropout=0.0)
    )
    adapted_path = tmp_path / "adapted.pt"
    checkpoint_owner.save_fada_adapted_checkpoint(
        adapted_path,
        adapted.policy,
        torch.optim.AdamW(adaptation.fada_adapter_parameters(adapted.policy), lr=3e-4),
        lora_config=adapted.lora_config,
        source_checkpoint_sha256="c" * 64,
        target_artifact_sha256="d" * 64,
        target_artifact_schema_version=checkpoint_owner.FADA_REAL_TARGET_ARTIFACT_SCHEMA_VERSION,
        completed_steps=0,
        samples_seen=0,
        runtime_config={"target_domain": {"target_domain_id": "g1_hang_book_real"}},
    )

    assert (
        checkpoint_owner.load_fada_deployable_policy_checkpoint(
            source_path, device="cpu"
        ).checkpoint["schema_version"]
        == 2
    )
    assert (
        checkpoint_owner.load_fada_deployable_policy_checkpoint(
            adapted_path, device="cpu"
        ).checkpoint["schema_version"]
        == "fada-adapted/v3"
    )
    loaded_adapted = checkpoint_owner.load_fada_deployable_policy_checkpoint(
        adapted_path, device="cpu"
    )
    assert (
        loaded_adapted.checkpoint["target_artifact_schema_version"]
        == checkpoint_owner.FADA_REAL_TARGET_ARTIFACT_SCHEMA_VERSION
    )
    assert loaded_adapted.checkpoint["target_domain_id"] == "g1_hang_book_real"
    with pytest.raises(ValueError, match="unsupported"):
        load_fada_policy_checkpoint(adapted_path, device="cpu")


def test_legacy_v2_linear_lora_remains_generic_playback_only(tmp_path: Path) -> None:
    adaptation, checkpoint_owner = _owners()
    legacy = adaptation._inject_fada_idm_legacy_linear_lora(
        FADAPlannerIDMPolicy(_config()), adaptation.FADALoRAConfig.legacy(dropout=0.0)
    )
    with torch.no_grad():
        next(
            parameter
            for name, parameter in adaptation.fada_adapter_named_parameters(legacy.policy)
            if name.endswith("lora_B.weight")
        ).fill_(0.125)
    path = tmp_path / "legacy-v2.pt"
    torch.save(
        {
            "schema_version": "fada-adapted/v2",
            "architecture": asdict(legacy.policy.config),
            "lora_config": {
                "rank": legacy.lora_config.rank,
                "alpha": legacy.lora_config.alpha,
                "dropout": legacy.lora_config.dropout,
                "target_modules": list(legacy.lora_config.target_modules),
            },
            "policy_state_dict": legacy.policy.state_dict(),
            "trainable_parameter_names": list(
                dict(adaptation.fada_adapter_named_parameters(legacy.policy))
            ),
            "optimizer_state_dict": {},
            "source_checkpoint_sha256": "a" * 64,
            "target_artifact_sha256": "b" * 64,
            "completed_steps": 1,
            "samples_seen": 1,
            "runtime_config": {},
        },
        path,
    )
    legacy.policy.eval()
    expected = legacy.policy(*_inputs(legacy.policy.config))

    loaded = checkpoint_owner.load_fada_deployable_policy_checkpoint(path, device="cpu")
    observed = loaded.policy(*_inputs(loaded.policy.config))

    assert loaded.checkpoint["schema_version"] == "fada-adapted/v2"
    assert loaded.checkpoint["lora_config"].get("adapter_type") is None
    torch.testing.assert_close(observed.action_chunk, expected.action_chunk, rtol=1e-6, atol=1e-7)

    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["schema_version"] = "fada-adapted/v1"
    payload["architecture"].pop("observation_contract")
    legacy_v1_path = tmp_path / "legacy-v1.pt"
    torch.save(payload, legacy_v1_path)
    loaded_v1 = checkpoint_owner.load_fada_deployable_policy_checkpoint(
        legacy_v1_path, device="cpu"
    )
    assert loaded_v1.policy.config.observation_contract == "legacy_actor_obs_v1"


@pytest.mark.parametrize("mutation", ["architecture", "manifest", "schema"])
def test_adapted_checkpoint_rejects_identity_drift_before_returning_policy(
    tmp_path: Path, mutation: str
) -> None:
    adaptation, checkpoint_owner = _owners()
    adapted = adaptation.inject_fada_idm_lora(
        FADAPlannerIDMPolicy(_config()), adaptation.FADALoRAConfig(dropout=0.0)
    )
    path = tmp_path / "adapted.pt"
    checkpoint_owner.save_fada_adapted_checkpoint(
        path,
        adapted.policy,
        torch.optim.AdamW(adaptation.fada_adapter_parameters(adapted.policy), lr=3e-4),
        lora_config=adapted.lora_config,
        source_checkpoint_sha256="e" * 64,
        target_artifact_sha256="f" * 64,
        target_artifact_schema_version="fada-target-batch/v2",
        completed_steps=1,
        samples_seen=6,
        runtime_config={},
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if mutation == "architecture":
        payload["architecture"]["obs_dim"] += 1
    elif mutation == "manifest":
        payload["lora_config"]["target_modules"] = ["action_head"]
    else:
        payload["schema_version"] = "fada-adapted/v999"
    torch.save(payload, path)

    with pytest.raises(ValueError, match="architecture|manifest|schema"):
        checkpoint_owner.load_fada_adapted_checkpoint(path, device="cpu")


def test_v3_adapted_checkpoint_requires_contract(
    tmp_path: Path,
) -> None:
    adaptation, checkpoint_owner = _owners()
    adapted = adaptation.inject_fada_idm_lora(
        FADAPlannerIDMPolicy(_config()), adaptation.FADALoRAConfig(dropout=0.0)
    )
    path = tmp_path / "adapted.pt"
    checkpoint_owner.save_fada_adapted_checkpoint(
        path,
        adapted.policy,
        torch.optim.AdamW(adaptation.fada_adapter_parameters(adapted.policy), lr=3e-4),
        lora_config=adapted.lora_config,
        source_checkpoint_sha256="a" * 64,
        target_artifact_sha256="b" * 64,
        target_artifact_schema_version="fada-target-batch/v2",
        completed_steps=0,
        samples_seen=0,
        runtime_config={},
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert checkpoint_owner.FADA_ADAPTED_CHECKPOINT_SCHEMA_VERSION == "fada-adapted/v3"
    payload["architecture"].pop("observation_contract")
    torch.save(payload, path)
    with pytest.raises(ValueError, match="observation_contract"):
        checkpoint_owner.load_fada_adapted_checkpoint(path, device="cpu")


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("target_artifact_schema_version", "fada-target-batch/v999", "artifact schema"),
        ("target_domain_id", "wrong-domain", "target-domain identity"),
    ],
)
def test_v3_adapted_checkpoint_rejects_target_lineage_drift(
    tmp_path: Path, field: str, value: str, match: str
) -> None:
    adaptation, checkpoint_owner = _owners()
    adapted = adaptation.inject_fada_idm_lora(
        FADAPlannerIDMPolicy(_config()), adaptation.FADALoRAConfig(dropout=0.0)
    )
    path = tmp_path / "adapted.pt"
    checkpoint_owner.save_fada_adapted_checkpoint(
        path,
        adapted.policy,
        torch.optim.AdamW(adaptation.fada_adapter_parameters(adapted.policy), lr=3e-4),
        lora_config=adapted.lora_config,
        source_checkpoint_sha256="a" * 64,
        target_artifact_sha256="b" * 64,
        target_artifact_schema_version="fada-target-batch/v3",
        completed_steps=0,
        samples_seen=0,
        runtime_config={"target_domain": {"target_domain_id": "g1_slope_15_mujoco"}},
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload[field] = value
    torch.save(payload, path)
    with pytest.raises(ValueError, match=match):
        checkpoint_owner.load_fada_adapted_checkpoint(path, device="cpu")
