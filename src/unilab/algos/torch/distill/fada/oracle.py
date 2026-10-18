"""Cold-path checkpoint Gateway for the single frozen FADA Oracle."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
from torch import nn

from unilab.algos.torch.common.actor_factory import build_actor
from unilab.algos.torch.common.normalization import EmpiricalNormalization
from unilab.algos.torch.distill.fada.privileged_oracle import (
    normalize_fada_oracle_checkpoint_identity,
    validate_fada_oracle_behavior_environment,
    validate_fada_oracle_lineage,
    validate_fada_single_reward,
)
from unilab.algos.torch.distill.learning.playback import load_distillation_student_policy
from unilab.algos.torch.distill.learning.teacher import (
    DistillationTeacherSpec,
    LoadedTeacherPolicy,
    load_sac_teacher_policy,
)
from unilab.algos.torch.hora.observations import split_hora_obs_with_priv_info

_FADA_ORACLE_COMMAND_LIMITS = ((-0.6, -0.4, -0.8), (1.0, 0.4, 0.8))


class _PrivilegedActor(Protocol):
    def explore(
        self,
        obs: torch.Tensor,
        priv_info: torch.Tensor,
        *,
        deterministic: bool,
    ) -> torch.Tensor: ...


def _is_distillation_student_checkpoint(payload: object) -> bool:
    return isinstance(payload, dict) and {
        "student_state_dict",
        "distill_runtime_cfg",
    }.issubset(payload)


class LoadedFADAPrivilegedOraclePolicy(nn.Module):
    """Frozen privileged Oracle with an env-observation inference boundary."""

    def __init__(
        self,
        *,
        actor: nn.Module,
        obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
        obs_normalizer: EmpiricalNormalization | None,
        checkpoint_identity: Mapping[str, object],
    ) -> None:
        super().__init__()
        self.actor = actor
        self.obs_dim = int(obs_dim)
        self.critic_obs_dim = int(critic_obs_dim)
        self.priv_info_dim = self.critic_obs_dim - self.obs_dim
        self.action_dim = int(action_dim)
        self.obs_normalizer = obs_normalizer
        self.checkpoint_identity = dict(checkpoint_identity)
        self.oracle_lineage_id = str(checkpoint_identity.get("oracle_lineage_id", ""))
        self.behavior_profile = str(checkpoint_identity.get("behavior_profile", ""))
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def forward(self, obs: torch.Tensor, priv_info: torch.Tensor) -> torch.Tensor:
        if obs.shape[-1] != self.obs_dim or priv_info.shape[-1] != self.priv_info_dim:
            raise ValueError(
                "privileged Oracle input shape mismatch: "
                f"obs={tuple(obs.shape)} priv_info={tuple(priv_info.shape)}"
            )
        actor_obs = obs
        if self.obs_normalizer is not None:
            actor_obs = self.obs_normalizer.forward(actor_obs, update=False)
        actor = cast(_PrivilegedActor, self.actor)
        action = actor.explore(actor_obs, priv_info, deterministic=True)
        if action.shape != (obs.shape[0], self.action_dim):
            raise ValueError(f"privileged Oracle action shape mismatch: {tuple(action.shape)}")
        return action

    def actions_from_env_observation(
        self,
        obs: Mapping[str, np.ndarray],
        info: Mapping[str, object] | None,
    ) -> np.ndarray:
        actor_obs, critic_obs, priv_info = split_hora_obs_with_priv_info(
            dict(obs), None if info is None else dict(info)
        )
        if critic_obs is None or critic_obs.shape != (actor_obs.shape[0], self.critic_obs_dim):
            raise ValueError(
                "privileged Oracle requires the exact critic observation shape "
                f"(*, {self.critic_obs_dim})"
            )
        if priv_info is None or priv_info.shape != (actor_obs.shape[0], self.priv_info_dim):
            raise ValueError(
                "privileged Oracle requires the exact critic-tail privileged shape "
                f"(*, {self.priv_info_dim})"
            )
        device = next(self.actor.parameters()).device
        action = self(
            torch.as_tensor(actor_obs, dtype=torch.float32, device=device),
            torch.as_tensor(priv_info, dtype=torch.float32, device=device),
        )
        result = action.detach().cpu().numpy().astype(np.float32)
        if not np.all(np.isfinite(result)):
            raise ValueError("privileged Oracle produced non-finite actions")
        return result


def validate_fada_oracle_environment_contract(
    checkpoint_identity: Mapping[str, object],
    env: Any,
    cfg: Any,
) -> None:
    """Reject same-shape Collector environments with different Actor semantics."""

    if checkpoint_identity.get("schema") in {"fada_official_oracle_v1", "fada_official_appo_oracle_v1"}:
        if checkpoint_identity["schema"] == "fada_official_appo_oracle_v1":
            from unilab.algos.torch.fada_official_appo.checkpoint import validate_environment
        else:
            from unilab.algos.torch.distill.fada.official_oracle import validate_environment
        if str(cfg.training.task_name) != "G1OfficialOracleFlat" or str(cfg.training.sim_backend) != "mujoco":
            raise ValueError("Official Oracle requires its MuJoCo task")
        validate_environment(checkpoint_identity, env)
        return
    expected_scalar_fields = {
        "task_name": str(cfg.training.task_name),
        "backend": str(cfg.training.sim_backend),
        "privileged_schema": str(cfg.env.fada_privileged_observation.schema),
    }
    for field_name, collector_value in expected_scalar_fields.items():
        if checkpoint_identity.get(field_name) != collector_value:
            raise ValueError(
                f"privileged Oracle Collector {field_name} mismatch: "
                f"checkpoint={checkpoint_identity.get(field_name)!r} "
                f"collector={collector_value!r}"
            )

    if not bool(cfg.env.fada_privileged_observation.enabled):
        raise ValueError("privileged Oracle Collector requires privileged observation enabled")
    if not np.isclose(float(cfg.env.ctrl_dt), 0.02):
        raise ValueError("privileged Oracle Collector requires ctrl_dt=0.02")
    if bool(cfg.env.mode_observation):
        raise ValueError("privileged Oracle Collector requires mode_observation=false")
    normalized_identity = normalize_fada_oracle_checkpoint_identity(checkpoint_identity)
    checkpoint_profile = str(normalized_identity["behavior_profile"])
    teacher_cfg = getattr(cfg, "teacher", None)
    configured_profile = getattr(teacher_cfg, "behavior_profile", checkpoint_profile)
    if str(configured_profile) != checkpoint_profile:
        raise ValueError(
            "privileged Oracle Collector behavior profile mismatch: "
            f"checkpoint={checkpoint_profile!r} collector={configured_profile!r}"
        )
    validate_fada_oracle_behavior_environment(cfg.env, checkpoint_profile)
    validate_fada_single_reward(
        reward_scales=dict(cfg.reward.scales),
        reward_config=dict(cfg.reward),
        behavior_profile=checkpoint_profile,
    )

    commands = cfg.env.commands
    observed_limits = tuple(tuple(float(value) for value in row) for row in commands.vel_limit)
    if observed_limits != _FADA_ORACLE_COMMAND_LIMITS:
        raise ValueError(
            "privileged Oracle Collector commands.vel_limit mismatch: "
            f"expected={_FADA_ORACLE_COMMAND_LIMITS} observed={observed_limits}"
        )

    raw_scale = cfg.env.control_config.action_scale
    collector_scale = (
        tuple(float(value) for value in raw_scale)
        if isinstance(raw_scale, (list, tuple))
        else (float(raw_scale),)
    )
    raw_checkpoint_scale = checkpoint_identity.get("action_scale")
    if not isinstance(raw_checkpoint_scale, (list, tuple)):
        raise ValueError("privileged Oracle checkpoint action_scale identity is missing")
    checkpoint_scale = tuple(float(value) for value in raw_checkpoint_scale)
    if checkpoint_scale != collector_scale:
        raise ValueError(
            "privileged Oracle Collector action_scale mismatch: "
            f"checkpoint={checkpoint_scale} collector={collector_scale}"
        )

    identity_getter = getattr(env, "get_fada_privileged_checkpoint_identity", None)
    if not callable(identity_getter):
        raise ValueError("Collector environment does not expose FADA checkpoint identity")
    environment_identity = cast(Any, identity_getter())
    identity_fields = {
        "body_names": tuple(environment_identity.body_names),
        "actuated_joint_names": tuple(environment_identity.actuated_joint_names),
        "privileged_field_slices": tuple(
            tuple(row) for row in environment_identity.field_slices
        ),
        "asset_sha256": str(environment_identity.asset_sha256),
    }
    for identity_name, identity_value in identity_fields.items():
        raw_checkpoint_value = checkpoint_identity.get(identity_name)
        checkpoint_value: object
        if identity_name == "asset_sha256":
            checkpoint_value = str(raw_checkpoint_value)
        elif identity_name == "privileged_field_slices":
            if not isinstance(raw_checkpoint_value, (list, tuple)):
                raise ValueError(
                    f"privileged Oracle checkpoint {identity_name} identity is missing"
                )
            checkpoint_value = tuple(tuple(row) for row in raw_checkpoint_value)
        else:
            if not isinstance(raw_checkpoint_value, (list, tuple)):
                raise ValueError(
                    f"privileged Oracle checkpoint {identity_name} identity is missing"
                )
            checkpoint_value = tuple(raw_checkpoint_value)
        if identity_value != checkpoint_value:
            raise ValueError(
                f"privileged Oracle Collector {identity_name} mismatch: "
                f"checkpoint={checkpoint_value!r} collector={identity_value!r}"
            )


def _privileged_checkpoint_metadata(payload: Mapping[str, object]) -> Mapping[str, object]:
    metadata = payload.get("fada_privileged_oracle")
    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint missing fada_privileged_oracle identity")
    metadata = normalize_fada_oracle_checkpoint_identity(metadata)
    dimensions = metadata.get("dimensions")
    if not isinstance(dimensions, Mapping):
        raise ValueError("privileged Oracle checkpoint identity missing dimensions")
    if metadata.get("actor_directly_privileged") is not True:
        raise ValueError("privileged Oracle checkpoint must bind actor_directly_privileged=true")
    return metadata


def _load_privileged_oracle_policy(
    payload: Mapping[str, object],
    spec: DistillationTeacherSpec,
    *,
    device: str | torch.device,
) -> LoadedFADAPrivilegedOraclePolicy:
    metadata = _privileged_checkpoint_metadata(payload)
    dimensions = metadata["dimensions"]
    assert isinstance(dimensions, Mapping)
    critic_obs_dim = int(dimensions["critic"])
    expected_critic_obs_dim = (
        critic_obs_dim if spec.critic_obs_dim is None else int(spec.critic_obs_dim)
    )
    expected = (int(spec.obs_dim), expected_critic_obs_dim, int(spec.action_dim))
    observed = (int(dimensions["obs"]), critic_obs_dim, int(dimensions["action"]))
    if observed != expected:
        raise ValueError(
            f"privileged Oracle checkpoint dimension mismatch: expected={expected} observed={observed}"
        )
    actor_state = payload.get("actor")
    if not isinstance(actor_state, Mapping):
        raise ValueError("privileged Oracle checkpoint is missing actor state")
    actor = build_actor(
        "privileged_locomotion_sac",
        int(spec.obs_dim),
        int(spec.action_dim),
        int(spec.actor_hidden_dim),
        bool(spec.use_layer_norm),
        device,
        priv_info_dim=critic_obs_dim - int(spec.obs_dim),
        priv_info_embed_dim=int(spec.priv_info_embed_dim),
        priv_mlp_hidden_dims=tuple(spec.priv_mlp_hidden_dims),
        priv_info_normalization=bool(spec.priv_info_normalization),
    )
    actor.load_state_dict(dict(actor_state), strict=True)
    obs_normalizer: EmpiricalNormalization | None = None
    if spec.obs_normalization:
        normalizer_state = payload.get("obs_normalizer")
        if not isinstance(normalizer_state, Mapping):
            raise ValueError("privileged Oracle checkpoint is missing obs_normalizer state")
        obs_normalizer = EmpiricalNormalization(shape=int(spec.obs_dim), device=device)
        obs_normalizer.load_state_dict(dict(normalizer_state), strict=True)
        obs_normalizer.eval()
    return LoadedFADAPrivilegedOraclePolicy(
        actor=actor,
        obs_dim=int(spec.obs_dim),
        critic_obs_dim=critic_obs_dim,
        action_dim=int(spec.action_dim),
        obs_normalizer=obs_normalizer,
        checkpoint_identity=metadata,
    )


def validate_loaded_fada_oracle_lineage(
    final_policy: nn.Module,
    intermediate_policies: list[nn.Module],
    *,
    expected_behavior_profile: str | None = None,
) -> None:
    """Validate the complete privileged 20+1 lineage before env construction."""

    from unilab.algos.torch.distill.fada.official_oracle import (
        LoadedOfficialOracle, BEHAVIOR_PROFILE, validate_lineage,
    )
    from unilab.algos.torch.fada_official_appo.checkpoint import (
        LoadedOfficialAPPOOracle, validate_lineage as validate_appo_lineage,
    )
    if isinstance(final_policy, LoadedOfficialAPPOOracle):
        if expected_behavior_profile not in (None, BEHAVIOR_PROFILE):
            raise ValueError("APPO Oracle behavior profile mismatch")
        if not all(isinstance(p, LoadedOfficialAPPOOracle) for p in intermediate_policies):
            raise ValueError("APPO Oracle lineage mixes algorithms")
        validate_appo_lineage(final_policy.checkpoint_identity,
                             [p.checkpoint_identity for p in intermediate_policies])
        return
    if any(isinstance(p, LoadedOfficialAPPOOracle) for p in intermediate_policies):
        raise ValueError("FADA Oracle lineage mixes algorithms")
    if isinstance(final_policy, LoadedOfficialOracle):
        if expected_behavior_profile not in (None, BEHAVIOR_PROFILE):
            raise ValueError("Official Oracle behavior profile mismatch")
        if not all(isinstance(p, LoadedOfficialOracle) for p in intermediate_policies):
            raise ValueError("Official Oracle lineage mixes algorithms")
        validate_lineage(final_policy.checkpoint_identity,
                         [p.checkpoint_identity for p in intermediate_policies])
        return
    if any(isinstance(p, LoadedOfficialOracle) for p in intermediate_policies):
        raise ValueError("FADA Oracle lineage mixes algorithms")
    if not isinstance(final_policy, LoadedFADAPrivilegedOraclePolicy):
        if expected_behavior_profile is not None:
            raise ValueError(
                "FADA source behavior profile requires a privileged Oracle checkpoint"
            )
        if any(
            isinstance(policy, LoadedFADAPrivilegedOraclePolicy) for policy in intermediate_policies
        ):
            raise ValueError("FADA Oracle lineage mixes privileged and non-privileged policies")
        return
    if not all(
        isinstance(policy, LoadedFADAPrivilegedOraclePolicy) for policy in intermediate_policies
    ):
        raise ValueError("privileged final Oracle requires privileged intermediate Oracles")
    records = [
        policy.checkpoint_identity
        for policy in intermediate_policies
        if isinstance(policy, LoadedFADAPrivilegedOraclePolicy)
    ]
    records.append(final_policy.checkpoint_identity)
    admitted = validate_fada_oracle_lineage(records)
    if (
        expected_behavior_profile is not None
        and admitted.behavior_profile != expected_behavior_profile
    ):
        raise ValueError(
            "FADA Oracle lineage behavior profile mismatch: "
            f"expected={expected_behavior_profile!r} "
            f"observed={admitted.behavior_profile!r}"
        )


def load_fada_oracle_policy(
    checkpoint_path: str | Path,
    spec: DistillationTeacherSpec,
    *,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Load one SAC or distillation policy behind the FADA tensor-policy contract."""

    path = Path(checkpoint_path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, Mapping) and "official_appo_oracle" in payload:
        if spec.algo_type != "fada_official_appo":
            raise ValueError("APPO Oracle checkpoint requires teacher.algo_type='fada_official_appo'")
        from unilab.algos.torch.distill.fada.official_oracle import load_official_teacher
        policy = load_official_teacher(path, spec, device=device)
    elif isinstance(payload, Mapping) and "official_oracle" in payload:
        if spec.algo_type != "fada_official_ppo":
            raise ValueError("PPO Oracle checkpoint requires teacher.algo_type='fada_official_ppo'")
        from unilab.algos.torch.distill.fada.official_oracle import load_official_teacher
        policy = load_official_teacher(path, spec, device=device)
    elif spec.algo_type in {"fada_official_ppo", "fada_official_appo"} or (
        isinstance(payload, Mapping) and payload.get("learner_schema") in
        {"fada_official_ppo_v1", "unilab_appo_training_state_v1"}
    ):
        raise ValueError("Official Oracle requires a sealed checkpoint, not a native learner state")
    elif _is_distillation_student_checkpoint(payload):
        loaded = load_distillation_student_policy(path, device=device)
        if loaded.obs_dim != int(spec.obs_dim):
            raise ValueError(
                "FADA Oracle obs dim mismatch: "
                f"checkpoint={loaded.obs_dim}, configured={int(spec.obs_dim)}"
            )
        if loaded.action_dim != int(spec.action_dim):
            raise ValueError(
                "FADA Oracle action dim mismatch: "
                f"checkpoint={loaded.action_dim}, configured={int(spec.action_dim)}"
            )
        policy = loaded.policy
    elif isinstance(payload, Mapping) and "fada_privileged_oracle" in payload:
        if spec.algo_type != "privileged_locomotion_sac":
            raise ValueError(
                "privileged Oracle checkpoint requires teacher.algo_type="
                "'privileged_locomotion_sac'"
            )
        policy = _load_privileged_oracle_policy(payload, spec, device=device)
    else:
        policy = load_sac_teacher_policy(path, spec, device=device)

    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return policy


def reload_fada_oracle_policy_(
    policy: nn.Module,
    checkpoint_path: str | Path,
    spec: DistillationTeacherSpec,
) -> None:
    """Reload a same-lineage Oracle without changing its runtime object identity."""

    replacement = load_fada_oracle_policy(
        checkpoint_path,
        spec,
        device=next(policy.parameters()).device,
    )
    if type(replacement) is not type(policy):
        raise ValueError("Oracle checkpoint runtime type changed during resident reload")
    from unilab.algos.torch.distill.fada.official_oracle import LoadedOfficialOracle, validate_same_run
    from unilab.algos.torch.fada_official_appo.checkpoint import LoadedOfficialAPPOOracle, validate_same_lineage
    if isinstance(policy, LoadedOfficialAPPOOracle):
        validate_same_lineage(policy.checkpoint_identity, replacement.checkpoint_identity)
        policy.actor.load_state_dict(replacement.actor.state_dict(), strict=True)
        policy.checkpoint_identity = replacement.checkpoint_identity
        policy.eval().requires_grad_(False)
        return
    if isinstance(policy, LoadedOfficialOracle):
        validate_same_run(policy.checkpoint_identity, replacement.checkpoint_identity)
        policy.actor.load_state_dict(replacement.actor.state_dict(), strict=True)
        policy.checkpoint_identity = replacement.checkpoint_identity
        policy.eval().requires_grad_(False)
        return
    if isinstance(policy, LoadedFADAPrivilegedOraclePolicy):
        assert isinstance(replacement, LoadedFADAPrivilegedOraclePolicy)
        if replacement.oracle_lineage_id != policy.oracle_lineage_id:
            raise ValueError("intermediate Oracle checkpoint lineage mismatch")
        if replacement.behavior_profile != policy.behavior_profile:
            raise ValueError("intermediate Oracle checkpoint behavior profile mismatch")
        policy.actor.load_state_dict(replacement.actor.state_dict(), strict=True)
        if (policy.obs_normalizer is None) != (replacement.obs_normalizer is None):
            raise ValueError("Oracle observation normalizer ownership changed during reload")
        if policy.obs_normalizer is not None and replacement.obs_normalizer is not None:
            policy.obs_normalizer.load_state_dict(
                replacement.obs_normalizer.state_dict(), strict=True
            )
        policy.eval()
        return
    if isinstance(policy, LoadedTeacherPolicy):
        assert isinstance(replacement, LoadedTeacherPolicy)
        policy.actor.load_state_dict(replacement.actor.state_dict(), strict=True)
        if policy.obs_normalizer is not None and replacement.obs_normalizer is not None:
            policy.obs_normalizer.load_state_dict(
                replacement.obs_normalizer.state_dict(), strict=True
            )
        policy.eval()
        return
    raise TypeError(f"unsupported resident Oracle policy type: {type(policy).__name__}")


__all__ = [
    "LoadedFADAPrivilegedOraclePolicy",
    "load_fada_oracle_policy",
    "reload_fada_oracle_policy_",
    "validate_fada_oracle_environment_contract",
    "validate_loaded_fada_oracle_lineage",
]
