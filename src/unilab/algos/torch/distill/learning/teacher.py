from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import torch
from torch import nn

from unilab.algos.torch.common.actor_factory import build_actor
from unilab.algos.torch.common.normalization import EmpiricalNormalization
from unilab.training.sim2sim import policy_load_dim_guard


@dataclass(frozen=True)
class DistillationTeacherSpec:
    """Shape and architecture contract for a loaded teacher actor."""

    obs_dim: int
    action_dim: int
    algo_type: Literal["sac", "privileged_locomotion_sac", "fada_official_ppo", "fada_official_appo"] = "sac"
    actor_hidden_dim: int = 512
    use_layer_norm: bool = True
    obs_normalization: bool = False
    critic_obs_dim: int | None = None
    priv_info_embed_dim: int = 32
    priv_mlp_hidden_dims: tuple[int, ...] = (256, 128, 32)
    priv_info_normalization: bool = True


@dataclass(frozen=True)
class DistillationTeacherCheckpointInfo:
    """Minimal checkpoint shape facts used before loading a frozen teacher."""

    checkpoint_path: str
    actor_input_dim: int
    first_weight_key: str


class LoadedTeacherPolicy(nn.Module):
    """Frozen teacher policy wrapper used by behavior distillation."""

    def __init__(
        self,
        *,
        actor: nn.Module,
        obs_dim: int,
        action_dim: int,
        obs_normalizer: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.obs_normalizer = obs_normalizer
        self.eval()
        for param in self.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(
                f"Teacher obs dim mismatch: expected {self.obs_dim}, got {obs.shape[-1]}"
            )
        actor_obs = obs
        if self.obs_normalizer is not None:
            actor_obs = cast(torch.Tensor, self.obs_normalizer(actor_obs, update=False))
        explore = getattr(self.actor, "explore", None)
        if callable(explore):
            action = explore(actor_obs, deterministic=True)
        else:
            action = self.actor(actor_obs)
        if isinstance(action, tuple):
            action = action[0]
        action = cast(torch.Tensor, action).detach()
        if action.shape[-1] != self.action_dim:
            raise ValueError(
                f"Teacher action dim mismatch: expected {self.action_dim}, got {action.shape[-1]}"
            )
        return action


def _load_optional_obs_normalizer(
    checkpoint: dict,
    spec: DistillationTeacherSpec,
    *,
    device: str | torch.device,
) -> EmpiricalNormalization | None:
    normalizer_state = checkpoint.get("obs_normalizer")
    if normalizer_state is None:
        return None
    normalizer = EmpiricalNormalization(shape=int(spec.obs_dim), device=device)
    normalizer.load_state_dict(normalizer_state)
    normalizer.eval()
    return normalizer


def inspect_sac_teacher_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> DistillationTeacherCheckpointInfo:
    """Inspect a SAC teacher checkpoint without constructing the actor."""

    del device
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    actor_state = checkpoint.get("actor") if isinstance(checkpoint, dict) else None
    if not isinstance(actor_state, dict):
        raise ValueError(f"SAC teacher checkpoint does not contain actor: {checkpoint_path}")
    for key, value in actor_state.items():
        shape = getattr(value, "shape", None)
        if "weight" in str(key) and shape is not None and len(shape) == 2:
            return DistillationTeacherCheckpointInfo(
                checkpoint_path=str(checkpoint_path),
                actor_input_dim=int(shape[1]),
                first_weight_key=str(key),
            )
    raise ValueError(f"SAC teacher checkpoint actor has no rank-2 weight: {checkpoint_path}")


def validate_sac_teacher_checkpoint_contract(
    checkpoint_path: str | Path,
    spec: DistillationTeacherSpec,
    *,
    device: str | torch.device = "cpu",
) -> DistillationTeacherCheckpointInfo:
    """Validate checkpoint input dim against the configured distillation teacher."""

    if spec.algo_type != "sac":
        raise ValueError(f"Unsupported distillation teacher algo_type: {spec.algo_type!r}")
    info = inspect_sac_teacher_checkpoint(checkpoint_path, device=device)
    if int(info.actor_input_dim) != int(spec.obs_dim):
        raise ValueError(
            "SAC teacher checkpoint obs dim mismatch: "
            f"checkpoint actor input dim={info.actor_input_dim} "
            f"({info.first_weight_key}), configured teacher.obs_dim={int(spec.obs_dim)}. "
            "For a legacy 100-D checkpoint, set `teacher.obs_dim=100` and "
            "`training.collect_teacher_projection=pad_zeros`; for the current "
            "G1WalkHeight default, use or train a 99-D teacher checkpoint."
        )
    return info


def load_sac_teacher_policy(
    checkpoint_path: str | Path,
    spec: DistillationTeacherSpec,
    *,
    device: str | torch.device = "cpu",
) -> LoadedTeacherPolicy:
    """Load a SAC actor checkpoint as a frozen distillation teacher."""

    if spec.algo_type != "sac":
        raise ValueError(f"Unsupported distillation teacher algo_type: {spec.algo_type!r}")

    validate_sac_teacher_checkpoint_contract(checkpoint_path, spec, device=device)
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    actor_state = checkpoint.get("actor")
    if actor_state is None:
        raise ValueError(f"SAC teacher checkpoint does not contain actor: {checkpoint_path}")

    actor = build_actor(
        "sac",
        int(spec.obs_dim),
        int(spec.action_dim),
        int(spec.actor_hidden_dim),
        bool(spec.use_layer_norm),
        device,
    )
    with policy_load_dim_guard(
        env_obs_dim=int(spec.obs_dim),
        env_action_dim=int(spec.action_dim),
        algo_name="sac_teacher",
    ):
        actor.load_state_dict(actor_state, strict=True)
    actor.eval()

    obs_normalizer = (
        _load_optional_obs_normalizer(checkpoint, spec, device=device)
        if spec.obs_normalization
        else None
    )
    return LoadedTeacherPolicy(
        actor=actor,
        obs_dim=int(spec.obs_dim),
        action_dim=int(spec.action_dim),
        obs_normalizer=obs_normalizer,
    )


def reload_sac_teacher_policy_(
    policy: LoadedTeacherPolicy,
    checkpoint_path: str | Path,
    spec: DistillationTeacherSpec,
) -> None:
    """Reload one CPU-owned SAC checkpoint into an existing teacher policy."""

    if policy.obs_dim != int(spec.obs_dim) or policy.action_dim != int(spec.action_dim):
        raise ValueError("resident SAC teacher shape contract does not match the reload spec")
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    actor_state = checkpoint.get("actor") if isinstance(checkpoint, dict) else None
    if not isinstance(actor_state, dict):
        raise ValueError(f"SAC teacher checkpoint does not contain actor: {checkpoint_path}")

    resident_state = policy.actor.state_dict()
    if actor_state.keys() != resident_state.keys():
        missing = sorted(set(resident_state) - set(actor_state))
        unexpected = sorted(set(actor_state) - set(resident_state))
        raise ValueError(
            "SAC teacher checkpoint actor keys do not match the resident actor: "
            f"missing={missing}, unexpected={unexpected}"
        )
    shape_mismatches = {
        key: (tuple(actor_state[key].shape), tuple(resident_state[key].shape))
        for key in resident_state
        if tuple(actor_state[key].shape) != tuple(resident_state[key].shape)
    }
    if shape_mismatches:
        raise ValueError(
            "SAC teacher checkpoint actor shapes do not match the resident actor: "
            f"{shape_mismatches}"
        )

    normalizer_state = checkpoint.get("obs_normalizer")
    resident_normalizer = policy.obs_normalizer
    if (normalizer_state is None) != (resident_normalizer is None):
        raise ValueError(
            "SAC teacher checkpoint normalization ownership does not match the resident actor"
        )

    with policy_load_dim_guard(
        env_obs_dim=int(spec.obs_dim),
        env_action_dim=int(spec.action_dim),
        algo_name="sac_teacher_reload",
    ):
        policy.actor.load_state_dict(actor_state, strict=True)
        if resident_normalizer is not None:
            if not isinstance(normalizer_state, dict):
                raise ValueError("SAC teacher checkpoint normalizer state must be a mapping")
            resident_normalizer.load_state_dict(normalizer_state, strict=True)
    policy.eval()
