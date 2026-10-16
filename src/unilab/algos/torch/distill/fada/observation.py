from __future__ import annotations

import numpy as np
import torch

FADA_LEGACY_OBSERVATION_CONTRACT = "legacy_actor_obs_v1"
FADA_G1_STATE_OBSERVATION_CONTRACT = "g1_fada_state_v2"
FADA_G1_ACTOR_OBS_DIM = 98
FADA_G1_STATE_DIM = 66
FADA_G1_ACTION_DIM = 29
FADA_G1_COMMAND_DIM = 3
FADA_OFFICIAL_STUDENT_CONTRACT = "g1_official_oracle_student66_v1"


def project_fada_g1_state(source_obs: np.ndarray) -> np.ndarray:
    """Project one G1 actor-observation batch into the non-leaking FADA state."""

    source = np.asarray(source_obs, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != FADA_G1_ACTOR_OBS_DIM:
        raise ValueError(
            "g1_fada_state_v2 requires finite rank-2 actor observations with width "
            f"{FADA_G1_ACTOR_OBS_DIM}, got {source.shape}"
        )
    if not bool(np.all(np.isfinite(source))):
        raise ValueError("g1_fada_state_v2 requires finite actor observations")
    return np.concatenate([source[:, :64], source[:, 96:98]], axis=1, dtype=np.float32)


def projection_for_fada_observation_contract(observation_contract: str) -> str:
    """Return the sole admitted raw-observation projection for one architecture identity."""

    if observation_contract == FADA_LEGACY_OBSERVATION_CONTRACT:
        return "identity"
    if observation_contract == FADA_G1_STATE_OBSERVATION_CONTRACT:
        return FADA_G1_STATE_OBSERVATION_CONTRACT
    if observation_contract == FADA_OFFICIAL_STUDENT_CONTRACT:
        return FADA_OFFICIAL_STUDENT_CONTRACT
    raise ValueError(f"unsupported FADA observation_contract: {observation_contract!r}")


def raw_observation_dim_for_fada_contract(
    observation_contract: str,
    *,
    policy_observation_dim: int,
) -> int:
    """Return the public environment width consumed before policy projection."""

    projection = projection_for_fada_observation_contract(observation_contract)
    if observation_contract == FADA_OFFICIAL_STUDENT_CONTRACT:
        return 322
    return int(policy_observation_dim) if projection == "identity" else FADA_G1_ACTOR_OBS_DIM


def project_fada_observation_tensor(
    observation: torch.Tensor,
    *,
    observation_contract: str,
) -> torch.Tensor:
    """Project a hot-path playback tensor without device or host transfer."""

    if observation_contract == FADA_LEGACY_OBSERVATION_CONTRACT:
        return observation
    if observation_contract != FADA_G1_STATE_OBSERVATION_CONTRACT:
        raise ValueError(f"unsupported FADA observation_contract: {observation_contract!r}")
    if observation.ndim != 2 or observation.shape[1] != FADA_G1_ACTOR_OBS_DIM:
        raise ValueError(
            "g1_fada_state_v2 playback requires rank-2 actor observations with width "
            f"{FADA_G1_ACTOR_OBS_DIM}, got {tuple(observation.shape)}"
        )
    return torch.cat((observation[:, :64], observation[:, 96:98]), dim=1)


def assert_fada_projection_matches_contract(
    *,
    observation_contract: str,
    projection: str,
) -> None:
    """Fail before lifecycle mutation when a route mixes observation semantics."""

    expected = projection_for_fada_observation_contract(observation_contract)
    if projection != expected:
        raise ValueError(
            "FADA observation projection does not match architecture contract: "
            f"contract={observation_contract!r} expected={expected!r} observed={projection!r}"
        )


def assert_fada_active_route_contract(
    *,
    observation_contract: str,
    projection: str,
) -> None:
    """Admit only the active non-leaking G1 contract at Stage C/D boundaries."""

    assert_fada_projection_matches_contract(
        observation_contract=observation_contract,
        projection=projection,
    )
    if observation_contract != FADA_G1_STATE_OBSERVATION_CONTRACT:
        raise ValueError(
            "active FADA route requires observation_contract="
            f"{FADA_G1_STATE_OBSERVATION_CONTRACT!r}, got {observation_contract!r}"
        )
