"""Sealed Oracle artifacts, separate from native learner and legacy SAC states."""
from __future__ import annotations

import copy
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tensordict import TensorDict

from unilab.envs.locomotion.g1.official_oracle.config import (
    JOINT_NAMES, OBS_SCHEMA, SOURCE_COMMIT, STUDENT_SCHEMA, build_environment_contract,
)
from unilab.algos.torch.fada_official_ppo import LEARNER_SCHEMA

ORACLE_SCHEMA = "fada_official_oracle_v1"
ACTION_SCHEMA = "g1_official_raw_pd025_v1"
BEHAVIOR_PROFILE = "official_oracle_flat_v1"


def _plain(value):
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def normalize_identity(record):
    if not isinstance(record, Mapping):
        raise ValueError("Official Oracle identity must be a mapping")
    result = _plain(record)
    contract = result.get("environment_contract")
    required = {"task_name": "G1OfficialOracleFlat", "backend": "mujoco",
                "source_commit": SOURCE_COMMIT, "observation_schema": OBS_SCHEMA,
                "student_schema": STUDENT_SCHEMA, "action_schema": ACTION_SCHEMA,
                "dimensions": [322, 322, 66, 29, 3], "joint_names": list(JOINT_NAMES)}
    if not isinstance(contract, dict) or any(contract.get(k) != v for k, v in required.items()):
        raise ValueError("Official Oracle environment/schema/action identity mismatch")
    for key in ("asset_sha256", "config_sha256"):
        value = contract.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Official Oracle missing {key}")
    if (result.get("schema") != ORACLE_SCHEMA or result.get("algorithm") != LEARNER_SCHEMA
            or not isinstance(result.get("oracle_lineage_id"), str)
            or not result["oracle_lineage_id"].strip()
            or type(result.get("completed_iterations")) is not int
            or result["completed_iterations"] < 1
            or result.get("role") not in {"intermediate", "final"}):
        raise ValueError("Official Oracle schema/lineage/iteration/role mismatch")
    reward = result.get("reward")
    if (not isinstance(reward, dict) or not isinstance(reward.get("statistic"), str)
            or not reward["statistic"].strip()
            or (reward.get("value") is not None and not math.isfinite(float(reward["value"])))):
        raise ValueError("Official Oracle reward provenance is missing or non-finite")
    _validate_model_config(result.get("actor_config"))
    _validate_model_config(result.get("critic_config"), actor=False)
    return result


def _validate_model_config(config, *, actor=True):
    if not isinstance(config, dict) or config.get("class_name") != "rsl_rl.models.MLPModel":
        raise ValueError("Official Oracle requires its checkpoint-owned MLPModel")
    allowed = {"class_name", "hidden_dims", "activation", "obs_normalization"}
    if actor:
        allowed.add("distribution_cfg")
    if set(config) - allowed or config.get("activation") != "elu" or config.get("obs_normalization") is not False:
        raise ValueError("Official Oracle MLP/normalization contract mismatch")
    if not config.get("hidden_dims") or any(type(n) is not int or n < 1 for n in config["hidden_dims"]):
        raise ValueError("Official Oracle hidden dimensions must be positive")
    if actor:
        distribution = config.get("distribution_cfg", {})
        if distribution != {"class_name": "unilab.algos.torch.fada_official_ppo:OfficialFadaGaussianDistribution",
                            "init_std": .8, "std_type": "scalar"}:
            raise ValueError("Official Oracle Gaussian distribution mismatch")


def make_identity(*, environment_contract, lineage_id, completed_iterations, role,
                  actor_config, critic_config, reward_value=None, reward_statistic):
    return normalize_identity({"schema": ORACLE_SCHEMA, "algorithm": LEARNER_SCHEMA,
        "environment_contract": environment_contract, "oracle_lineage_id": lineage_id,
        "completed_iterations": completed_iterations, "role": role,
        "actor_config": actor_config, "critic_config": critic_config,
        "reward": {"value": reward_value, "statistic": reward_statistic}})


def save_oracle_checkpoint(path, learner, *, identity, samples_seen=0):
    identity = normalize_identity(identity)
    if type(samples_seen) is not int or samples_seen < 0:
        raise ValueError("Official Oracle samples_seen must be nonnegative")
    if learner.storage.step or learner.transition.actions is not None:
        raise ValueError("Official Oracle save requires an empty rollout")
    state = learner.save()
    # Validate network identity before publishing, not at first playback.
    probe = LoadedOfficialOracle(identity, device="cpu")
    probe.actor.load_state_dict(state["actor_state_dict"], strict=True)
    from rsl_rl.models import MLPModel
    critic_cfg = copy.deepcopy(identity["critic_config"])
    critic_cfg.pop("class_name")
    with torch.random.fork_rng(devices=[]):
        obs = TensorDict({"critic": torch.zeros(1, 322)}, [1])
        critic = MLPModel(obs, {"critic": ["critic"]}, "critic", 1, **critic_cfg)
    critic.load_state_dict(state["critic_state_dict"], strict=True)
    payload = {"official_oracle": identity, "learner": state, "samples_seen": samples_seen}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite Oracle artifact: {target}")
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        # Hard-link publication fails atomically if another writer won the target.
        os.link(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def read_oracle_checkpoint(path):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Not a sealed Official Oracle checkpoint")
    payload["official_oracle"] = normalize_identity(payload.get("official_oracle"))
    state = payload.get("learner")
    if not isinstance(state, dict) or state.get("learner_schema") != LEARNER_SCHEMA:
        raise ValueError("Official Oracle native learner schema mismatch")
    if type(payload.get("samples_seen")) is not int or payload["samples_seen"] < 0:
        raise ValueError("Official Oracle sample cursor is missing")
    return payload


def restore_oracle_checkpoint(path, learner, *, expected_identity):
    payload = read_oracle_checkpoint(path)
    observed = payload["official_oracle"]
    expected = normalize_identity(expected_identity)
    validate_same_run(observed, expected)
    learner.load(payload["learner"])
    return {"completed_iterations": observed["completed_iterations"],
            "samples_seen": payload["samples_seen"]}


def validate_same_run(one, two):
    for name in ("schema", "algorithm", "environment_contract", "oracle_lineage_id",
                 "actor_config", "critic_config"):
        if one.get(name) != two.get(name):
            raise ValueError(f"Official Oracle lineage identity mismatch: {name}")
    if one["reward"]["statistic"] != two["reward"]["statistic"]:
        raise ValueError("Official Oracle reward statistic mismatch")


def validate_lineage(final, intermediates):
    final = normalize_identity(final)
    records = [normalize_identity(record) for record in intermediates]
    if final["role"] != "final" or len(records) != 20:
        raise ValueError("Official Oracle requires 20 intermediates plus one final")
    iterations = [record["completed_iterations"] for record in records]
    if len(set(iterations)) != 20:
        raise ValueError("Official Oracle intermediate iterations must be unique")
    for record in records:
        validate_same_run(final, record)
        if record["role"] != "intermediate" or record["completed_iterations"] >= final["completed_iterations"]:
            raise ValueError("Official Oracle intermediate role/iteration mismatch")
    return final


def make_selection_record(final, intermediates, *, sampling_mode):
    """Persistable selection provenance; the selection algorithm belongs to step 6."""
    admitted = validate_lineage(final, intermediates)
    if sampling_mode not in {"reward_aware", "logspace"}:
        raise ValueError("Unsupported Official Oracle checkpoint selection mode")
    return {"sampling_mode": sampling_mode, "final": admitted,
            "intermediates": [normalize_identity(record) for record in intermediates]}


class LoadedOfficialOracle(nn.Module):
    """Shared deterministic raw-action owner for playback and distillation."""
    obs_dim, critic_obs_dim, action_dim = 322, 322, 29
    behavior_profile = BEHAVIOR_PROFILE

    def __init__(self, identity, *, device="cpu"):
        super().__init__()
        from rsl_rl.models import MLPModel
        self.checkpoint_identity = normalize_identity(identity)
        self.oracle_lineage_id = self.checkpoint_identity["oracle_lineage_id"]
        cfg = copy.deepcopy(self.checkpoint_identity["actor_config"])
        cfg.pop("class_name")
        # Construction must not change caller RNG, especially during resident reload.
        with torch.random.fork_rng(devices=[]):
            obs = TensorDict({"actor": torch.zeros(1, 322)}, [1])
            self.actor = MLPModel(obs, {"actor": ["actor"]}, "actor", 29, **cfg).to(device)
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, observation):
        if isinstance(observation, Mapping) or isinstance(observation, TensorDict):
            observation = observation["actor"]
        if observation.ndim != 2 or observation.shape[1] != 322 or not torch.isfinite(observation).all():
            raise ValueError("Official Oracle requires finite teacher322 observations")
        action = self.actor(TensorDict({"actor": observation}, [observation.shape[0]]))
        if action.shape != (observation.shape[0], 29) or not torch.isfinite(action).all():
            raise ValueError("Official Oracle emitted invalid raw actions")
        return action

    def actions_from_env_observation(self, obs, info=None):
        value = np.asarray(obs["obs"], dtype=np.float32)
        critic = np.asarray(obs.get("critic"))
        if critic.shape != value.shape or not np.array_equal(critic, value):
            raise ValueError("Official Oracle actor/critic views must be identical teacher322")
        device = next(self.parameters()).device
        return self(torch.as_tensor(value, device=device)).cpu().numpy().copy()


def load_oracle_policy(path, *, device="cpu"):
    payload = read_oracle_checkpoint(path)
    policy = LoadedOfficialOracle(payload["official_oracle"], device=device)
    policy.actor.load_state_dict(payload["learner"]["actor_state_dict"], strict=True)
    return policy


def validate_environment(identity, env):
    identity = normalize_identity(identity)
    getter = getattr(env, "get_official_oracle_contract", None)
    if not callable(getter) or getter() != identity["environment_contract"]:
        raise ValueError("Official Oracle environment identity mismatch")
    if env.obs_groups_spec != {"obs": 322, "critic": 322} or env.action_space.shape != (29,):
        raise ValueError("Official Oracle environment dimensions mismatch")


def validate_student_binding(config, metadata):
    """New source/student artifacts bind the entire teacher/action/run identity."""
    if config.observation_contract != STUDENT_SCHEMA:
        return
    if not isinstance(metadata, Mapping):
        raise ValueError("Official student artifact requires source identity metadata")
    identity = normalize_identity(metadata.get("official_oracle_identity"))
    if identity["role"] != "final":
        raise ValueError("Official student artifact must bind the final teacher")
