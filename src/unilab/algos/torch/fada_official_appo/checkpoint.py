"""Sealed APPO artifacts. No legacy PPO/SAC loader or simulator is involved."""
from copy import deepcopy
from collections.abc import Mapping
import math
import os
from pathlib import Path
import tempfile

import torch
import numpy as np
from tensordict import TensorDict

from unilab.envs.locomotion.g1.official_oracle.config import (
    JOINT_NAMES, OBS_SCHEMA, SOURCE_COMMIT, STUDENT_SCHEMA,
)
from .config import validate_model_contract

ORACLE_SCHEMA = "fada_official_appo_oracle_v1"
ALGORITHM = "fada_official_appo"
LEARNER_SCHEMA = "unilab_appo_training_state_v1"
COUNTERS = ("generated_samples", "published_samples", "accepted_samples", "unique_used_samples",
            "reused_samples", "stale_dropped_samples", "abandoned_samples", "learner_updates",
            "optimizer_steps", "sample_exposures")


def training_contract(rl_cfg, *, num_envs, replay_queue_size):
    """Only output, device and cumulative stopping budgets may change on resume."""
    mutable = {"max_iterations", "max_unique_samples", "save_interval", "resume_path", "load_run",
               "resume", "algo_log_name"}
    result = deepcopy({k: v for k, v in rl_cfg.items() if k not in mutable})
    result.update(num_envs=num_envs, replay_queue_size=replay_queue_size)
    return result


def normalize_identity(identity):
    if not isinstance(identity, dict):
        raise ValueError("Not a sealed FADA APPO identity")
    if identity.get("schema") != ORACLE_SCHEMA or identity.get("algorithm") != ALGORITHM:
        raise ValueError("FADA APPO checkpoint algorithm/schema mismatch")
    contract = identity.get("environment_contract")
    required = dict(task_name="G1OfficialOracleFlat", backend="mujoco", source_commit=SOURCE_COMMIT,
        observation_schema=OBS_SCHEMA, student_schema=STUDENT_SCHEMA,
        action_schema="g1_official_raw_pd025_v1", dimensions=[322, 322, 66, 29, 3],
        joint_names=list(JOINT_NAMES))
    if not isinstance(contract, dict) or any(contract.get(k) != v for k, v in required.items()):
        raise ValueError("FADA APPO environment identity mismatch")
    for name in ("asset_sha256", "config_sha256"):
        value = contract.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("FADA APPO environment hash missing")
    for name in ("oracle_lineage_id", "session_id"):
        if not isinstance(identity.get(name), str) or not identity[name].strip():
            raise ValueError(f"FADA APPO {name} missing")
    if (identity.get("role") not in {"intermediate", "final"}
            or type(identity.get("completed_updates")) is not int or identity["completed_updates"] < 1):
        raise ValueError("FADA APPO checkpoint role/update cursor mismatch")
    static = identity.get("training_contract")
    validate_model_contract(static)
    if any(identity.get(k + "_config") != static[k] for k in ("actor", "critic")):
        raise ValueError("FADA APPO checkpoint model identity mismatch")
    reward = identity.get("reward")
    if (not isinstance(reward, dict) or reward.get("statistic") != "raw_episode_return_last100"
            or reward.get("session_id") != identity["session_id"]
            or type(reward.get("window_count")) is not int or not 0 <= reward["window_count"] <= 100
            or (reward["window_count"] == 0) != (reward.get("value") is None)
            or (reward.get("value") is not None and (type(reward["value"]) not in (int, float)
                                                     or not math.isfinite(reward["value"])))):
        raise ValueError("FADA APPO raw reward provenance mismatch")
    if identity.get("parent_checkpoint") is not None and not isinstance(identity["parent_checkpoint"], str):
        raise ValueError("FADA APPO parent checkpoint must be a path")
    return deepcopy(identity)


def validate_cursor(cursor, identity):
    if not isinstance(cursor, dict) or set(cursor) != set(COUNTERS):
        raise ValueError("FADA APPO sample cursor fields mismatch")
    if any(type(v) is not int or v < 0 for v in cursor.values()):
        raise ValueError("FADA APPO sample cursor must contain nonnegative integers")
    c = cursor
    if (c["learner_updates"] != identity["completed_updates"] or c["reused_samples"] != 0
            or not c["unique_used_samples"] <= c["accepted_samples"] <= c["published_samples"] <= c["generated_samples"]
            or c["unique_used_samples"] + c["stale_dropped_samples"] + c["abandoned_samples"] > c["generated_samples"]
            or c["accepted_samples"] + c["stale_dropped_samples"] > c["published_samples"]
            or c["optimizer_steps"] < c["learner_updates"]
            or c["sample_exposures"] < c["unique_used_samples"]):
        raise ValueError("FADA APPO sample/update cursor is inconsistent")
    return dict(c)


def validate_same_run(observed, *, environment_contract, static_contract, lineage_id=None):
    observed = normalize_identity(observed)
    if (observed["environment_contract"] != environment_contract
            or observed["training_contract"] != static_contract
            or (lineage_id is not None and lineage_id != observed["oracle_lineage_id"])):
        raise ValueError("FADA APPO resume environment/training contract/lineage mismatch")


def validate_same_lineage(one, two):
    one, two = normalize_identity(one), normalize_identity(two)
    validate_same_run(one, environment_contract=two["environment_contract"],
        static_contract=two["training_contract"], lineage_id=two["oracle_lineage_id"])


def validate_lineage(final, intermediates):
    final = normalize_identity(final)
    records = [normalize_identity(record) for record in intermediates]
    updates = [record["completed_updates"] for record in records]
    if final["role"] != "final" or len(records) != 20 or len(set(updates)) != 20:
        raise ValueError("FADA APPO requires 20 unique intermediates plus one final")
    for record in records:
        validate_same_lineage(final, record)
        if record["role"] != "intermediate" or record["completed_updates"] >= final["completed_updates"]:
            raise ValueError("FADA APPO intermediate role/update mismatch")
    return final


def validate_environment(identity, env):
    identity = normalize_identity(identity)
    getter = getattr(env, "get_official_oracle_contract", None)
    if not callable(getter) or getter() != identity["environment_contract"]:
        raise ValueError("FADA APPO environment identity mismatch")
    if env.obs_groups_spec != {"obs": 322, "critic": 322} or env.action_space.shape != (29,):
        raise ValueError("FADA APPO environment dimensions mismatch")


def cpu_snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {k: cpu_snapshot(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_snapshot(v) for v in value)
    return deepcopy(value)


def _validate_payload(payload):
    if not isinstance(payload, dict) or set(payload) != {"official_appo_oracle", "learner", "cursor"}:
        raise ValueError("Not a sealed FADA APPO checkpoint")
    identity = normalize_identity(payload["official_appo_oracle"])
    validate_cursor(payload["cursor"], identity)
    state = payload["learner"]
    if (not isinstance(state, dict) or state.get("learner_schema") != LEARNER_SCHEMA
            or state.get("update_counter") != identity["completed_updates"]):
        raise ValueError("FADA APPO learner/update cursor mismatch")
    return payload


def save_oracle_checkpoint(path, learner, *, identity, cursor):
    """Capture owned CPU state at a completed-update boundary, then publish once."""
    payload = _validate_payload({"official_appo_oracle": normalize_identity(identity),
        "cursor": dict(cursor), "learner": cpu_snapshot(learner.get_state_dict(complete=True))})
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite APPO artifact: {target}")
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.link(temporary, target)  # Atomic publication, including racing writers.
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def read_oracle_checkpoint(path):
    return _validate_payload(torch.load(Path(path), map_location="cpu", weights_only=True))


def restore_oracle_checkpoint(payload, learner, *, environment_contract, static_contract, lineage_id=None):
    _validate_payload(payload)
    validate_same_run(payload["official_appo_oracle"], environment_contract=environment_contract,
                      static_contract=static_contract, lineage_id=lineage_id)
    learner.restore_training_state(payload["learner"])
    return dict(payload["cursor"])


class LoadedOfficialAPPOOracle(torch.nn.Module):
    """One deterministic raw-action teacher for playback and source collection."""
    obs_dim, critic_obs_dim, action_dim = 322, 322, 29
    behavior_profile = "official_oracle_flat_v1"

    def __init__(self, identity, *, device="cpu"):
        super().__init__()
        from rsl_rl.models import MLPModel
        self.checkpoint_identity = normalize_identity(identity)
        self.oracle_lineage_id = identity["oracle_lineage_id"]
        config = deepcopy(identity["actor_config"])
        config.pop("class_name")
        with torch.random.fork_rng(devices=[]):
            self.actor = MLPModel(TensorDict({"policy": torch.zeros(1, 322)}, [1]),
                {"actor": ["policy"]}, "actor", 29, **config).to(device)
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, observation):
        if isinstance(observation, (Mapping, TensorDict)):
            observation = observation["actor"]
        if observation.ndim != 2 or observation.shape[1] != 322 or not torch.isfinite(observation).all():
            raise ValueError("FADA APPO requires finite teacher322 observations")
        result = self.actor(TensorDict({"policy": observation}, [observation.shape[0]]))
        if result.shape != (observation.shape[0], 29) or not torch.isfinite(result).all():
            raise ValueError("FADA APPO emitted invalid raw actions")
        return result

    def actions_from_env_observation(self, obs, info=None):
        value = np.asarray(obs["obs"], dtype=np.float32)
        device = next(self.actor.parameters()).device
        return self(torch.as_tensor(value, device=device)).cpu().numpy().copy()


def load_oracle_policy(path, *, device="cpu"):
    payload = read_oracle_checkpoint(path)
    policy = LoadedOfficialAPPOOracle(payload["official_appo_oracle"], device=device)
    reference = policy.actor.state_dict()
    saved = payload["learner"].get("actor")
    if (not isinstance(saved, dict) or saved.keys() != reference.keys()
            or any(not isinstance(v, torch.Tensor) or v.shape != reference[k].shape
                   or v.dtype != reference[k].dtype or not torch.isfinite(v).all() for k, v in saved.items())):
        raise ValueError("FADA APPO frozen actor state mismatch")
    policy.actor.load_state_dict(saved, strict=True)
    return policy
