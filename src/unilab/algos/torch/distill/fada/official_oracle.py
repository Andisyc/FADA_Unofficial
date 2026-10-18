"""Explicit bridge; no dependency from the PPO learner into distillation."""
from collections.abc import Mapping

from omegaconf import OmegaConf

from unilab.algos.torch.fada_official.checkpoint import (
    BEHAVIOR_PROFILE, LoadedOfficialOracle, load_oracle_policy,
    validate_environment, validate_lineage, validate_same_run,
)


def load_official_teacher(path, spec, *, device):
    if (spec.obs_dim, spec.critic_obs_dim, spec.action_dim) != (322, 322, 29):
        raise ValueError("Official Oracle teacher spec must be 322/322/29")
    if spec.algo_type == "fada_official_appo":
        from unilab.algos.torch.fada_official_appo.checkpoint import load_oracle_policy as load_appo
        return load_appo(path, device=device)
    if spec.algo_type != "fada_official_ppo":
        raise ValueError("Official Oracle requires an explicit PPO or APPO teacher.algo_type")
    return load_oracle_policy(path, device=device)


def validate_student_binding(config, metadata):
    """Bind students to their actual sealed teacher without weakening PPO checks."""
    from unilab.envs.locomotion.g1.official_oracle.config import STUDENT_SCHEMA
    from unilab.algos.torch.fada_official.checkpoint import validate_student_binding as validate_ppo
    if config.observation_contract != STUDENT_SCHEMA:
        return
    if not isinstance(metadata, Mapping):
        raise ValueError("Official student artifact requires source identity metadata")
    identity = metadata.get("official_oracle_identity")
    if isinstance(identity, Mapping) and identity.get("schema") == "fada_official_appo_oracle_v1":
        from unilab.algos.torch.fada_official_appo.checkpoint import normalize_identity
        if OmegaConf.is_config(identity):
            identity = OmegaConf.to_container(identity, resolve=True)
        normalized = normalize_identity(identity)
        if normalized["role"] != "final":
            raise ValueError("Official student artifact must bind the final APPO teacher")
        return
    validate_ppo(config, metadata)
