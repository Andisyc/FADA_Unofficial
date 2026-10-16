"""Explicit bridge; no dependency from the PPO learner into distillation."""
from unilab.algos.torch.fada_official.checkpoint import (
    BEHAVIOR_PROFILE, LoadedOfficialOracle, load_oracle_policy,
    validate_environment, validate_lineage, validate_same_run,
)


def load_official_teacher(path, spec, *, device):
    if spec.algo_type != "fada_official_ppo":
        raise ValueError("Official Oracle requires teacher.algo_type='fada_official_ppo'")
    if (spec.obs_dim, spec.critic_obs_dim, spec.action_dim) != (322, 322, 29):
        raise ValueError("Official Oracle teacher spec must be 322/322/29")
    return load_oracle_policy(path, device=device)
