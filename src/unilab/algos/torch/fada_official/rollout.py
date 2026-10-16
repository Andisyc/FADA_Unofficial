"""Strict collected-rollout validation and the existing PPO transition boundary."""
import numpy as np
import torch
from tensordict import TensorDict

from unilab.ipc.rollout_ring_buffer import FADA_PPO_PROTOCOL, rollout_field_specs


def model_observations(actor, critic):
    return TensorDict({"actor": actor, "policy": actor, "critic": critic}, batch_size=[len(actor)])


def validate_round_payload(payload, *, expected_version, num_envs, num_steps, obs_dim, action_dim, critic_dim):
    specs = rollout_field_specs(1, num_envs, num_steps, obs_dim, action_dim,
                                critic_dim, protocol=FADA_PPO_PROTOCOL)
    if payload.keys() != specs.keys():
        raise ValueError("PPO rollout fields mismatch")
    for name, (shape, dtype) in specs.items():
        value = payload[name]
        if value.shape != shape[1:] or value.dtype != dtype:
            raise ValueError(f"PPO rollout shape/dtype mismatch: {name}")
        if not np.isfinite(value).all():
            raise ValueError(f"PPO rollout contains nonfinite values: {name}")
    if int(payload["policy_version"][0]) != expected_version:
        raise ValueError("PPO rollout policy version mismatch (stale, future or duplicate)")
    if int(payload["valid_steps"][0]) != num_steps:
        raise ValueError("PPO rollout is incomplete")
    for name in ("dones", "truncated"):
        if not np.isin(payload[name], [0, 1]).all():
            raise ValueError(f"PPO rollout mask must contain only 0/1: {name}")
    if np.any(payload["truncated"] > payload["dones"]):
        raise ValueError("PPO rollout timeout must also be done")
    if np.any(payload["action_std"] <= 0):
        raise ValueError("PPO rollout action_std must be positive")
    if np.any(payload["final_critic"][payload["truncated"] == 0] != 0):
        raise ValueError("PPO rollout non-timeout final_critic rows must be cleared")


def import_round_into_learner(learner, payload):
    """Import validated, owned env-major tensors; never resample old statistics.

    Bootstrap and GAE remain owned by OfficialFadaPPO. This function does not
    update parameters or acknowledge any borrowed shared-memory views.
    """
    storage = learner.storage
    if storage.step or learner.transition.actions is not None:
        raise RuntimeError("PPO rollout import requires empty learner storage")
    n, t = payload["rewards"].shape
    if (n, t) != (storage.num_envs, storage.num_transitions_per_env):
        raise ValueError("PPO rollout allocation does not match learner")
    tail = model_observations(payload["last_obs"], payload["last_critic"])
    with torch.no_grad():
        for step in range(t):
            tr = learner.transition
            tr.observations = model_observations(payload["obs"][:, step], payload["critic"][:, step])
            tr.actions = payload["actions"][:, step]
            tr.values = payload["values"][:, step, None]
            tr.actions_log_prob = payload["log_probs"][:, step]
            tr.distribution_params = (payload["action_mean"][:, step], payload["action_std"][:, step])
            next_obs = (model_observations(payload["obs"][:, step + 1], payload["critic"][:, step + 1])
                        if step + 1 < t else tail)
            timeout = payload["truncated"][:, step].bool()
            extras = {"time_outs": timeout}
            if torch.any(timeout):
                extras["time_out_bootstrap_obs"] = TensorDict(
                    {"critic": payload["final_critic"][:, step]}, batch_size=[n])
            learner.process_env_step(next_obs, payload["rewards"][:, step],
                                     payload["dones"][:, step].bool(), extras)
        learner.compute_returns(tail)
