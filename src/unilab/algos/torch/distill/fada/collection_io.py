from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, ContextManager, cast

import numpy as np
import torch

from unilab.algos.torch.distill.collection.common import (
    project_teacher_obs,
)
from unilab.algos.torch.distill.fada.model import FADAArchitectureConfig, FADAPlannerIDMPolicy
from unilab.algos.torch.distill.fada.collection_views import student_view


def _module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _obs_array(obs: Mapping[str, Any], key: str) -> np.ndarray:
    if key not in obs:
        raise KeyError(f"observation key {key!r} not found; available={sorted(obs)}")
    value = np.asarray(obs[key], dtype=np.float32)
    if value.ndim != 2 or not np.all(np.isfinite(value)):
        raise ValueError(f"observation {key!r} must be finite rank-2, got {value.shape}")
    return value


def _command_array(
    info: Mapping[str, Any],
    keys: Sequence[str],
    *,
    expected_rows: int,
    expected_dim: int,
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for key in keys:
        if key not in info:
            raise KeyError(f"command info key {key!r} not found; available={sorted(info)}")
        value = np.asarray(info[key], dtype=np.float32)
        if value.ndim == 1:
            value = value[:, None]
        if value.ndim != 2 or value.shape[0] != int(expected_rows):
            raise ValueError(
                f"command info key {key!r} must have {expected_rows} rows, got {value.shape}"
            )
        columns.append(value)
    command = np.concatenate(columns, axis=1)
    if command.shape != (int(expected_rows), int(expected_dim)):
        raise ValueError(
            f"complete command shape mismatch: expected {(expected_rows, expected_dim)}, "
            f"got {command.shape} from keys={list(keys)}"
        )
    if not np.all(np.isfinite(command)):
        raise ValueError("complete command must contain only finite values")
    return command


def _policy_actions(policy: torch.nn.Module, obs: np.ndarray, *, action_dim: int) -> np.ndarray:
    tensor = torch.as_tensor(obs, dtype=torch.float32, device=_module_device(policy))
    with torch.inference_mode():
        output = policy(tensor)
    if isinstance(output, tuple):
        output = output[0]
    action = torch.as_tensor(output).detach()
    if action.shape != (obs.shape[0], int(action_dim)):
        raise ValueError(
            f"Oracle action shape mismatch: expected {(obs.shape[0], action_dim)}, "
            f"got {tuple(action.shape)}"
        )
    result = action.cpu().numpy().astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError("Oracle produced non-finite actions")
    return result


def _oracle_actions(
    policy: torch.nn.Module,
    env_obs: Mapping[str, Any],
    info: Mapping[str, Any],
    projected_obs: np.ndarray,
    *,
    action_dim: int,
) -> np.ndarray:
    """Query either a deployable-only or privileged Oracle through one strict seam."""

    privileged_inference = getattr(policy, "actions_from_env_observation", None)
    if not callable(privileged_inference):
        return _policy_actions(policy, projected_obs, action_dim=action_dim)
    result = np.asarray(privileged_inference(env_obs, info), dtype=np.float32)
    expected = (projected_obs.shape[0], int(action_dim))
    if result.shape != expected:
        raise ValueError(
            f"privileged Oracle action shape mismatch: expected {expected}, got {result.shape}"
        )
    if not np.all(np.isfinite(result)):
        raise ValueError("privileged Oracle produced non-finite actions")
    return result


def _fada_actions(
    policy: FADAPlannerIDMPolicy,
    observation_history: np.ndarray,
    action_history: np.ndarray,
    command: np.ndarray,
) -> np.ndarray:
    device = _module_device(policy)
    with torch.inference_mode():
        output = policy(
            torch.as_tensor(observation_history, dtype=torch.float32, device=device),
            torch.as_tensor(action_history, dtype=torch.float32, device=device),
            torch.as_tensor(command, dtype=torch.float32, device=device),
        )
    result = output.action.detach().cpu().numpy().astype(np.float32)
    if result.shape != (observation_history.shape[0], policy.config.action_dim):
        raise ValueError(f"FADA rollout action shape mismatch: got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError("FADA rollout produced non-finite actions")
    return result


def _done_mask(state: Any, *, num_envs: int) -> np.ndarray:
    done = np.zeros((int(num_envs),), dtype=np.bool_)
    for value in (getattr(state, "terminated", None), getattr(state, "truncated", None)):
        if value is None:
            continue
        current = np.asarray(value, dtype=np.bool_).reshape(-1)
        if current.shape != done.shape:
            raise ValueError(
                f"done mask shape mismatch: expected {done.shape}, got {current.shape}"
            )
        done |= current
    return done


def _next_after_done(
    env: Any,
    state: Any,
    done: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    obs = {key: np.asarray(value).copy() for key, value in state.obs.items()}
    info = dict(getattr(state, "info", {}))
    if not bool(np.any(done)):
        return obs, info
    final_observation = getattr(state, "final_observation", None)
    if isinstance(final_observation, Mapping) or isinstance(info.get("final_observation"), Mapping):
        return obs, info
    indices = np.flatnonzero(done).astype(np.int32)
    reset_obs, reset_info = env.reset(indices)
    for key, value in reset_obs.items():
        obs[key][indices] = value
    for key, value in reset_info.items():
        if isinstance(value, np.ndarray):
            if key not in info:
                info[key] = np.zeros((len(done),) + value.shape[1:], dtype=value.dtype)
            info[key][indices] = value
        else:
            info[key] = value
    return obs, info


def _oracle_shadow_pair(
    env: Any,
    *,
    teacher_policy: torch.nn.Module,
    initial_oracle_actions: np.ndarray,
    initial_command: np.ndarray,
    config: FADAArchitectureConfig,
    observation_key: str,
    teacher_projection: str,
    student_projection: str,
    student_drop_index: int | None,
    command_info_keys: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll final Oracle for K steps and restore the exact visited-state snapshot."""

    isolated_branch = getattr(env, "isolated_rollout_branch", None)
    if not callable(isolated_branch):
        raise TypeError(
            "Oracle-shadow collection requires env.isolated_rollout_branch() native isolation"
        )
    futures: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    valid = np.ones((int(env.num_envs),), dtype=np.bool_)
    oracle_actions = initial_oracle_actions

    branch_context = cast(Callable[[], ContextManager[None]], isolated_branch)
    with branch_context():
        for offset in range(config.prediction_horizon):
            actions.append(oracle_actions.copy())
            shadow_state = env.step(oracle_actions)
            valid &= ~_done_mask(shadow_state, num_envs=int(env.num_envs))
            shadow_obs = {key: np.asarray(value) for key, value in shadow_state.obs.items()}
            future = student_view(
                _obs_array(shadow_obs, observation_key), shadow_state.info,
                projection=student_projection,
                expected_student_obs_dim=config.obs_dim,
                student_drop_index=student_drop_index,
            )
            futures.append(future.copy())
            shadow_info = dict(shadow_state.info)
            shadow_command = _command_array(
                shadow_info,
                command_info_keys,
                expected_rows=int(env.num_envs),
                expected_dim=config.command_dim,
            )
            valid &= np.all(shadow_command == initial_command, axis=1)
            if offset + 1 < config.prediction_horizon:
                teacher_obs, _ = project_teacher_obs(
                    _obs_array(shadow_obs, observation_key),
                    projection=teacher_projection,
                    expected_teacher_obs_dim=int(
                        getattr(teacher_policy, "obs_dim", future.shape[1])
                    ),
                )
                oracle_actions = _oracle_actions(
                    teacher_policy,
                    shadow_obs,
                    shadow_info,
                    teacher_obs,
                    action_dim=config.action_dim,
                )
                if not bool(np.all(valid)):
                    oracle_actions = oracle_actions.copy()
                    oracle_actions[~valid] = 0.0

    return np.stack(futures, axis=1), np.stack(actions, axis=1), valid
