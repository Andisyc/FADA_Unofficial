"""Walk/stand transition distillation collection."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from unilab.algos.torch.distill.collection.common import (
    _attach_collector_performance,
    _info_array,
    _module_device,
    _obs_array,
    _performance_span,
    _policy_actions,
    _reset_done_rows_after_step,
    _resolve_collection_reset,
    _state_done_mask,
    _target_height_array,
    project_student_obs,
    project_teacher_obs,
)
from unilab.algos.torch.distill.collection.transition_state import (
    TransitionRowBuffer,
    _build_transition_case_assignment,
    _build_transition_case_metadata,
    _validate_transition_coverage,
)
from unilab.algos.torch.distill.datasets.dataset import (
    DistillationTensorDataset,
    build_distillation_dataset,
)
from unilab.algos.torch.distill.observability.performance import (
    DistillationStageObservationAccumulator,
)


def set_transition_input_rows(
    env: Any,
    *,
    command_info_key: str,
    command_rows: np.ndarray,
    target_height_info_key: str | None = None,
    target_height_rows: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Atomically update command-owned transition inputs and refresh observations."""

    state = getattr(env, "state", None)
    info = getattr(state, "info", None)
    commands = info.get(command_info_key) if isinstance(info, Mapping) else None
    commands_np = np.asarray(commands) if commands is not None else None
    command_rows = np.asarray(command_rows, dtype=np.float32)
    if commands_np is None or commands_np.ndim != 2 or commands_np.shape[1] < 3:
        raise RuntimeError(
            "transition collection requires env.state.info[command_info_key] "
            "with shape (num_envs, >=3)"
        )
    if command_rows.shape != (commands_np.shape[0], 3):
        raise ValueError(
            "transition command rows shape mismatch: "
            f"expected {(commands_np.shape[0], 3)}, got {command_rows.shape}"
        )
    if not np.all(np.isfinite(command_rows)):
        raise ValueError("transition command rows must contain only finite values")

    normalized_target_key = (
        None if target_height_info_key in (None, "") else str(target_height_info_key)
    )
    if normalized_target_key is None and target_height_rows is not None:
        raise ValueError("transition target-height rows require target_height_info_key")
    target_height_np: np.ndarray | None = None
    target_rows_np: np.ndarray | None = None
    if normalized_target_key is not None and target_height_rows is not None:
        target_heights = info.get(normalized_target_key) if isinstance(info, Mapping) else None
        target_height_np = np.asarray(target_heights) if target_heights is not None else None
        target_rows_np = np.asarray(target_height_rows, dtype=np.float32)
        expected_target_shape = (commands_np.shape[0], 1)
        if target_height_np is None or target_height_np.shape != expected_target_shape:
            observed_shape = getattr(target_height_np, "shape", None)
            raise RuntimeError(
                "transition collection requires env.state.info[target_height_info_key] "
                f"with shape {expected_target_shape}, got {observed_shape}"
            )
        if target_rows_np.shape != expected_target_shape:
            raise ValueError(
                "transition target-height rows shape mismatch: "
                f"expected {expected_target_shape}, got {target_rows_np.shape}"
            )
        if not np.all(np.isfinite(target_rows_np)):
            raise ValueError("transition target-height rows must contain only finite values")

    # Validate both fields before mutating either input owner.
    command_setter = getattr(env, "set_external_command", None)
    if callable(command_setter) and info.get("student_observation_contract") == "g1_official_oracle_student66_v1":
        command_setter(command_rows)
    else:
        commands_np[:, :3] = command_rows.astype(commands_np.dtype, copy=False)
    if target_height_np is not None and target_rows_np is not None:
        target_height_np[:, :] = target_rows_np.astype(target_height_np.dtype, copy=False)
    refresh_state = getattr(env, "refresh_state", None)
    if not callable(refresh_state):
        raise RuntimeError("transition collection requires env.refresh_state()")
    refreshed = refresh_state()
    refreshed_state = refreshed if refreshed is not None else getattr(env, "state", None)
    if refreshed_state is None:
        raise RuntimeError("env.refresh_state() did not return or retain an env state")
    refreshed_obs = getattr(refreshed_state, "obs", None)
    refreshed_info = getattr(refreshed_state, "info", None)
    if not isinstance(refreshed_obs, dict) or not isinstance(refreshed_info, Mapping):
        raise RuntimeError("transition input refresh must return dict obs and info")
    observed_commands = np.asarray(refreshed_info.get(command_info_key))
    if (
        observed_commands.ndim != 2
        or observed_commands.shape[0] != command_rows.shape[0]
        or observed_commands.shape[1] < 3
        or not np.allclose(observed_commands[:, :3], command_rows, atol=1.0e-6, rtol=0.0)
    ):
        raise RuntimeError("transition command rows changed during observation refresh")
    if normalized_target_key is not None and target_rows_np is not None:
        observed_targets = np.asarray(refreshed_info.get(normalized_target_key))
        if observed_targets.shape != target_rows_np.shape or not np.allclose(
            observed_targets,
            target_rows_np,
            atol=1.0e-6,
            rtol=0.0,
        ):
            raise RuntimeError("transition target-height rows changed during observation refresh")
    return refreshed_obs, dict(refreshed_info)


@dataclass(frozen=True)
class _PreparedTransitionCollection:
    env: Any
    num_envs: int
    action_dim: int
    transition_cases: Any
    effective_max_env_steps: int
    obs: dict[str, Any]
    current_info: dict[str, Any]
    active_command_rows: np.ndarray
    zero_command_rows: np.ndarray


@dataclass(frozen=True)
class _TransitionCollectionOutcome:
    prepared: _PreparedTransitionCollection
    rows: TransitionRowBuffer
    env_steps: int
    switch_count: int
    post_switch_rows: int
    nominal_settle_rows: int
    height_tracking_rows: int
    done_seen_samples: int
    action_abs_max: float
    synthetic_teacher_tail: bool
    case_sample_counts: np.ndarray
    case_post_switch_counts: np.ndarray
    case_max_post_switch_ages: np.ndarray
    case_nominal_settle_counts: np.ndarray
    case_height_tracking_counts: np.ndarray
    case_max_height_tracking_ages: np.ndarray
    performance: DistillationStageObservationAccumulator | None
    teacher_inference_rows: int
    student_inference_rows: int


@dataclass(frozen=True)
class _TransitionStepLabels:
    student_obs: np.ndarray
    teacher_obs: np.ndarray
    teacher_actions: np.ndarray
    rollout_actions: np.ndarray
    commands: np.ndarray
    target_height: np.ndarray | None
    height_tracking: np.ndarray
    nominal_settling: np.ndarray
    synthetic_teacher_tail: bool
    teacher_inference_rows: int
    student_inference_rows: int


@dataclass
class _TransitionCollectionState:
    rows: TransitionRowBuffer
    post_switch: np.ndarray
    pre_age: np.ndarray
    transition_ages: np.ndarray
    case_sample_counts: np.ndarray
    case_post_switch_counts: np.ndarray
    case_max_post_switch_ages: np.ndarray
    case_nominal_settle_counts: np.ndarray
    case_height_tracking_counts: np.ndarray
    case_max_height_tracking_ages: np.ndarray
    performance: DistillationStageObservationAccumulator | None
    collected_count: int = 0
    env_steps: int = 0
    switch_count: int = 0
    post_switch_rows: int = 0
    nominal_settle_rows: int = 0
    height_tracking_rows: int = 0
    done_seen_samples: int = 0
    action_abs_max: float = 0.0
    synthetic_teacher_tail: bool = False
    teacher_inference_rows: int = 0
    student_inference_rows: int = 0

    @classmethod
    def create(
        cls,
        prepared: _PreparedTransitionCollection,
        *,
        performance_clock: Callable[[], float] | None,
    ) -> _TransitionCollectionState:
        num_envs = prepared.num_envs
        case_count = prepared.transition_cases.case_commands.shape[0]
        counts = np.zeros((case_count,), dtype=np.int64)
        return cls(
            rows=TransitionRowBuffer(),
            post_switch=np.zeros((num_envs,), dtype=np.bool_),
            pre_age=np.zeros((num_envs,), dtype=np.int64),
            transition_ages=np.full((num_envs,), -1, dtype=np.int64),
            case_sample_counts=counts,
            case_post_switch_counts=np.zeros_like(counts),
            case_max_post_switch_ages=np.full_like(counts, -1),
            case_nominal_settle_counts=np.zeros_like(counts),
            case_height_tracking_counts=np.zeros_like(counts),
            case_max_height_tracking_ages=np.full_like(counts, -1),
            performance=(
                None
                if performance_clock is None
                else DistillationStageObservationAccumulator(clock=performance_clock)
            ),
        )

    def record(
        self,
        prepared: _PreparedTransitionCollection,
        labels: _TransitionStepLabels,
        *,
        num_samples: int,
        nominal_settle_steps: int,
        walking_role_label: str,
        standing_role_label: str,
        scenario_label: str,
    ) -> None:
        take = min(int(num_samples) - self.collected_count, prepared.num_envs)
        with _performance_span(self.performance, "tensor_pack"):
            self.rows.append(
                student_obs=torch.as_tensor(labels.student_obs[:take], dtype=torch.float32).clone(),
                teacher_obs=torch.as_tensor(labels.teacher_obs[:take], dtype=torch.float32).clone(),
                teacher_actions=torch.as_tensor(
                    labels.teacher_actions[:take], dtype=torch.float32
                ).clone(),
                commands=torch.as_tensor(labels.commands[:take], dtype=torch.float32).clone(),
                target_height=(
                    None
                    if labels.target_height is None
                    else torch.as_tensor(labels.target_height[:take], dtype=torch.float32).clone()
                ),
                command_before=torch.as_tensor(
                    prepared.active_command_rows[:take], dtype=torch.float32
                ).clone(),
                command_after=torch.as_tensor(
                    labels.commands[:take], dtype=torch.float32
                ).clone(),
                transition_ages=torch.as_tensor(
                    self.transition_ages[:take], dtype=torch.int64
                ).clone(),
                role_labels=tuple(
                    str(standing_role_label) if value else str(walking_role_label)
                    for value in self.post_switch[:take]
                ),
                command_intents=tuple(
                    "inactive" if value else "active" for value in self.post_switch[:take]
                ),
                scenario_labels=tuple(str(scenario_label) for _ in range(take)),
            )
            case_indices = prepared.transition_cases.env_case_indices[:take]
            np.add.at(self.case_sample_counts, case_indices, 1)
            taken_post_switch = self.post_switch[:take]
            if bool(np.any(taken_post_switch)):
                post_indices = case_indices[taken_post_switch]
                post_ages = self.transition_ages[:take][taken_post_switch]
                np.add.at(self.case_post_switch_counts, post_indices, 1)
                np.maximum.at(self.case_max_post_switch_ages, post_indices, post_ages)
            nominal = labels.nominal_settling[:take]
            if bool(np.any(nominal)):
                np.add.at(self.case_nominal_settle_counts, case_indices[nominal], 1)
            tracking = labels.height_tracking[:take]
            if bool(np.any(tracking)):
                tracking_indices = case_indices[tracking]
                tracking_ages = (
                    self.transition_ages[:take][tracking] - int(nominal_settle_steps)
                )
                np.add.at(self.case_height_tracking_counts, tracking_indices, 1)
                np.maximum.at(
                    self.case_max_height_tracking_ages, tracking_indices, tracking_ages
                )
            self.post_switch_rows += int(np.count_nonzero(taken_post_switch))
            self.nominal_settle_rows += int(np.count_nonzero(nominal))
            self.height_tracking_rows += int(np.count_nonzero(tracking))
            self.collected_count += take
            self.action_abs_max = max(
                self.action_abs_max, float(np.max(np.abs(labels.rollout_actions)))
            )
        self.synthetic_teacher_tail = (
            self.synthetic_teacher_tail or labels.synthetic_teacher_tail
        )
        self.teacher_inference_rows += labels.teacher_inference_rows
        self.student_inference_rows += labels.student_inference_rows

    def outcome(self, prepared: _PreparedTransitionCollection) -> _TransitionCollectionOutcome:
        return _TransitionCollectionOutcome(
            prepared=prepared,
            rows=self.rows,
            env_steps=self.env_steps,
            switch_count=self.switch_count,
            post_switch_rows=self.post_switch_rows,
            nominal_settle_rows=self.nominal_settle_rows,
            height_tracking_rows=self.height_tracking_rows,
            done_seen_samples=self.done_seen_samples,
            action_abs_max=self.action_abs_max,
            synthetic_teacher_tail=self.synthetic_teacher_tail,
            case_sample_counts=self.case_sample_counts,
            case_post_switch_counts=self.case_post_switch_counts,
            case_max_post_switch_ages=self.case_max_post_switch_ages,
            case_nominal_settle_counts=self.case_nominal_settle_counts,
            case_height_tracking_counts=self.case_height_tracking_counts,
            case_max_height_tracking_ages=self.case_max_height_tracking_ages,
            performance=self.performance,
            teacher_inference_rows=self.teacher_inference_rows,
            student_inference_rows=self.student_inference_rows,
        )


def _label_transition_step(
    prepared: _PreparedTransitionCollection,
    *,
    obs: Mapping[str, Any],
    current_info: Mapping[str, Any],
    post_switch: np.ndarray,
    transition_ages: np.ndarray,
    expected_student_obs_dim: int,
    expected_teacher_obs_dim: int,
    walking_teacher_policy: torch.nn.Module,
    standing_teacher_policy: torch.nn.Module,
    rollout_policy: torch.nn.Module | None,
    rollout_policies_by_intent: Mapping[str, torch.nn.Module] | None,
    nominal_settle_steps: int,
    teacher_obs_key: str,
    teacher_projection: str,
    student_projection: str,
    student_drop_index: int | None,
    command_info_key: str,
    target_height_info_key: str | None,
    performance: DistillationStageObservationAccumulator | None,
) -> _TransitionStepLabels:
    source = _obs_array(obs, teacher_obs_key)
    teacher_obs, synthetic_tail = project_teacher_obs(
        source,
        projection=str(teacher_projection),
        expected_teacher_obs_dim=int(expected_teacher_obs_dim),
    )
    student_obs = project_student_obs(
        source,
        projection=str(student_projection),
        expected_student_obs_dim=int(expected_student_obs_dim),
        student_drop_index=student_drop_index,
    )
    commands = _info_array(current_info, str(command_info_key), expected_rows=prepared.num_envs)[
        :, :3
    ]
    target_height = (
        None
        if target_height_info_key in (None, "")
        else _target_height_array(
            current_info, str(target_height_info_key), expected_rows=prepared.num_envs
        )
    )
    height_tracking = post_switch & (transition_ages >= int(nominal_settle_steps))
    with _performance_span(performance, "teacher_inference"):
        walking_actions = _policy_actions(
            walking_teacher_policy,
            teacher_obs,
            action_dim=prepared.action_dim,
            policy_name="walking_teacher_policy",
        )
        standing_actions = _policy_actions(
            standing_teacher_policy,
            teacher_obs,
            action_dim=prepared.action_dim,
            policy_name="standing_teacher_policy",
        )
    teacher_actions = np.where(post_switch[:, None], standing_actions, walking_actions)
    with _performance_span(performance, "student_inference"):
        if rollout_policies_by_intent is None:
            if rollout_policy is None:
                raise RuntimeError("transition rollout policy contract was not materialized")
            rollout_actions = _policy_actions(
                rollout_policy,
                student_obs,
                action_dim=prepared.action_dim,
                policy_name="rollout_policy",
            )
            student_rows = int(student_obs.shape[0])
        else:
            active_actions = _policy_actions(
                rollout_policies_by_intent["active"],
                student_obs,
                action_dim=prepared.action_dim,
                policy_name="active_rollout_policy",
            )
            inactive_actions = _policy_actions(
                rollout_policies_by_intent["inactive"],
                student_obs,
                action_dim=prepared.action_dim,
                policy_name="inactive_rollout_policy",
            )
            rollout_actions = np.where(post_switch[:, None], inactive_actions, active_actions)
            student_rows = 2 * int(student_obs.shape[0])
    if not np.all(np.isfinite(teacher_actions)) or not np.all(np.isfinite(rollout_actions)):
        raise ValueError("transition collection produced non-finite actions")
    return _TransitionStepLabels(
        student_obs=student_obs,
        teacher_obs=teacher_obs,
        teacher_actions=teacher_actions,
        rollout_actions=rollout_actions,
        commands=commands,
        target_height=target_height,
        height_tracking=height_tracking,
        nominal_settling=post_switch & ~height_tracking,
        synthetic_teacher_tail=synthetic_tail,
        teacher_inference_rows=2 * int(teacher_obs.shape[0]),
        student_inference_rows=student_rows,
    )


def _advance_transition_step(
    prepared: _PreparedTransitionCollection,
    *,
    rollout_actions: np.ndarray,
    post_switch: np.ndarray,
    pre_age: np.ndarray,
    transition_ages: np.ndarray,
    pre_switch_steps: int,
    nominal_settle_steps: int,
    command_info_key: str,
    target_height_info_key: str | None,
    performance: DistillationStageObservationAccumulator | None,
) -> tuple[dict[str, Any], dict[str, Any], int, int]:
    with _performance_span(performance, "env_step"):
        state = prepared.env.step(rollout_actions)
    done_mask = _state_done_mask(state, expected_rows=prepared.num_envs)
    obs, info, done_count, _autoreset_count, _manual_reset_count = _reset_done_rows_after_step(
        prepared.env, state, num_envs=prepared.num_envs
    )
    previous_post_switch = post_switch.copy()
    post_switch[done_mask] = False
    pre_age[done_mask] = 0
    transition_ages[done_mask] = -1
    pre_age[(~previous_post_switch) & ~done_mask] += 1
    transition_ages[previous_post_switch & ~done_mask] += 1
    switch_mask = (~post_switch) & ~done_mask & (pre_age >= int(pre_switch_steps))
    post_switch[switch_mask] = True
    transition_ages[switch_mask] = 0
    height_switch_mask = (
        post_switch
        & ~done_mask
        & (transition_ages == int(nominal_settle_steps))
        & (int(nominal_settle_steps) > 0)
    )
    command_rows = prepared.active_command_rows.copy()
    command_rows[post_switch] = prepared.zero_command_rows[post_switch]
    target_height_rows = None
    cases = prepared.transition_cases
    if cases.nominal_target_rows is not None:
        if cases.post_switch_target_rows is None:
            raise RuntimeError("transition post-switch target rows unexpectedly missing")
        target_height_rows = cases.nominal_target_rows.copy()
        requested = post_switch & (transition_ages >= int(nominal_settle_steps))
        target_height_rows[requested] = cases.post_switch_target_rows[requested]
    if done_count > 0 or bool(np.any(switch_mask)) or bool(np.any(height_switch_mask)):
        obs, info = set_transition_input_rows(
            prepared.env,
            command_info_key=str(command_info_key),
            command_rows=command_rows,
            target_height_info_key=target_height_info_key,
            target_height_rows=target_height_rows,
        )
    return obs, info, int(np.count_nonzero(done_mask)), int(np.count_nonzero(switch_mask))


def _prepare_transition_collection(
    env: Any,
    *,
    num_samples: int,
    pre_switch_steps: int,
    nominal_settle_steps: int,
    min_post_switch_steps: int,
    walk_command: np.ndarray | tuple[float, float, float],
    walk_commands: Sequence[Sequence[float]] | np.ndarray | None,
    nominal_walk_target_height: float | None,
    post_switch_target_heights: Sequence[float] | np.ndarray | None,
    command_info_key: str,
    target_height_info_key: str | None,
    walking_role_label: str,
    standing_role_label: str,
    scenario_label: str,
    rollout_policy: torch.nn.Module | None,
    rollout_policies_by_intent: Mapping[str, torch.nn.Module] | None,
    max_env_steps: int | None,
    initial_reset: tuple[Any, Any] | None,
) -> _PreparedTransitionCollection:
    if int(num_samples) <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if int(pre_switch_steps) <= 0:
        raise ValueError(f"pre_switch_steps must be positive, got {pre_switch_steps}")
    if int(nominal_settle_steps) < 0:
        raise ValueError(f"nominal_settle_steps must be non-negative, got {nominal_settle_steps}")
    if int(min_post_switch_steps) < 0:
        raise ValueError(f"min_post_switch_steps must be non-negative, got {min_post_switch_steps}")
    if not str(walking_role_label) or not str(standing_role_label):
        raise ValueError("transition role labels must be non-empty")
    if not str(scenario_label):
        raise ValueError("transition scenario_label must be non-empty")
    action_shape = getattr(getattr(env, "action_space", None), "shape", None)
    if action_shape is None:
        raise ValueError("env.action_space.shape must be defined for transition collection")
    num_envs = int(getattr(env, "num_envs"))
    minimum_samples = int(num_envs) * (
        int(pre_switch_steps) + int(nominal_settle_steps) + int(min_post_switch_steps)
    )
    if int(min_post_switch_steps) > 0 and int(num_samples) < minimum_samples:
        raise ValueError(
            "transition collection requires enough samples to cover the configured "
            f"post-switch horizon: num_samples={int(num_samples)} "
            f"minimum={minimum_samples} num_envs={int(num_envs)} "
            f"pre_switch_steps={int(pre_switch_steps)} "
            f"nominal_settle_steps={int(nominal_settle_steps)} "
            f"min_post_switch_steps={int(min_post_switch_steps)}"
        )
    if rollout_policy is None and rollout_policies_by_intent is None:
        raise ValueError(
            "transition collection requires rollout_policy or rollout_policies_by_intent"
        )
    if rollout_policy is not None and rollout_policies_by_intent is not None:
        raise ValueError("transition collection accepts only one rollout policy contract")
    if rollout_policies_by_intent is not None:
        missing_intents = {"active", "inactive"} - set(rollout_policies_by_intent)
        if missing_intents:
            raise ValueError(
                f"rollout_policies_by_intent is missing intents: {sorted(missing_intents)}"
            )
    action_dim = int(action_shape[0])
    transition_cases = _build_transition_case_assignment(
        num_envs=num_envs,
        walk_command=walk_command,
        walk_commands=walk_commands,
        target_height_info_key=target_height_info_key,
        nominal_walk_target_height=nominal_walk_target_height,
        post_switch_target_heights=post_switch_target_heights,
    )
    if int(nominal_settle_steps) > 0 and transition_cases.nominal_target_rows is None:
        raise ValueError(
            "nominal_settle_steps requires nominal_walk_target_height and "
            "post_switch_target_heights"
        )
    effective_max_env_steps = (
        int(max_env_steps)
        if max_env_steps is not None
        else max(
            int(np.ceil(int(num_samples) / max(num_envs, 1)))
            * (int(pre_switch_steps) + int(nominal_settle_steps) + 16),
            1,
        )
    )
    if effective_max_env_steps < 1:
        raise ValueError(f"max_env_steps must be positive, got {effective_max_env_steps}")

    if getattr(env, "state", None) is None and callable(getattr(env, "init_state", None)):
        env.init_state()
    obs, current_info = _resolve_collection_reset(
        env,
        num_envs=num_envs,
        initial_reset=initial_reset,
    )
    active_command_rows = transition_cases.active_command_rows
    zero_command_rows = np.zeros((num_envs, 3), dtype=np.float32)
    obs, current_info = set_transition_input_rows(
        env,
        command_info_key=str(command_info_key),
        command_rows=active_command_rows,
        target_height_info_key=target_height_info_key,
        target_height_rows=transition_cases.nominal_target_rows,
    )

    return _PreparedTransitionCollection(
        env=env,
        num_envs=num_envs,
        action_dim=action_dim,
        transition_cases=transition_cases,
        effective_max_env_steps=effective_max_env_steps,
        obs=obs,
        current_info=current_info,
        active_command_rows=active_command_rows,
        zero_command_rows=zero_command_rows,
    )


def _collect_transition_rows(
    prepared: _PreparedTransitionCollection,
    *,
    num_samples: int,
    expected_student_obs_dim: int,
    expected_teacher_obs_dim: int,
    walking_teacher_policy: torch.nn.Module,
    standing_teacher_policy: torch.nn.Module,
    rollout_policy: torch.nn.Module | None,
    rollout_policies_by_intent: Mapping[str, torch.nn.Module] | None,
    pre_switch_steps: int,
    nominal_settle_steps: int,
    teacher_obs_key: str,
    teacher_projection: str,
    student_projection: str,
    student_drop_index: int | None,
    command_info_key: str,
    target_height_info_key: str | None,
    walking_role_label: str,
    standing_role_label: str,
    scenario_label: str,
    performance_clock: Callable[[], float] | None,
) -> _TransitionCollectionOutcome:
    effective_max_env_steps = prepared.effective_max_env_steps
    obs = prepared.obs
    current_info = prepared.current_info
    state = _TransitionCollectionState.create(
        prepared,
        performance_clock=performance_clock,
    )

    while state.collected_count < int(num_samples):
        labels = _label_transition_step(
            prepared,
            obs=obs,
            current_info=current_info,
            post_switch=state.post_switch,
            transition_ages=state.transition_ages,
            expected_student_obs_dim=expected_student_obs_dim,
            expected_teacher_obs_dim=expected_teacher_obs_dim,
            walking_teacher_policy=walking_teacher_policy,
            standing_teacher_policy=standing_teacher_policy,
            rollout_policy=rollout_policy,
            rollout_policies_by_intent=rollout_policies_by_intent,
            nominal_settle_steps=nominal_settle_steps,
            teacher_obs_key=teacher_obs_key,
            teacher_projection=teacher_projection,
            student_projection=student_projection,
            student_drop_index=student_drop_index,
            command_info_key=command_info_key,
            target_height_info_key=target_height_info_key,
            performance=state.performance,
        )
        rollout_actions = labels.rollout_actions
        state.record(
            prepared,
            labels,
            num_samples=int(num_samples),
            nominal_settle_steps=int(nominal_settle_steps),
            walking_role_label=walking_role_label,
            standing_role_label=standing_role_label,
            scenario_label=scenario_label,
        )
        if state.collected_count >= int(num_samples):
            break
        if state.env_steps >= effective_max_env_steps:
            raise RuntimeError(
                "transition collection exceeded max_env_steps before reaching "
                f"{num_samples} samples; collected={state.collected_count}"
            )

        obs, current_info, done_increment, switch_increment = _advance_transition_step(
            prepared,
            rollout_actions=rollout_actions,
            post_switch=state.post_switch,
            pre_age=state.pre_age,
            transition_ages=state.transition_ages,
            pre_switch_steps=pre_switch_steps,
            nominal_settle_steps=nominal_settle_steps,
            command_info_key=command_info_key,
            target_height_info_key=target_height_info_key,
            performance=state.performance,
        )
        state.done_seen_samples += done_increment
        state.switch_count += switch_increment
        state.env_steps += 1

    return state.outcome(prepared)


def _finalize_transition_collection(
    outcome: _TransitionCollectionOutcome,
    *,
    num_samples: int,
    expected_student_obs_dim: int,
    expected_teacher_obs_dim: int,
    nominal_settle_steps: int,
    min_post_switch_steps: int,
    nominal_walk_target_height: float | None,
    teacher_obs_key: str,
    teacher_projection: str,
    student_projection: str,
    student_drop_index: int | None,
    command_info_key: str,
    target_height_info_key: str | None,
    walking_role_label: str,
    standing_role_label: str,
    scenario_label: str,
    pre_switch_steps: int,
    rollout_policies_by_intent: Mapping[str, torch.nn.Module] | None,
    metadata: Mapping[str, Any] | None,
) -> DistillationTensorDataset:
    rows = outcome.rows
    prepared = outcome.prepared
    action_dim = prepared.action_dim
    transition_cases = prepared.transition_cases
    zero_command_rows = prepared.zero_command_rows
    env_steps = outcome.env_steps
    switch_count = outcome.switch_count
    post_switch_rows = outcome.post_switch_rows
    nominal_settle_rows = outcome.nominal_settle_rows
    height_tracking_rows = outcome.height_tracking_rows
    done_seen_samples = outcome.done_seen_samples
    action_abs_max = outcome.action_abs_max
    synthetic_teacher_tail = outcome.synthetic_teacher_tail
    case_sample_counts = outcome.case_sample_counts
    case_post_switch_counts = outcome.case_post_switch_counts
    case_max_post_switch_ages = outcome.case_max_post_switch_ages
    case_nominal_settle_counts = outcome.case_nominal_settle_counts
    case_height_tracking_counts = outcome.case_height_tracking_counts
    case_max_height_tracking_ages = outcome.case_max_height_tracking_ages
    performance = outcome.performance
    teacher_inference_rows = outcome.teacher_inference_rows
    student_inference_rows = outcome.student_inference_rows
    transition_ages_tensor, max_post_switch_age, max_height_tracking_age = (
        _validate_transition_coverage(
            rows=rows,
            num_samples=int(num_samples),
            switch_count=switch_count,
            post_switch_rows=post_switch_rows,
            nominal_settle_steps=int(nominal_settle_steps),
            min_post_switch_steps=int(min_post_switch_steps),
            case_height_tracking_counts=case_height_tracking_counts,
            case_max_height_tracking_ages=case_max_height_tracking_ages,
        )
    )
    transition_case_metadata = _build_transition_case_metadata(
        transition_cases=transition_cases,
        case_sample_counts=case_sample_counts,
        case_post_switch_counts=case_post_switch_counts,
        case_max_post_switch_ages=case_max_post_switch_ages,
        case_nominal_settle_counts=case_nominal_settle_counts,
        case_height_tracking_counts=case_height_tracking_counts,
        case_max_height_tracking_ages=case_max_height_tracking_ages,
    )
    payload = dict(metadata or {})
    payload.update(
        {
            "source": "live_env_transition_rollout",
            "scenario": str(scenario_label),
            "teacher_obs_key": str(teacher_obs_key),
            "teacher_projection": str(teacher_projection),
            "student_projection": str(student_projection),
            "student_drop_index": student_drop_index,
            "rollout_policy": (
                "command_intent_experts"
                if rollout_policies_by_intent is not None
                else "distillation_student"
            ),
            "pre_switch_steps": int(pre_switch_steps),
            "nominal_settle_steps": int(nominal_settle_steps),
            "height_switch_age": int(nominal_settle_steps),
            "min_post_switch_steps": int(min_post_switch_steps),
            "max_post_switch_age": int(max_post_switch_age),
            "max_height_tracking_age": int(max_height_tracking_age),
            "walk_command": transition_cases.walk_commands[0].tolist(),
            "walk_commands": transition_cases.walk_commands.tolist(),
            "nominal_walk_target_height": (
                None if nominal_walk_target_height is None else float(nominal_walk_target_height)
            ),
            "post_switch_target_heights": (
                None
                if transition_cases.post_switch_target_heights is None
                else transition_cases.post_switch_target_heights.tolist()
            ),
            "transition_case_count": len(transition_case_metadata),
            "transition_cases": transition_case_metadata,
            "zero_command": zero_command_rows[0].tolist(),
            "command_info_key": str(command_info_key),
            "target_height_info_key": (
                None if target_height_info_key in (None, "") else str(target_height_info_key)
            ),
            "walking_role_label": str(walking_role_label),
            "standing_role_label": str(standing_role_label),
            "env_steps": int(env_steps),
            "switch_count": int(switch_count),
            "post_switch_rows": int(post_switch_rows),
            "nominal_settle_rows": int(nominal_settle_rows),
            "height_tracking_rows": int(height_tracking_rows),
            "done_seen_samples": int(done_seen_samples),
            "action_abs_max": float(action_abs_max),
            "synthetic_teacher_tail": bool(synthetic_teacher_tail),
        }
    )
    with _performance_span(performance, "tensor_pack"):
        dataset = build_distillation_dataset(
            torch.cat(rows.student_obs, dim=0)[: int(num_samples)],
            torch.cat(rows.teacher_obs, dim=0)[: int(num_samples)],
            expected_student_obs_dim=int(expected_student_obs_dim),
            expected_teacher_obs_dim=int(expected_teacher_obs_dim),
            expected_teacher_action_dim=action_dim,
            metadata=payload,
            role_labels=tuple(rows.role_labels[: int(num_samples)]),
            teacher_actions=torch.cat(rows.teacher_actions, dim=0)[: int(num_samples)],
            commands=torch.cat(rows.commands, dim=0)[: int(num_samples)],
            target_height=(
                torch.cat(rows.target_height, dim=0)[: int(num_samples)]
                if rows.target_height
                else None
            ),
            command_intents=tuple(rows.command_intents[: int(num_samples)]),
            scenario_labels=tuple(rows.scenario_labels[: int(num_samples)]),
            transition_ages=transition_ages_tensor,
            command_before=torch.cat(rows.command_before, dim=0)[: int(num_samples)],
            command_after=torch.cat(rows.command_after, dim=0)[: int(num_samples)],
        )
    return _attach_collector_performance(
        dataset,
        accumulator=performance,
        teacher_inference_rows=teacher_inference_rows,
        student_inference_rows=student_inference_rows,
        env_steps=env_steps,
    )


def collect_transition_distillation_dataset_from_env(
    env: Any,
    *,
    num_samples: int,
    expected_student_obs_dim: int,
    expected_teacher_obs_dim: int,
    walking_teacher_policy: torch.nn.Module,
    standing_teacher_policy: torch.nn.Module,
    rollout_policy: torch.nn.Module | None = None,
    rollout_policies_by_intent: Mapping[str, torch.nn.Module] | None = None,
    pre_switch_steps: int = 8,
    nominal_settle_steps: int = 0,
    min_post_switch_steps: int = 0,
    walk_command: np.ndarray | tuple[float, float, float] = (0.4, 0.0, 0.0),
    walk_commands: Sequence[Sequence[float]] | np.ndarray | None = None,
    nominal_walk_target_height: float | None = None,
    post_switch_target_heights: Sequence[float] | np.ndarray | None = None,
    teacher_obs_key: str = "obs",
    teacher_projection: str = "identity",
    student_projection: str = "identity",
    student_drop_index: int | None = None,
    command_info_key: str = "commands",
    target_height_info_key: str | None = None,
    walking_role_label: str = "walk_flat",
    standing_role_label: str = "stand",
    scenario_label: str = "walk_to_stop",
    max_env_steps: int | None = None,
    metadata: Mapping[str, Any] | None = None,
    initial_reset: tuple[Any, Any] | None = None,
    performance_clock: Callable[[], float] | None = None,
) -> DistillationTensorDataset:
    """Collect one opt-in walk-to-stop student-state DAgger scenario."""

    prepared = _prepare_transition_collection(
        env,
        num_samples=num_samples,
        pre_switch_steps=pre_switch_steps,
        nominal_settle_steps=nominal_settle_steps,
        min_post_switch_steps=min_post_switch_steps,
        walk_command=walk_command,
        walk_commands=walk_commands,
        nominal_walk_target_height=nominal_walk_target_height,
        post_switch_target_heights=post_switch_target_heights,
        command_info_key=command_info_key,
        target_height_info_key=target_height_info_key,
        walking_role_label=walking_role_label,
        standing_role_label=standing_role_label,
        scenario_label=scenario_label,
        rollout_policy=rollout_policy,
        rollout_policies_by_intent=rollout_policies_by_intent,
        max_env_steps=max_env_steps,
        initial_reset=initial_reset,
    )
    outcome = _collect_transition_rows(
        prepared,
        num_samples=num_samples,
        expected_student_obs_dim=expected_student_obs_dim,
        expected_teacher_obs_dim=expected_teacher_obs_dim,
        walking_teacher_policy=walking_teacher_policy,
        standing_teacher_policy=standing_teacher_policy,
        rollout_policy=rollout_policy,
        rollout_policies_by_intent=rollout_policies_by_intent,
        pre_switch_steps=pre_switch_steps,
        nominal_settle_steps=nominal_settle_steps,
        teacher_obs_key=teacher_obs_key,
        teacher_projection=teacher_projection,
        student_projection=student_projection,
        student_drop_index=student_drop_index,
        command_info_key=command_info_key,
        target_height_info_key=target_height_info_key,
        walking_role_label=walking_role_label,
        standing_role_label=standing_role_label,
        scenario_label=scenario_label,
        performance_clock=performance_clock,
    )
    return _finalize_transition_collection(
        outcome,
        num_samples=num_samples,
        expected_student_obs_dim=expected_student_obs_dim,
        expected_teacher_obs_dim=expected_teacher_obs_dim,
        nominal_settle_steps=nominal_settle_steps,
        min_post_switch_steps=min_post_switch_steps,
        nominal_walk_target_height=nominal_walk_target_height,
        teacher_obs_key=teacher_obs_key,
        teacher_projection=teacher_projection,
        student_projection=student_projection,
        student_drop_index=student_drop_index,
        command_info_key=command_info_key,
        target_height_info_key=target_height_info_key,
        walking_role_label=walking_role_label,
        standing_role_label=standing_role_label,
        scenario_label=scenario_label,
        pre_switch_steps=pre_switch_steps,
        rollout_policies_by_intent=rollout_policies_by_intent,
        metadata=metadata,
    )
