from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch

from unilab.algos.torch.distill.collection.common import (
    project_teacher_obs,
)
from unilab.algos.torch.distill.collection.transition import set_transition_input_rows
from unilab.algos.torch.distill.fada.collection_views import student_view
from unilab.algos.torch.distill.fada.collection_contract import (
    FADACollectionResult,
    FADACollectionSpec,
    FADACollectionTransition,
)
from unilab.algos.torch.distill.fada.collection_io import (
    _command_array,
    _done_mask,
    _fada_actions,
    _next_after_done,
    _obs_array,
    _oracle_actions,
    _oracle_shadow_pair,
    _policy_actions,
)
from unilab.algos.torch.distill.fada.collection_state import FADAWindowAccumulator
from unilab.algos.torch.distill.fada.collection_windows import (
    _cold_start_window_from_records,
    _concat_batches,
    _terminal_planner_window,
    _walking_recovery_window,
    _window_from_records,
)
from unilab.algos.torch.distill.fada.model import (
    FADAArchitectureConfig,
    FADAPlannerIDMPolicy,
    FADASourceBatch,
)
from unilab.algos.torch.distill.fada.observation import assert_fada_projection_matches_contract
from unilab.algos.torch.distill.fada.training_diagnostics import FADACollectionProgressReporter

_BATCH_COMPACTION_SIZE = 256


def _default_collection_step_limit(
    *,
    num_windows: int,
    num_envs: int,
    record_count: int,
) -> int:
    """Return a conservative cap without assuming >=10% row acceptance."""

    ideal_parallel_steps = int(np.ceil(int(num_windows) / max(int(num_envs), 1)))
    legacy_headroom = int(record_count) + ideal_parallel_steps * 10
    one_window_per_vector_step = int(record_count) + int(num_windows)
    return max(legacy_headroom, one_window_per_vector_step, 1)


@dataclass(frozen=True)
class _FADAStepLabels:
    teacher_obs: np.ndarray
    student_obs: np.ndarray
    command: np.ndarray
    oracle_actions: np.ndarray
    oracle_future: np.ndarray
    oracle_action_chunk: np.ndarray
    oracle_shadow_valid: np.ndarray


@dataclass(frozen=True)
class _FADAEnvironmentStep:
    actions: np.ndarray
    previous_actions: np.ndarray
    done: np.ndarray
    next_obs: dict[str, Any]
    next_info: dict[str, Any]
    next_student_obs: np.ndarray


@dataclass
class FADACollectionTransaction:
    env: Any
    teacher_policy: torch.nn.Module
    config: FADAArchitectureConfig
    num_windows: int
    rollout_policy: FADAPlannerIDMPolicy | None
    rollout_teacher_policy: torch.nn.Module | None
    spec: FADACollectionSpec
    observation_key: str
    teacher_projection: str
    student_projection: str
    student_drop_index: int | None
    command_info_keys: tuple[str, ...]
    collect_oracle_shadow: bool
    command_scenario: str
    transition_walk_command: tuple[float, float, float]
    planner_eligible: bool
    cold_start_windows: bool
    idm_source_role: int
    walking_recovery: bool
    num_envs: int
    obs: dict[str, Any]
    info: dict[str, Any]
    transition_phase_step: int
    transition_pre_steps: int
    transition_post_steps: int
    observation_history: np.ndarray
    action_history: np.ndarray
    record_count: int
    records: list[deque[FADACollectionTransition]]
    episode_ids: np.ndarray
    episode_timesteps: np.ndarray
    windows: FADAWindowAccumulator
    rollout_mode: str
    step_limit: int
    reporter: FADACollectionProgressReporter
    rejected_done: int = 0
    rejected_command: int = 0
    rejected_scenario: int = 0
    env_steps: int = 0

    def run(self) -> FADACollectionResult:
        while self.windows.window_count < self.num_windows:
            self._ensure_step_budget()
            labels = self._query_same_state_labels()
            if self._append_walking_recovery(labels):
                break
            step = self._step_environment(labels)
            self._admit_windows(labels, step)
            self._advance_history(step)
        return FADACollectionResult(
            batch=self.windows.finalize(),
            env_steps=self.env_steps,
            rejected_done_transitions=self.rejected_done,
            rejected_command_windows=self.rejected_command,
            rollout_mode=self.rollout_mode,
            command_scenario=self.command_scenario,
            oracle_role="unified",
            rejected_scenario_windows=self.rejected_scenario,
            window_profile=("cold_start" if self.cold_start_windows else "steady_state"),
        )

    def _ensure_step_budget(self) -> None:
        if self.env_steps < self.step_limit:
            return
        self._report(force=True)
        attempted_rows = self.env_steps * self.num_envs
        acceptance = (
            0.0 if attempted_rows <= 0 else 100.0 * self.windows.window_count / attempted_rows
        )
        raise RuntimeError(
            f"FADA collector produced {self.windows.window_count}/{self.num_windows} windows after "
            f"{self.env_steps} env steps (acceptance={acceptance:.2f}%, "
            f"rejected_done={self.rejected_done}, rejected_command={self.rejected_command}, "
            f"rejected_scenario={self.rejected_scenario}); increase max_env_steps or "
            "inspect command resets"
        )

    def _query_same_state_labels(self) -> _FADAStepLabels:
        source = _obs_array(self.obs, self.observation_key)
        teacher_obs, _ = project_teacher_obs(
            source,
            projection=self.teacher_projection,
            expected_teacher_obs_dim=int(getattr(self.teacher_policy, "obs_dim", source.shape[1])),
        )
        student_obs = student_view(
            source, self.info,
            projection=self.student_projection,
            expected_student_obs_dim=self.config.obs_dim,
            student_drop_index=self.student_drop_index,
        )
        command = _command_array(
            self.info,
            self.command_info_keys,
            expected_rows=self.num_envs,
            expected_dim=self.config.command_dim,
        )
        authoritative_teacher_obs, _ = project_teacher_obs(
            source,
            projection=self.teacher_projection,
            expected_teacher_obs_dim=int(getattr(self.teacher_policy, "obs_dim", source.shape[1])),
        )
        oracle_actions = _oracle_actions(
            self.teacher_policy,
            self.obs,
            self.info,
            authoritative_teacher_obs,
            action_dim=self.config.action_dim,
        )
        if self.collect_oracle_shadow:
            oracle_future, oracle_action_chunk, oracle_shadow_valid = _oracle_shadow_pair(
                self.env,
                teacher_policy=self.teacher_policy,
                initial_oracle_actions=oracle_actions,
                initial_command=command,
                config=self.config,
                observation_key=self.observation_key,
                teacher_projection=self.teacher_projection,
                student_projection=self.student_projection,
                student_drop_index=self.student_drop_index,
                command_info_keys=self.command_info_keys,
            )
        else:
            oracle_future = np.zeros(
                (self.num_envs, self.config.prediction_horizon, self.config.obs_dim),
                dtype=np.float32,
            )
            oracle_action_chunk = np.zeros(
                (self.num_envs, self.config.prediction_horizon, self.config.action_dim),
                dtype=np.float32,
            )
            oracle_shadow_valid = np.zeros((self.num_envs,), dtype=np.bool_)
        return _FADAStepLabels(
            teacher_obs=teacher_obs,
            student_obs=student_obs,
            command=command,
            oracle_actions=oracle_actions,
            oracle_future=oracle_future,
            oracle_action_chunk=oracle_action_chunk,
            oracle_shadow_valid=oracle_shadow_valid,
        )

    def _append_walking_recovery(self, labels: _FADAStepLabels) -> bool:
        if not self.walking_recovery:
            return False
        for index in range(self.num_envs):
            if self.windows.window_count >= self.num_windows:
                break
            if int(self.episode_timesteps[index]) < self.config.history_length and bool(
                labels.oracle_shadow_valid[index]
            ):
                self.windows.append(
                    _walking_recovery_window(
                        index=index,
                        observation_history=self.observation_history,
                        action_history=self.action_history,
                        command=labels.command,
                        oracle_future=labels.oracle_future,
                        oracle_action_chunk=labels.oracle_action_chunk,
                        oracle_first_action=labels.oracle_actions,
                        config=self.config,
                        planner_eligible=self.planner_eligible,
                    )
                )
        if self.windows.window_count < self.num_windows:
            return False
        self._report()
        return True

    def _step_environment(self, labels: _FADAStepLabels) -> _FADAEnvironmentStep:
        if self.rollout_teacher_policy is not None:
            actions = _oracle_actions(
                self.rollout_teacher_policy,
                self.obs,
                self.info,
                labels.teacher_obs,
                action_dim=self.config.action_dim,
            )
        elif self.rollout_policy is not None:
            actions = _fada_actions(
                self.rollout_policy,
                self.observation_history,
                self.action_history,
                labels.command,
            )
        else:
            actions = labels.oracle_actions
        previous_actions = self.action_history[:, -1].copy()
        state = self.env.step(actions)
        self.env_steps += 1
        done = _done_mask(state, num_envs=self.num_envs)
        next_obs, next_info = _next_after_done(self.env, state, done)
        next_student_obs = self._project_student(next_obs, next_info)
        next_obs, next_info, next_student_obs = self._apply_scenario_command(
            next_obs, next_info, next_student_obs
        )
        return _FADAEnvironmentStep(
            actions=actions,
            previous_actions=previous_actions,
            done=done,
            next_obs=dict(next_obs),
            next_info=dict(next_info),
            next_student_obs=next_student_obs,
        )

    def _apply_scenario_command(
        self,
        next_obs: Mapping[str, Any],
        next_info: Mapping[str, Any],
        next_student_obs: np.ndarray,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any], np.ndarray]:
        if self.command_scenario == "static_stand":
            next_obs, next_info = set_transition_input_rows(
                self.env,
                command_info_key="commands",
                command_rows=np.zeros((self.num_envs, 3), dtype=np.float32),
            )
            next_student_obs = self._project_student(next_obs, next_info)
        elif self.command_scenario == "walk_to_stand":
            cycle_steps = self.transition_pre_steps + self.transition_post_steps
            self.transition_phase_step = (self.transition_phase_step + 1) % cycle_steps
            next_command = np.zeros((self.num_envs, 3), dtype=np.float32)
            if self.transition_phase_step < self.transition_pre_steps:
                next_command[:] = np.asarray(self.transition_walk_command, dtype=np.float32)
            next_obs, next_info = set_transition_input_rows(
                self.env,
                command_info_key="commands",
                command_rows=next_command,
            )
            next_student_obs = self._project_student(next_obs, next_info)
        return next_obs, next_info, next_student_obs

    def _admit_windows(self, labels: _FADAStepLabels, step: _FADAEnvironmentStep) -> None:
        for index in range(self.num_envs):
            if self.windows.window_count >= self.num_windows:
                break
            if bool(step.done[index]):
                self._admit_terminal_window(index, labels)
                self.records[index].clear()
                self.episode_ids[index] += 1
                self.episode_timesteps[index] = 0
                self.rejected_done += 1
                continue
            if self.walking_recovery:
                self.episode_timesteps[index] += 1
                continue
            self.records[index].append(
                FADACollectionTransition(
                    observation=labels.student_obs[index].copy(),
                    previous_action=step.previous_actions[index].copy(),
                    command=labels.command[index].copy(),
                    executed_action=step.actions[index].copy(),
                    oracle_action=labels.oracle_actions[index].copy(),
                    next_observation=step.next_student_obs[index].copy(),
                    episode_id=int(self.episode_ids[index]),
                    timestep=int(self.episode_timesteps[index]),
                    oracle_future=labels.oracle_future[index].copy(),
                    oracle_action_chunk=labels.oracle_action_chunk[index].copy(),
                    oracle_shadow_valid=bool(labels.oracle_shadow_valid[index]),
                )
            )
            self.episode_timesteps[index] += 1
            if len(self.records[index]) == self.record_count:
                self._admit_record_window(index)
                if self.windows.window_count >= self.num_windows:
                    break
        self._report()

    def _admit_terminal_window(self, index: int, labels: _FADAStepLabels) -> None:
        if not (
            self.windows.window_count < self.num_windows
            and self.planner_eligible
            and not self.cold_start_windows
            and int(self.episode_timesteps[index]) >= self.config.history_length - 1
        ):
            return
        self.windows.append(
            _terminal_planner_window(
                observation_history=self.observation_history[index : index + 1],
                action_history=self.action_history[index : index + 1],
                command=labels.command[index : index + 1],
                oracle_future=labels.oracle_future[index : index + 1],
                oracle_action_chunk=labels.oracle_action_chunk[index : index + 1],
                oracle_first_action=labels.oracle_actions[index : index + 1],
                config=self.config,
                command_scenario=self.command_scenario,
                planner_eligible=self.planner_eligible,
            )
        )

    def _admit_record_window(self, index: int) -> None:
        records = tuple(self.records[index])
        window = (
            _cold_start_window_from_records(
                records,
                self.config,
                planner_eligible=self.planner_eligible,
                idm_source_role=self.idm_source_role,
            )
            if self.cold_start_windows
            else _window_from_records(
                records,
                self.config,
                command_scenario=self.command_scenario,
                planner_eligible=self.planner_eligible,
                idm_source_role=self.idm_source_role,
            )
        )
        if window is not None:
            self.windows.append(window)
            return
        future_commands = (
            records
            if self.cold_start_windows
            else records[
                self.config.history_length - 1 : self.config.history_length
                - 1
                + self.config.prediction_horizon
            ]
        )
        if any(
            not np.array_equal(record.command, future_commands[0].command)
            for record in future_commands[1:]
        ):
            self.rejected_command += 1
        else:
            self.rejected_scenario += 1

    def _advance_history(self, step: _FADAEnvironmentStep) -> None:
        self.observation_history = np.roll(self.observation_history, shift=-1, axis=1)
        self.observation_history[:, -1] = step.next_student_obs
        self.action_history = np.roll(self.action_history, shift=-1, axis=1)
        self.action_history[:, -1] = step.actions
        if bool(np.any(step.done)):
            self.observation_history[step.done] = np.repeat(
                step.next_student_obs[step.done, None, :],
                self.config.history_length,
                axis=1,
            )
            self.action_history[step.done] = 0.0
        self.obs, self.info = step.next_obs, step.next_info

    def _project_student(self, obs: Mapping[str, Any], info: Mapping[str, Any]) -> np.ndarray:
        return student_view(
            _obs_array(obs, self.observation_key), info,
            projection=self.student_projection,
            expected_student_obs_dim=self.config.obs_dim,
            student_drop_index=self.student_drop_index,
        )

    def _report(self, *, force: bool = False) -> None:
        self.reporter.report(
            windows=self.windows.window_count,
            env_steps=self.env_steps,
            rejected_done=self.rejected_done,
            rejected_command=self.rejected_command,
            rejected_scenario=self.rejected_scenario,
            force=force,
        )


def _prepare_fada_collection(
    env: Any,
    *,
    teacher_policy: torch.nn.Module,
    config: FADAArchitectureConfig,
    num_windows: int,
    rollout_policy: FADAPlannerIDMPolicy | None,
    rollout_teacher_policy: torch.nn.Module | None,
    spec: FADACollectionSpec | None,
) -> FADACollectionTransaction:
    spec = FADACollectionSpec() if spec is None else spec
    assert_fada_projection_matches_contract(
        observation_contract=config.observation_contract,
        projection=spec.student_projection,
    )
    if int(num_windows) <= 0:
        raise ValueError(f"num_windows must be positive, got {num_windows}")
    if rollout_policy is not None and rollout_teacher_policy is not None:
        raise ValueError("set only one of rollout_policy and rollout_teacher_policy")
    if spec.command_scenario not in {"walk", "static_stand", "walk_to_stand"}:
        raise ValueError(f"unsupported FADA command_scenario: {spec.command_scenario!r}")
    if spec.command_scenario != "walk" and rollout_teacher_policy is not None:
        raise ValueError("intermediate Oracle rollouts support only the walk scenario")
    if spec.cold_start_windows and spec.command_scenario == "walk_to_stand":
        raise ValueError("cold_start_windows does not support command_scenario='walk_to_stand'")
    walking_recovery = bool(spec.cold_start_windows and spec.command_scenario == "walk")
    if walking_recovery and not spec.collect_oracle_shadow:
        raise ValueError("walking cold-start collection requires Oracle shadow supervision")
    num_envs = int(env.num_envs)
    reset_all = getattr(env, "reset_all", None)
    if callable(reset_all):
        state = reset_all()
        initial_obs = cast(Mapping[str, Any], getattr(state, "obs"))
        initial_info = cast(Mapping[str, Any], getattr(state, "info"))
        obs = {key: np.asarray(value).copy() for key, value in initial_obs.items()}
        info = dict(initial_info)
    else:
        if getattr(env, "state", None) is None and callable(getattr(env, "init_state", None)):
            env.init_state()
        obs, info = env.reset(np.arange(num_envs, dtype=np.int32))
        obs, info = dict(obs), dict(info)
    transition_pre_steps = (
        config.history_length
        if spec.transition_pre_switch_steps is None
        else int(spec.transition_pre_switch_steps)
    )
    transition_post_steps = (
        config.history_length + config.prediction_horizon
        if spec.transition_post_switch_steps is None
        else int(spec.transition_post_switch_steps)
    )
    if spec.command_scenario != "walk":
        if config.command_dim != 3 or list(spec.command_info_keys) != ["commands"]:
            raise ValueError(
                "standing curriculum requires command_dim=3 and command_info_keys=['commands']"
            )
        if transition_pre_steps < config.history_length:
            raise ValueError(
                "transition_pre_switch_steps must be at least history_length, "
                f"got {transition_pre_steps} < {config.history_length}"
            )
        if transition_post_steps < config.prediction_horizon:
            raise ValueError(
                "transition_post_switch_steps must be at least prediction_horizon, "
                f"got {transition_post_steps} < {config.prediction_horizon}"
            )
        walk_command = np.asarray(spec.transition_walk_command, dtype=np.float32)
        if walk_command.shape != (3,) or not bool(np.all(np.isfinite(walk_command))):
            raise ValueError("transition_walk_command must be a finite 3-D command")
        if not bool(np.any(np.abs(walk_command) > 1.0e-6)):
            raise ValueError("transition_walk_command must be active")
        command_rows = np.zeros((num_envs, 3), dtype=np.float32)
        if spec.command_scenario == "walk_to_stand":
            command_rows[:] = walk_command
        obs, info = set_transition_input_rows(
            env, command_info_key="commands", command_rows=command_rows
        )
    source = _obs_array(obs, spec.observation_key)
    student_obs = student_view(
        source, info,
        projection=spec.student_projection,
        expected_student_obs_dim=config.obs_dim,
        student_drop_index=spec.student_drop_index,
    )
    observation_history = np.repeat(student_obs[:, None, :], config.history_length, axis=1)
    action_history = np.zeros(
        (num_envs, config.history_length, config.action_dim), dtype=np.float32
    )
    record_count = (
        config.prediction_horizon
        if spec.cold_start_windows
        else config.history_length + config.prediction_horizon - 1
    )
    rollout_mode = (
        "intermediate_oracle"
        if rollout_teacher_policy is not None
        else ("oracle" if rollout_policy is None else "planner_idm")
    )
    step_limit = (
        int(spec.max_env_steps)
        if spec.max_env_steps is not None
        else _default_collection_step_limit(
            num_windows=int(num_windows), num_envs=num_envs, record_count=record_count
        )
    )
    reporter = FADACollectionProgressReporter(
        scenario=spec.command_scenario,
        window_profile=("cold_start" if spec.cold_start_windows else "steady_state"),
        rollout_mode=rollout_mode,
        target_windows=int(num_windows),
        num_envs=num_envs,
    )
    transaction = FADACollectionTransaction(
        env=env,
        teacher_policy=teacher_policy,
        config=config,
        num_windows=int(num_windows),
        rollout_policy=rollout_policy,
        rollout_teacher_policy=rollout_teacher_policy,
        spec=spec,
        observation_key=spec.observation_key,
        teacher_projection=spec.teacher_projection,
        student_projection=spec.student_projection,
        student_drop_index=spec.student_drop_index,
        command_info_keys=tuple(spec.command_info_keys),
        collect_oracle_shadow=spec.collect_oracle_shadow,
        command_scenario=spec.command_scenario,
        transition_walk_command=tuple(spec.transition_walk_command),
        planner_eligible=spec.planner_eligible,
        cold_start_windows=spec.cold_start_windows,
        idm_source_role=(1 if rollout_policy is None and rollout_teacher_policy is None else 0),
        walking_recovery=walking_recovery,
        num_envs=num_envs,
        obs=dict(obs),
        info=dict(info),
        transition_phase_step=0,
        transition_pre_steps=transition_pre_steps,
        transition_post_steps=transition_post_steps,
        observation_history=observation_history,
        action_history=action_history,
        record_count=record_count,
        records=[deque(maxlen=record_count) for _ in range(num_envs)],
        episode_ids=np.zeros((num_envs,), dtype=np.int64),
        episode_timesteps=np.zeros((num_envs,), dtype=np.int64),
        windows=FADAWindowAccumulator(
            config=config, compact_size=_BATCH_COMPACTION_SIZE, merge=_concat_batches
        ),
        rollout_mode=rollout_mode,
        step_limit=step_limit,
        reporter=reporter,
    )
    transaction._report()
    return transaction


def collect_fada_source_windows(
    env: Any,
    *,
    teacher_policy: torch.nn.Module,
    config: FADAArchitectureConfig,
    num_windows: int,
    rollout_policy: FADAPlannerIDMPolicy | None = None,
    rollout_teacher_policy: torch.nn.Module | None = None,
    spec: FADACollectionSpec | None = None,
) -> FADACollectionResult:
    """Collect causal source windows with same-state unified-Oracle labels."""

    return _prepare_fada_collection(
        env,
        teacher_policy=teacher_policy,
        config=config,
        num_windows=num_windows,
        rollout_policy=rollout_policy,
        rollout_teacher_policy=rollout_teacher_policy,
        spec=spec,
    ).run()
