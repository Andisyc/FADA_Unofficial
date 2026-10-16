from __future__ import annotations

import math
import os
import time
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, ContextManager, Literal, Protocol, cast

import torch

from unilab.algos.torch.distill.fada.async_config import (
    curriculum_and_allocations,
    fada_training_schedule,
    v005_collection_profile_ratios,
    v005_replay_cfg,
)
from unilab.algos.torch.distill.fada.collection_contract import (
    FADACollectionResult,
    FADACollectionSpec,
)
from unilab.algos.torch.distill.fada.collection_transaction import collect_fada_source_windows
from unilab.algos.torch.distill.fada.model import (
    FADAArchitectureConfig,
    FADAPlannerIDMPolicy,
    FADASourceBatch,
)
from unilab.algos.torch.distill.fada.source_artifact import FADAShardedSourceWriter
from unilab.algos.torch.distill.runtime.async_runtime import (
    DaggerCollectRequest,
    DaggerCollectResult,
)

FADA_ASYNC_SCENARIO = "fada_iteration"


class FADAAsyncCollectorState(Protocol):
    """Resident resources required by one asynchronous collection transaction."""

    config: FADAArchitectureConfig
    device: str
    cfg: Any
    student: FADAPlannerIDMPolicy
    final_teacher: torch.nn.Module
    teacher_spec: Any
    source_allocations: tuple[tuple[str, int], ...]
    intermediate_teacher: torch.nn.Module | None
    intermediate_teacher_checkpoint: str | None
    local_weight_version: int
    weight_sync: Any
    _intermediate_teacher_loader: Any
    _intermediate_teacher_reloader: Any

    def _collection_spec(self) -> FADACollectionSpec: ...

    def collection_environment(self) -> ContextManager[Any]: ...

    @property
    def physics_guard_trip_count(self) -> int: ...


def _concat_source_batches(
    batches: Sequence[FADASourceBatch],
    config: FADAArchitectureConfig,
) -> FADASourceBatch:
    if not batches:
        raise ValueError("FADA async collector produced no source batches")
    return FADASourceBatch(
        **{
            field: torch.cat([getattr(batch, field) for batch in batches], dim=0)
            for field in FADASourceBatch.__dataclass_fields__
        }
    ).validate(config)


def _summary(
    collection: FADACollectionResult,
    *,
    iteration: int,
    source: str,
    source_checkpoint: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "iteration": int(iteration),
        "source": source,
        "rollout_mode": collection.rollout_mode,
        "windows": int(collection.batch.command.shape[0]),
        "env_steps": int(collection.env_steps),
        "rejected_done_transitions": int(collection.rejected_done_transitions),
        "rejected_command_windows": int(collection.rejected_command_windows),
        "rejected_scenario_windows": int(collection.rejected_scenario_windows),
        "command_scenario": collection.command_scenario,
        "oracle_role": ("walking" if source == "intermediate_oracle" else collection.oracle_role),
        "window_profile": collection.window_profile,
        "idm_source_role": (
            "oracle_shadow" if collection.rollout_mode == "oracle" else "trajectory"
        ),
    }
    if source_checkpoint is not None:
        result["source_checkpoint"] = source_checkpoint
    return result


def _require_exact_collection_rows(
    collection: FADACollectionResult,
    *,
    expected: int,
    scenario: str,
    profile: str,
) -> None:
    observed = int(collection.batch.command.shape[0])
    if observed != int(expected):
        raise RuntimeError(
            "FADA collector violated its exact window allocation before artifact write: "
            f"scenario={scenario!r} profile={profile!r} "
            f"expected={expected} observed={observed}"
        )


def _collect_cold_start_windows(
    env: Any,
    *,
    teacher_policy: torch.nn.Module,
    rollout_policy: FADAPlannerIDMPolicy | None,
    config: FADAArchitectureConfig,
    num_windows: int,
    spec: FADACollectionSpec,
    command_scenario: Literal["walk", "static_stand"],
) -> FADACollectionResult:
    """Collect reset/early-prefix windows in bounded batches from the resident env."""

    batches: list[FADASourceBatch] = []
    results: list[FADACollectionResult] = []
    remaining = int(num_windows)
    per_reset_capacity = int(env.num_envs) * (
        config.history_length if command_scenario == "walk" else 1
    )
    while remaining > 0:
        current = min(remaining, per_reset_capacity)
        result = collect_fada_source_windows(
            env,
            teacher_policy=teacher_policy,
            rollout_policy=rollout_policy,
            config=config,
            num_windows=current,
            spec=replace(spec, command_scenario=command_scenario, cold_start_windows=True),
        )
        batches.append(result.batch)
        results.append(result)
        remaining -= current
    return FADACollectionResult(
        batch=_concat_source_batches(batches, config),
        env_steps=sum(result.env_steps for result in results),
        rejected_done_transitions=sum(result.rejected_done_transitions for result in results),
        rejected_command_windows=sum(result.rejected_command_windows for result in results),
        rollout_mode=results[0].rollout_mode,
        command_scenario=command_scenario,
        oracle_role="unified",
        rejected_scenario_windows=sum(result.rejected_scenario_windows for result in results),
        window_profile="cold_start",
    )


def _collect_fada_iteration(
    worker: FADAAsyncCollectorState,
    request: DaggerCollectRequest,
    artifact_writer: FADAShardedSourceWriter,
) -> DaggerCollectResult:
    """在一个 weight-version barrier 内产出完整 scenario source artifact.

    函数名说明:
        该 worker owner 负责 collection 生命周期和 Oracle role 路由, 不更新 learner.

    主链路:
        上游: parent learner 发布的 DaggerCollectRequest 与 SharedWeightSync version.
        下游: schema-validated FADA source artifact, 随后由 parent replay consumer 读取.

    语义:
        main source 可分为 walk/static_stand/walk_to_stand; intermediate Oracle 永远只属于 walk.
    """

    # B1: 同步唯一 student weight version, 产出已校验的 scenario allocations.
    if request.scenario != FADA_ASYNC_SCENARIO:
        raise ValueError(f"unsupported FADA async scenario: {request.scenario!r}")
    if worker.weight_sync is None:
        raise RuntimeError("FADA collector worker is closed")
    started = time.perf_counter()
    worker.local_weight_version = worker.weight_sync.read_weights_into(worker.student.state_dict())
    sync_finished = time.perf_counter()
    fada_cfg = worker.cfg.training.fada
    training_schedule = fada_training_schedule(fada_cfg)
    student_rollout_enabled = training_schedule == "alternating_idm_then_planner"
    common = worker._collection_spec()

    curriculum, allocations = curriculum_and_allocations(fada_cfg, worker.config)
    curriculum_enabled = bool(curriculum.enabled)
    replay_cfg = v005_replay_cfg(fada_cfg)
    v005_enabled = bool(replay_cfg.enabled)
    cold_start_ratios = v005_collection_profile_ratios(fada_cfg)
    for scenario, ratio in cold_start_ratios.items():
        if v005_enabled and not math.isfinite(ratio):
            raise ValueError(f"v005 {scenario} cold-start ratio must be finite")
        if v005_enabled and not 0.0 < ratio < 1.0:
            raise ValueError(f"v005 {scenario} cold-start ratio must be strictly between 0 and 1")
    if not callable(getattr(worker, "collection_environment", None)):
        raise TypeError("FADA collector worker must own collection_environment()")

    # B2: 按 scenario-authoritative Oracle 收集 main source, 再追加 walking intermediate source.
    summaries: list[dict[str, Any]] = []
    main_windows = 0
    for scenario, scenario_windows in allocations:
        scenario_spec = replace(
            common,
            collect_oracle_shadow=bool(fada_cfg.oracle_shadow_enabled),
            transition_walk_command=tuple(float(value) for value in curriculum.walk_command),
            transition_pre_switch_steps=int(curriculum.pre_switch_steps),
            transition_post_switch_steps=int(curriculum.post_switch_steps),
            command_scenario=cast(Any, scenario),
        )
        profiles: tuple[tuple[bool, int], ...] = ((False, scenario_windows),)
        if v005_enabled and scenario in cold_start_ratios:
            cold_windows = int(math.floor(scenario_windows * cold_start_ratios[scenario] + 0.5))
            steady_windows = int(scenario_windows) - cold_windows
            if cold_windows <= 0 or steady_windows <= 0:
                raise ValueError(
                    f"v005 {scenario} allocation must contain cold-start and steady-state windows"
                )
            profiles = ((True, cold_windows), (False, steady_windows))
        for cold_start, profile_windows in profiles:
            with worker.collection_environment() as scenario_env:
                if cold_start:
                    main = _collect_cold_start_windows(
                        scenario_env,
                        teacher_policy=worker.final_teacher,
                        rollout_policy=(
                            worker.student
                            if student_rollout_enabled and request.iteration > 0
                            else None
                        ),
                        config=worker.config,
                        num_windows=profile_windows,
                        spec=scenario_spec,
                        command_scenario=cast(Literal["walk", "static_stand"], scenario),
                    )
                else:
                    main = collect_fada_source_windows(
                        scenario_env,
                        teacher_policy=worker.final_teacher,
                        rollout_policy=(
                            worker.student
                            if student_rollout_enabled and request.iteration > 0
                            else None
                        ),
                        config=worker.config,
                        num_windows=profile_windows,
                        spec=scenario_spec,
                    )
            _require_exact_collection_rows(
                main,
                expected=profile_windows,
                scenario=str(scenario),
                profile=("cold_start" if cold_start else "steady_state"),
            )
            artifact_writer.append(main.batch)
            main_windows += int(main.batch.command.shape[0])
            summaries.append(
                _summary(
                    main,
                    iteration=request.iteration,
                    source="optimal_or_current_policy",
                )
            )

    # One resident intermediate Oracle is reused across all source identities. Checkpoint
    # payloads stay CPU-owned and only their weights cross into the resident device model.
    for source_path, source_windows in worker.source_allocations:
        intermediate = getattr(worker, "intermediate_teacher", None)
        current_source = getattr(worker, "intermediate_teacher_checkpoint", None)
        if intermediate is None:
            intermediate = worker._intermediate_teacher_loader(
                source_path,
                worker.teacher_spec,
                device=worker.device,
            )
            worker.intermediate_teacher = intermediate
        elif current_source != source_path:
            worker._intermediate_teacher_reloader(intermediate, source_path, worker.teacher_spec)
        worker.intermediate_teacher_checkpoint = source_path
        with worker.collection_environment() as source_env:
            collection = collect_fada_source_windows(
                source_env,
                teacher_policy=worker.final_teacher,
                rollout_teacher_policy=intermediate,
                config=worker.config,
                num_windows=source_windows,
                spec=replace(
                    common,
                    collect_oracle_shadow=True,
                    planner_eligible=not v005_enabled,
                ),
            )
        artifact_writer.append(collection.batch)
        summaries.append(
            _summary(
                collection,
                iteration=request.iteration,
                source="intermediate_oracle",
                source_checkpoint=source_path,
            )
        )

    # B3: 原子提交只引用已落盘 shards 的 manifest, 不再构造 iteration 级大 tensor.
    collected = time.perf_counter()
    producer_pid = os.getpid()
    artifact_writer.commit(
        metadata={
            "iteration": request.iteration,
            "request_id": request.request_id,
            "scenario": request.scenario,
            "checkpoint_path": request.checkpoint_path,
            "expected_weight_version": request.expected_weight_version,
            "producer_pid": producer_pid,
            "training_schedule": training_schedule,
            "main_windows": main_windows,
            "stand_transition_curriculum_enabled": curriculum_enabled,
            "v005_replay_enabled": v005_enabled,
            "scenario_allocations": dict(allocations),
            "collections": summaries,
            **({"official_oracle_identity": worker.final_teacher.checkpoint_identity}
               if worker.config.observation_contract == "g1_official_oracle_student66_v1" else {}),
        },
    )
    written = time.perf_counter()
    return DaggerCollectResult(
        request_id=request.request_id,
        scenario=request.scenario,
        iteration=request.iteration,
        checkpoint_path=request.checkpoint_path,
        output_path=request.output_path,
        expected_weight_version=request.expected_weight_version,
        observed_weight_version=worker.local_weight_version,
        num_samples=artifact_writer.num_samples,
        worker_pid=producer_pid,
        metrics={
            "weight_sync_seconds": sync_finished - started,
            "collect_seconds": collected - sync_finished,
            "artifact_write_seconds": written - collected,
        },
        metadata={
            "main_windows": main_windows,
            "scenario_allocations": dict(allocations),
            "physics_guard_trips": float(getattr(worker, "physics_guard_trip_count", 0)),
        },
    )


def collect_fada_iteration(
    worker: FADAAsyncCollectorState,
    request: DaggerCollectRequest,
) -> DaggerCollectResult:
    """Collect one transaction with deterministic shard cleanup on every exit."""

    with FADAShardedSourceWriter(
        request.output_path,
        config=worker.config,
        replace_existing=True,
    ) as writer:
        return _collect_fada_iteration(worker, request, writer)
