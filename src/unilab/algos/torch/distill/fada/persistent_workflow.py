"""Persistent-async FADA learner unit of work."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.distill.fada.artifact_admission import (
    require_fada_curriculum_artifact,
)
from unilab.algos.torch.distill.fada.async_config import fada_training_schedule
from unilab.algos.torch.distill.fada.async_runtime import FADA_ASYNC_SCENARIO
from unilab.algos.torch.distill.fada.checkpoint import save_fada_checkpoint
from unilab.algos.torch.distill.fada.model import FADAArchitectureConfig, FADAPlannerIDMPolicy
from unilab.algos.torch.distill.fada.replay import FADAReplayBuffer
from unilab.algos.torch.distill.fada.replay_sampling import planner_sample_indices
from unilab.algos.torch.distill.fada.source_artifact import (
    LoadedFADASourceArtifact,
    open_fada_source_artifact,
    validate_fada_async_artifact_identity,
)
from unilab.algos.torch.distill.fada.source_evaluation import evaluate_fada_source_batch
from unilab.algos.torch.distill.fada.source_plan import FADAPaperSourcePlan
from unilab.algos.torch.distill.fada.trainer import FADATrainer
from unilab.algos.torch.distill.fada.training_diagnostics import print_fada_training_diagnostic
from unilab.algos.torch.distill.fada.workflow_setup import (
    FADAWorkflowDependencies,
    distill_device,
    fada_v005_replay_settings,
    resolve_fada_path,
)
from unilab.algos.torch.distill.runtime.async_runtime import (
    DaggerCollectRequest,
    DaggerCollectResult,
)


def _reuse_complete_artifact(
    artifact_path: Path,
    *,
    config: FADAArchitectureConfig,
    request: DaggerCollectRequest,
) -> tuple[DaggerCollectResult, LoadedFADASourceArtifact]:
    """Reuse only an immutable artifact with the exact repeated request identity."""

    loaded = open_fada_source_artifact(artifact_path, config=config)
    producer_pid = int(loaded.metadata.get("producer_pid", -1))
    validate_fada_async_artifact_identity(
        loaded.metadata,
        expected={
            "request_id": request.request_id,
            "scenario": request.scenario,
            "iteration": request.iteration,
            "checkpoint_path": request.checkpoint_path,
            "expected_weight_version": request.expected_weight_version,
            "producer_pid": producer_pid,
        },
    )
    return (
        DaggerCollectResult(
            request_id=request.request_id,
            scenario=request.scenario,
            iteration=request.iteration,
            checkpoint_path=request.checkpoint_path,
            output_path=str(artifact_path),
            expected_weight_version=request.expected_weight_version,
            observed_weight_version=request.expected_weight_version,
            num_samples=loaded.num_samples,
            worker_pid=producer_pid,
            metrics={"artifact_reused": 1.0},
            metadata={"artifact_reused": True},
        ),
        loaded,
    )


def _admit_fada_artifact(
    cfg: DictConfig,
    *,
    loaded: LoadedFADASourceArtifact,
    result: DaggerCollectResult,
    request: DaggerCollectRequest,
    training_schedule: str,
) -> tuple[int, list[Mapping[str, Any]], dict[str, torch.Tensor]]:
    """Run the complete pre-mutation admission for new and reusable artifacts."""

    if loaded.config.observation_contract == "g1_official_oracle_student66_v1":
        expected_identity = OmegaConf.to_container(
            cfg.training.fada.official_oracle_identity, resolve=True)
        if loaded.metadata.get("official_oracle_identity") != expected_identity:
            raise ValueError("Official student artifact teacher lineage/action identity mismatch")
    validate_fada_async_artifact_identity(
        loaded.metadata,
        expected={
            "request_id": request.request_id,
            "scenario": request.scenario,
            "iteration": request.iteration,
            "checkpoint_path": request.checkpoint_path,
            "expected_weight_version": request.expected_weight_version,
            "producer_pid": result.worker_pid,
        },
    )
    if loaded.metadata.get("training_schedule") != training_schedule:
        raise ValueError(
            "FADA async artifact training schedule mismatch: "
            f"expected={training_schedule!r} "
            f"observed={loaded.metadata.get('training_schedule')!r}"
        )
    if loaded.num_samples != result.num_samples:
        raise ValueError(
            "FADA async artifact/result sample mismatch: "
            f"artifact={loaded.num_samples} result={result.num_samples}"
        )
    main_windows = int(loaded.metadata.get("main_windows", 0))
    if main_windows <= 0 or main_windows > result.num_samples:
        raise ValueError(f"invalid FADA async main_windows={main_windows}")
    raw_summaries = loaded.metadata.get("collections")
    if not isinstance(raw_summaries, list) or not all(
        isinstance(item, Mapping) for item in raw_summaries
    ):
        raise ValueError("FADA async artifact collections must be a list of mappings")
    summaries = cast(list[Mapping[str, Any]], raw_summaries)
    identity = loaded.identity_fields()
    require_fada_curriculum_artifact(
        cfg,
        loaded.metadata,
        identity=identity,
    )
    return main_windows, summaries, identity


def _load_or_collect_admitted_artifact(
    cfg: DictConfig,
    *,
    runtime: Any,
    artifact_path: Path,
    config: FADAArchitectureConfig,
    request: DaggerCollectRequest,
    training_schedule: str,
) -> tuple[
    DaggerCollectResult,
    LoadedFADASourceArtifact,
    int,
    list[Mapping[str, Any]],
    dict[str, torch.Tensor],
]:
    """Reuse one fully admitted artifact, otherwise replace it with one collection."""

    if artifact_path.is_file():
        try:
            result, loaded = _reuse_complete_artifact(
                artifact_path,
                config=config,
                request=request,
            )
            admitted = _admit_fada_artifact(
                cfg,
                loaded=loaded,
                result=result,
                request=request,
                training_schedule=training_schedule,
            )
            return result, loaded, *admitted
        except (AttributeError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError):
            pass

    result = runtime.collect(request)
    loaded = open_fada_source_artifact(artifact_path, config=config)
    admitted = _admit_fada_artifact(
        cfg,
        loaded=loaded,
        result=result,
        request=request,
        training_schedule=training_schedule,
    )
    return result, loaded, *admitted


def run_fada_persistent_async(
    cfg: DictConfig,
    *,
    config: FADAArchitectureConfig,
    paper_source_plan: FADAPaperSourcePlan,
    resolved_teacher: Path,
    checkpoint_path: Path,
    policy: FADAPlannerIDMPolicy,
    trainer: FADATrainer,
    replay: FADAReplayBuffer,
    start_iteration: int,
    samples_seen: int,
    dependencies: FADAWorkflowDependencies,
) -> dict[str, Any]:
    """Run the learner in the parent and each FADA iteration in a fresh child."""

    fada_cfg = cfg.training.fada
    iterations = int(fada_cfg.iterations)
    training_schedule = fada_training_schedule(fada_cfg)
    v005_enabled, replay_sampling_spec = fada_v005_replay_settings(
        fada_cfg,
        batch_size=int(fada_cfg.batch_size),
    )
    runtime_config = cast(dict[str, Any], OmegaConf.to_container(fada_cfg, resolve=True))
    artifact_dir_value = OmegaConf.select(
        fada_cfg, "async_artifact_dir", default="logs/fada/source_batches"
    )
    artifact_dir = resolve_fada_path(
        artifact_dir_value,
        field_name="training.fada.async_artifact_dir",
        required=True,
    )
    if artifact_dir is None:
        raise RuntimeError("FADA async artifact directory was not materialized")

    # The first publication is a complete paired checkpoint even before update 0.
    save_fada_checkpoint(
        checkpoint_path,
        policy,
        trainer,
        completed_iterations=start_iteration,
        samples_seen=samples_seen,
        runtime_config=runtime_config,
    )
    runtime = dependencies.build_persistent_fada_runtime(
        cfg=cfg,
        architecture=config,
        paper_source_plan=paper_source_plan,
        final_teacher_checkpoint=resolved_teacher,
        request_timeout_seconds=float(
            OmegaConf.select(fada_cfg, "async_request_timeout_seconds", default=3600.0)
        ),
    )
    last_stats = None
    last_quality_metrics: dict[str, float] = {}
    collection_summaries: list[dict[str, Any]] = []
    try:
        weight_version = runtime.activate_checkpoint(checkpoint_path)
        for iteration in range(start_iteration, iterations):
            artifact_path = (artifact_dir / f"iteration_{iteration:04d}.pt").resolve()
            request = DaggerCollectRequest(
                request_id=f"fada-{iteration:04d}-v{weight_version}",
                scenario=FADA_ASYNC_SCENARIO,
                iteration=iteration,
                checkpoint_path=str(checkpoint_path.resolve()),
                output_path=str(artifact_path),
                expected_weight_version=weight_version,
            )
            result, loaded, main_windows, summaries, identity = (
                _load_or_collect_admitted_artifact(
                    cfg,
                    runtime=runtime,
                    artifact_path=artifact_path,
                    config=config,
                    request=request,
                    training_schedule=training_schedule,
                )
            )
            quality_batch = None
            if bool(fada_cfg.oracle_shadow_enabled):
                quality_limit = int(
                    OmegaConf.select(fada_cfg, "quality_eval_max_windows", default=4096)
                )
                quality_size = min(quality_limit, main_windows)
                quality_indices = (
                    planner_sample_indices(
                        quality_size,
                        planner_eligible=identity["planner_eligible"][:main_windows],
                        command_scenario=identity["command_scenario"][:main_windows],
                        cold_start=identity["cold_start"][:main_windows],
                        command=identity["command"][:main_windows],
                        sampling_spec=replay_sampling_spec,
                    )
                    if v005_enabled
                    else torch.arange(quality_size, dtype=torch.int64)
                )
                quality_batch = loaded.select_indices(quality_indices)
            collection_summaries.extend(cast(list[dict[str, Any]], summaries))
            replay.add_artifact(loaded)
            samples_seen += result.num_samples

            coverage = (
                replay.validate_sampling_spec(
                    replay_sampling_spec,
                    batch_size=int(fada_cfg.batch_size),
                )
                if v005_enabled
                else None
            )
            idm_updates = max(
                int(fada_cfg.idm_updates),
                0 if coverage is None else coverage.required_idm_updates,
            )
            planner_updates = max(
                int(fada_cfg.planner_updates),
                0 if coverage is None else coverage.required_planner_updates,
            )
            last_stats = trainer.update_from_replay(
                replay,
                batch_size=int(fada_cfg.batch_size),
                idm_updates=idm_updates,
                planner_updates=planner_updates,
                device=distill_device(cfg),
                replay_sampling_spec=replay_sampling_spec if v005_enabled else None,
            )
            if quality_batch is not None:
                last_quality_metrics = evaluate_fada_source_batch(
                    policy,
                    quality_batch,
                    require_scenario_metrics=v005_enabled,
                )
                last_quality_metrics.update(
                    {
                        "rollout_rejected_done_transitions": float(
                            sum(int(item["rejected_done_transitions"]) for item in summaries)
                        ),
                        "rollout_rejected_command_windows": float(
                            sum(int(item["rejected_command_windows"]) for item in summaries)
                        ),
                    }
                )
            del identity
            save_fada_checkpoint(
                checkpoint_path,
                policy,
                trainer,
                completed_iterations=iteration + 1,
                samples_seen=samples_seen,
                runtime_config=runtime_config,
                quality_metrics=last_quality_metrics,
            )
            print_fada_training_diagnostic(
                schedule=training_schedule,
                iteration=iteration,
                iterations=iterations,
                stats=last_stats,
                idm_updates=idm_updates,
                planner_updates=planner_updates,
                configured_idm_updates=int(fada_cfg.idm_updates),
                configured_planner_updates=int(fada_cfg.planner_updates),
                replay_coverage=coverage,
                replay_size=len(replay),
                samples_seen=samples_seen,
                collection_summaries=cast(list[dict[str, Any]], summaries),
                collector_metrics=cast(dict[str, Any], getattr(result, "metrics", {})),
                checkpoint_path=checkpoint_path,
            )
            if iteration + 1 < iterations:
                weight_version = runtime.activate_checkpoint(checkpoint_path)
    finally:
        runtime.close()

    replay_role_counts = replay.source_role_counts()
    return {
        "mode": f"fada_{training_schedule}_training",
        "training_schedule": training_schedule,
        "execution_mode": "persistent_async",
        "checkpoint_path": str(checkpoint_path),
        "completed_iterations": iterations,
        "samples_seen": samples_seen,
        "replay_size": len(replay),
        "replay_effective_capacity": replay.effective_capacity,
        "replay_role_counts": {
            "planner_eligible": replay_role_counts.planner_eligible,
            "planner_ineligible": replay_role_counts.planner_ineligible,
        },
        "last_idm_loss": None if last_stats is None else last_stats.idm_loss,
        "last_planner_loss": None if last_stats is None else last_stats.planner_loss,
        "quality_metrics": last_quality_metrics,
        "collections": collection_summaries,
    }
