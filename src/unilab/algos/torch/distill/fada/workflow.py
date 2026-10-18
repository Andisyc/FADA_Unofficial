"""FADA training composition root and compatibility facade."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.distill.fada.artifact_admission import (
    fada_quality_batch,
    require_fada_curriculum_artifact,
    slice_fada_batch,
)
from unilab.algos.torch.distill.fada.async_runtime import (
    FADA_ASYNC_SCENARIO,
    allocate_fada_command_scenarios,
)
from unilab.algos.torch.distill.fada.checkpoint import (
    load_fada_checkpoint,
    load_fada_policy_checkpoint,
    save_fada_checkpoint,
)
from unilab.algos.torch.distill.fada.collector import (
    FADACollectionSpec,
    collect_fada_source_windows,
)
from unilab.algos.torch.distill.fada.legacy_workflow import run_fada_legacy
from unilab.algos.torch.distill.fada.model import (
    FADA_SCENARIO_IDS,
    FADAArchitectureConfig,
    FADAPlannerIDMPolicy,
    FADASourceBatch,
)
from unilab.algos.torch.distill.fada.observation import (
    FADA_G1_STATE_OBSERVATION_CONTRACT,
    assert_fada_active_route_contract,
    assert_fada_projection_matches_contract,
)
from unilab.algos.torch.distill.fada.oracle import validate_loaded_fada_oracle_lineage
from unilab.algos.torch.distill.fada.persistent_workflow import run_fada_persistent_async
from unilab.algos.torch.distill.fada.replay import FADAReplayBuffer
from unilab.algos.torch.distill.fada.source_artifact import load_fada_source_batch
from unilab.algos.torch.distill.fada.source_evaluation import evaluate_fada_source_batch
from unilab.algos.torch.distill.fada.source_plan import (
    FADAPaperSourcePlan,
    build_fada_paper_source_plan,
)
from unilab.algos.torch.distill.fada.trainer import FADATrainer
from unilab.algos.torch.distill.fada.workflow_setup import (
    ROOT_DIR,
    FADAWorkflowDependencies,
    assert_fada_source_route_contract,
    assert_fada_training_run_contract,
    build_fada_architecture_config,
    distill_device,
    fada_execution_mode,
    fada_v005_replay_settings,
    paper_source_plan,
    resolve_fada_path,
)
from unilab.algos.torch.distill.runtime.async_runtime import DaggerCollectRequest

_distill_device = distill_device
_fada_execution_mode = fada_execution_mode
_fada_path = resolve_fada_path
_paper_source_plan = paper_source_plan
_fada_v005_replay_settings = fada_v005_replay_settings
_require_fada_curriculum_artifact = require_fada_curriculum_artifact
_slice_fada_batch = slice_fada_batch
_fada_quality_batch = fada_quality_batch
_run_fada_persistent_async = run_fada_persistent_async
_run_fada_legacy = run_fada_legacy

__all__ = [
    "DaggerCollectRequest",
    "FADA_ASYNC_SCENARIO",
    "FADA_G1_STATE_OBSERVATION_CONTRACT",
    "FADA_SCENARIO_IDS",
    "FADAArchitectureConfig",
    "FADACollectionSpec",
    "FADAPaperSourcePlan",
    "FADAPlannerIDMPolicy",
    "FADAReplayBuffer",
    "FADASourceBatch",
    "FADATrainer",
    "FADAWorkflowDependencies",
    "allocate_fada_command_scenarios",
    "assert_fada_active_route_contract",
    "assert_fada_projection_matches_contract",
    "assert_fada_source_route_contract",
    "build_fada_architecture_config",
    "build_fada_paper_source_plan",
    "collect_fada_source_windows",
    "evaluate_fada_source_batch",
    "load_fada_checkpoint",
    "load_fada_policy_checkpoint",
    "load_fada_source_batch",
    "run_fada_training_owner",
    "save_fada_checkpoint",
]


def run_fada_training_owner(
    cfg: DictConfig,
    *,
    teacher_checkpoint: str | Path | None = None,
    create_env_fn: Any | None = None,
    env_cfg_override_fn: Any | None = None,
    dependencies: FADAWorkflowDependencies,
) -> dict[str, Any]:
    """Assemble the UniLab Oracle/Planner-IDM DAgger training route."""

    # B1: 由唯一 FADA flag 配置族构造 model, Oracle, optimizer 与 resume cursor.
    if not bool(OmegaConf.select(cfg, "training.fada.enabled", default=False)):
        raise ValueError("run_fada_training requires training.fada.enabled=true")
    config = build_fada_architecture_config(cfg)
    fada_cfg = cfg.training.fada
    assert_fada_training_run_contract(cfg)
    assert_fada_source_route_contract(cfg, config)
    dependencies.require_teacher_policy_collection_route(cfg)
    dependencies.apply_collect_command_distribution_overrides(cfg)
    device = _distill_device(cfg)
    execution_mode = _fada_execution_mode(cfg)
    iterations = int(fada_cfg.iterations)
    windows_per_iteration = int(fada_cfg.windows_per_iteration)
    batch_size = int(fada_cfg.batch_size)
    _fada_v005_replay_settings(fada_cfg, batch_size=batch_size)
    if min(iterations, windows_per_iteration, batch_size) <= 0:
        raise ValueError("FADA iterations, windows_per_iteration, and batch_size must be positive")
    resolved_teacher = (
        Path(teacher_checkpoint)
        if teacher_checkpoint is not None
        else dependencies.resolve_teacher_checkpoint(cfg, root_dir=ROOT_DIR)[0]
    )
    if resolved_teacher is None:
        raise FileNotFoundError(
            "No Oracle checkpoint resolved for FADA training. Set teacher.checkpoint_path "
            "or teacher.load_run/teacher.checkpoint."
        )
    teacher_spec = dependencies.build_teacher_spec(cfg)
    if teacher_spec.algo_type == "fada_official_ppo":
        from unilab.algos.torch.fada_official.checkpoint_selection import configure_source_selection
        configure_source_selection(cfg, resolved_teacher)
    elif teacher_spec.algo_type == "fada_official_appo":
        from unilab.algos.torch.fada_official_appo.checkpoint_selection import configure_source_selection
        configure_source_selection(cfg, resolved_teacher)
    paper_source_plan = _paper_source_plan(cfg)
    loaded_final_oracle = dependencies.load_fada_oracle_policy(
        resolved_teacher, teacher_spec, device="cpu"
    )
    # B2: cold-path strict-load every intermediate Oracle before env/replay mutation.
    loaded_intermediate_oracles: list[torch.nn.Module] = []
    if paper_source_plan.enabled:
        for intermediate_path in paper_source_plan.checkpoint_paths:
            loaded_intermediate_oracles.append(
                dependencies.load_fada_oracle_policy(intermediate_path, teacher_spec, device="cpu")
            )
    source_behavior_profile = OmegaConf.select(
        fada_cfg, "source_behavior_profile", default=None
    )
    teacher_behavior_profile = OmegaConf.select(
        cfg, "teacher.behavior_profile", default=None
    )
    if source_behavior_profile != teacher_behavior_profile:
        raise ValueError(
            "FADA source and teacher behavior profiles must match: "
            f"source={source_behavior_profile!r} teacher={teacher_behavior_profile!r}"
        )
    validate_loaded_fada_oracle_lineage(
        loaded_final_oracle,
        loaded_intermediate_oracles,
        expected_behavior_profile=(
            None if teacher_behavior_profile is None else str(teacher_behavior_profile)
        ),
    )
    if teacher_spec.algo_type in {"fada_official_ppo", "fada_official_appo"}:
        if config.observation_contract != "g1_official_oracle_student66_v1":
            raise ValueError("Official Oracle requires its independent student contract")
        OmegaConf.update(cfg, "training.fada.official_oracle_identity",
                         loaded_final_oracle.checkpoint_identity, force_add=True)
    policy = FADAPlannerIDMPolicy(config).to(device)
    trainer = FADATrainer(
        policy,
        idm_optimizer=torch.optim.Adam(
            policy.idm.parameters(), lr=float(fada_cfg.idm_learning_rate)
        ),
        planner_optimizer=torch.optim.Adam(
            policy.planner.parameters(), lr=float(fada_cfg.planner_learning_rate)
        ),
        max_grad_norm=float(fada_cfg.max_grad_norm),
    )
    start_iteration = 0
    samples_seen = 0

    checkpoint_path = _fada_path(
        fada_cfg.checkpoint_path,
        field_name="training.fada.checkpoint_path",
        required=True,
    )
    if checkpoint_path is None:
        raise RuntimeError("FADA checkpoint path contract was not materialized")
    paper_retention_ratio = (
        int(fada_cfg.suboptimal_data_ratio)
        if execution_mode == "persistent_async"
        and paper_source_plan.enabled
        and bool(OmegaConf.select(fada_cfg, "v005_replay.enabled", default=False))
        else None
    )
    replay = FADAReplayBuffer(
        config,
        capacity=int(fada_cfg.replay_capacity),
        suboptimal_retention_ratio=paper_retention_ratio,
    )

    if execution_mode == "persistent_async":
        return _run_fada_persistent_async(
            cfg,
            config=config,
            paper_source_plan=paper_source_plan,
            resolved_teacher=resolved_teacher,
            checkpoint_path=checkpoint_path,
            policy=policy,
            trainer=trainer,
            replay=replay,
            start_iteration=start_iteration,
            samples_seen=samples_seen,
            dependencies=dependencies,
        )

    return _run_fada_legacy(
        cfg,
        config=config,
        paper_source_plan=paper_source_plan,
        resolved_teacher=resolved_teacher,
        teacher_spec=teacher_spec,
        checkpoint_path=checkpoint_path,
        policy=policy,
        trainer=trainer,
        replay=replay,
        start_iteration=start_iteration,
        samples_seen=samples_seen,
        dependencies=dependencies,
        create_env_fn=create_env_fn,
        env_cfg_override_fn=env_cfg_override_fn,
    )
