"""Persistent UniLab collector process for FADA Planner-IDM DAgger."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import torch
from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.distill.fada.async_collection import (
    FADA_ASYNC_SCENARIO,
    collect_fada_iteration,
)
from unilab.algos.torch.distill.fada.async_config import (
    allocate_fada_command_scenarios,
    curriculum_and_allocations,
    fada_runtime_device,
    stand_transition_curriculum_cfg,
    standing_owner_cfg,
    teacher_spec,
    v005_replay_cfg,
)
from unilab.algos.torch.distill.fada.checkpoint import load_fada_policy_checkpoint
from unilab.algos.torch.distill.fada.collection_contract import FADACollectionSpec
from unilab.algos.torch.distill.fada.model import FADA_COMMAND_SCENARIOS, FADAArchitectureConfig
from unilab.algos.torch.distill.fada.oracle import (
    load_fada_oracle_policy,
    reload_fada_oracle_policy_,
    validate_fada_oracle_environment_contract,
)
from unilab.algos.torch.distill.fada.source_plan import FADAPaperSourcePlan
from unilab.algos.torch.distill.runtime.async_runtime import (
    DaggerCollectRequest,
    DaggerCollectResult,
)
from unilab.algos.torch.distill.runtime.persistent_runtime import PersistentDistillationRuntime
from unilab.base.registry import ensure_registries
from unilab.ipc import SharedWeightSync
from unilab.training import BackendAdapter, create_env

# Compatibility aliases for callers of the former monolithic module.
_fada_runtime_device = fada_runtime_device
_teacher_spec = teacher_spec
_stand_transition_curriculum_cfg = stand_transition_curriculum_cfg
_v005_replay_cfg = v005_replay_cfg
_standing_owner_cfg = standing_owner_cfg
_curriculum_and_allocations = curriculum_and_allocations


class PersistentFADACollectorWorker:
    """Keep policies resident while isolating each native collection environment."""

    def __init__(
        self,
        *,
        root_dir: str,
        cfg_payload: Mapping[str, Any],
        standing_curriculum_enabled: bool,
        architecture: Mapping[str, Any],
        final_teacher_checkpoint: str,
        source_allocations: Sequence[tuple[str, int]],
        initial_checkpoint_path: str,
        device: str,
        weight_sync_name: str,
        weight_sync_lock: Any,
        weight_param_shapes: Mapping[str, torch.Size],
        env_factory: Callable[..., Any] = create_env,
        oracle_loader: Callable[..., torch.nn.Module] = load_fada_oracle_policy,
        intermediate_teacher_loader: Callable[..., torch.nn.Module] = load_fada_oracle_policy,
        intermediate_teacher_reloader: Callable[..., None] = reload_fada_oracle_policy_,
    ) -> None:
        # B1: 解析 worker-owned config 与 source identity, 产出不可变 collection 配置.
        self.weight_sync: SharedWeightSync | None = None
        self.env: Any | None = None
        self.standing_env: Any | None = None
        self.root_dir = Path(root_dir)
        self.cfg = OmegaConf.create(dict(cfg_payload))
        self.standing_curriculum_enabled = bool(standing_curriculum_enabled)
        self.config = FADAArchitectureConfig(**dict(architecture))
        self.device = str(device)
        self.source_allocations = tuple(
            (str(path), int(windows)) for path, windows in source_allocations
        )
        self.teacher_spec = _teacher_spec(self.cfg)
        self._oracle_loader = oracle_loader
        self._intermediate_teacher_loader = intermediate_teacher_loader
        self._intermediate_teacher_reloader = intermediate_teacher_reloader
        self.intermediate_teacher: torch.nn.Module | None = None
        self.intermediate_teacher_checkpoint: str | None = None
        self._env_factory = env_factory
        self._env_cfg_override: Mapping[str, Any] | None = None
        self._checkpoint_identity: Mapping[str, Any] | None = None
        self._physics_guard_max_abs = 1.0e4
        self._retired_physics_guard_trips = 0
        self._environment_in_use = False

        try:
            # B2: 恢复 student, shared-weight barrier 与 Oracle, 产出 resident policy resources.
            self.student = load_fada_policy_checkpoint(
                initial_checkpoint_path, device=self.device
            ).policy
            self.weight_sync = SharedWeightSync(
                dict(weight_param_shapes),
                create=False,
                shm_name=weight_sync_name,
                lock=weight_sync_lock,
            )
            self.local_weight_version = self.weight_sync.read_weights_into(
                self.student.state_dict()
            )
            self.final_teacher = self._oracle_loader(
                final_teacher_checkpoint,
                self.teacher_spec,
                device=self.device,
            )
            # A spawned worker starts with an empty registry. FADA owns the G1 route, so
            # importing unrelated optional robot families here would make their extras
            # accidental hard dependencies of Planner-IDM training.
            # B3: 创建唯一 G1WalkFlat environment; standing/transition 只改变 command scenario.
            ensure_registries(packages=("unilab.envs.locomotion.g1",))
            self._env_cfg_override = BackendAdapter(
                self.cfg,
                root_dir=self.root_dir,
                algo_name="distill",
            ).build_task_env_cfg_override()
            fada_cfg = self.cfg.training.fada
            checkpoint_identity = getattr(self.final_teacher, "checkpoint_identity", None)
            if self.teacher_spec.algo_type in {"privileged_locomotion_sac", "fada_official_ppo"}:
                if not isinstance(checkpoint_identity, Mapping):
                    raise ValueError(
                        "privileged FADA Oracle must expose sealed checkpoint_identity"
                    )
                self._checkpoint_identity = checkpoint_identity

            # B4: 每个 collection transaction 独占一个 native pool; 首个实例在 worker
            # 初始化时完成正式 Oracle/environment contract 验证，随后由同一 owner 重建.
            self._physics_guard_max_abs = float(
                OmegaConf.select(fada_cfg, "physics_guard_max_abs", default=1.0e4)
            )
            self.env = self._materialize_collection_environment()
            if self.standing_curriculum_enabled:
                self.standing_env = self.env
        except BaseException:
            self._close_resources(raise_errors=False)
            raise

    def _materialize_collection_environment(self) -> Any:
        if self._env_cfg_override is None:
            raise RuntimeError("FADA collection environment configuration is not initialized")
        environment = self._env_factory(
            self.cfg,
            num_envs=int(self.cfg.training.fada.num_envs),
            env_cfg_override=self._env_cfg_override,
            sim_backend=str(self.cfg.training.sim_backend),
            task_name=str(self.cfg.training.task_name),
        )
        try:
            if self._checkpoint_identity is not None:
                validate_fada_oracle_environment_contract(
                    self._checkpoint_identity,
                    environment,
                    self.cfg,
                )
            environment.set_physics_envelope_guard(self._physics_guard_max_abs)
            fada_cfg = self.cfg.training.fada
            needs_oracle_shadow = bool(
                OmegaConf.select(fada_cfg, "oracle_shadow_enabled", default=False)
            ) or bool(self.source_allocations)
            if needs_oracle_shadow:
                prepare = getattr(environment, "prepare_isolated_rollout_branch", None)
                if not callable(prepare):
                    raise TypeError(
                        "FADA Oracle-shadow collection requires an environment with "
                        "prepare_isolated_rollout_branch()"
                    )
                prepare()
            return environment
        except BaseException:
            close = getattr(environment, "close", None)
            if callable(close):
                close()
            raise

    def _retire_collection_environment(self, environment: Any) -> None:
        self._retired_physics_guard_trips += int(
            getattr(environment, "physics_guard_trip_count", 0)
        )
        try:
            close = getattr(environment, "close", None)
            if callable(close):
                close()
        finally:
            if self.env is environment:
                self.env = None
            if self.standing_env is environment:
                self.standing_env = None

    @contextmanager
    def collection_environment(self) -> Iterator[Any]:
        """Own one G1WalkFlat environment for exactly one source collection."""

        if self._environment_in_use:
            raise RuntimeError("FADA collection environments cannot be nested")
        if self.env is None:
            self.env = self._materialize_collection_environment()
        environment = self.env
        if self.standing_curriculum_enabled:
            self.standing_env = environment
        self._environment_in_use = True
        try:
            yield environment
        finally:
            try:
                self._retire_collection_environment(environment)
            finally:
                self._environment_in_use = False

    @property
    def physics_guard_trip_count(self) -> int:
        active = 0 if self.env is None else int(getattr(self.env, "physics_guard_trip_count", 0))
        return self._retired_physics_guard_trips + active

    def _close_resources(self, *, raise_errors: bool) -> None:
        """Close every materialized resident resource once, including partial construction."""

        errors: list[BaseException] = []
        if self.env is not None:
            try:
                self._retire_collection_environment(self.env)
            except BaseException as exc:
                errors.append(exc)
        self.intermediate_teacher = None
        self.intermediate_teacher_checkpoint = None
        if self.weight_sync is not None:
            try:
                self.weight_sync.close()
            except BaseException as exc:
                errors.append(exc)
            self.weight_sync = None
        if errors and raise_errors:
            raise errors[0]

    def _collection_spec(self) -> FADACollectionSpec:
        fada_cfg = self.cfg.training.fada
        command_keys = OmegaConf.to_container(fada_cfg.command_info_keys, resolve=True)
        if not isinstance(command_keys, list) or not command_keys:
            raise ValueError("training.fada.command_info_keys must be a non-empty list")
        return FADACollectionSpec(
            observation_key=str(fada_cfg.observation_key),
            teacher_projection=str(fada_cfg.teacher_projection),
            student_projection=str(fada_cfg.student_projection),
            student_drop_index=OmegaConf.select(fada_cfg, "student_drop_index"),
            command_info_keys=tuple(str(key) for key in command_keys),
            max_env_steps=OmegaConf.select(fada_cfg, "max_env_steps"),
        )

    def collect(self, request: DaggerCollectRequest) -> DaggerCollectResult:
        return collect_fada_iteration(self, request)

    def close(self) -> None:
        self._close_resources(raise_errors=True)


def _build_persistent_fada_worker(**kwargs: Any) -> PersistentFADACollectorWorker:
    return PersistentFADACollectorWorker(**kwargs)


def build_persistent_fada_runtime(
    *,
    cfg: DictConfig,
    architecture: FADAArchitectureConfig,
    paper_source_plan: FADAPaperSourcePlan,
    final_teacher_checkpoint: str | Path,
    request_timeout_seconds: float,
) -> PersistentDistillationRuntime:
    """Build the FADA-specific UniLab persistent collector runtime."""

    root_dir = Path(__file__).resolve().parents[6]
    cfg_payload = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_payload, dict):
        raise ValueError("composed FADA config must resolve to a mapping")

    def load_student(path: Path) -> torch.nn.Module:
        return load_fada_policy_checkpoint(path, device="cpu").policy

    curriculum, _ = _curriculum_and_allocations(cfg.training.fada, architecture)
    return PersistentDistillationRuntime(
        student_loader=load_student,
        worker_factory=_build_persistent_fada_worker,
        worker_kwargs={
            "root_dir": str(root_dir),
            "cfg_payload": cfg_payload,
            "standing_curriculum_enabled": bool(curriculum.enabled),
            "architecture": asdict(architecture),
            "final_teacher_checkpoint": str(Path(final_teacher_checkpoint).resolve()),
            "source_allocations": [
                (str(path.resolve()), windows)
                for path, windows in paper_source_plan.source_allocations
            ],
            "device": _fada_runtime_device(cfg),
        },
        worker_kwargs_factory=lambda checkpoint_path: {
            "initial_checkpoint_path": str(checkpoint_path.resolve())
        },
        request_timeout_seconds=float(request_timeout_seconds),
        worker_lifecycle="request",
    )


__all__ = [
    "FADA_COMMAND_SCENARIOS",
    "FADA_ASYNC_SCENARIO",
    "PersistentFADACollectorWorker",
    "allocate_fada_command_scenarios",
    "build_persistent_fada_runtime",
]
