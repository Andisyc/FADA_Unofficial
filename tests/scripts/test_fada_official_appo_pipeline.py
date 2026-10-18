"""Formal CPU pipeline with real updates/artifacts and only fake native physics.

This is entrypoint composition, not a shell CLI, throughput or gait-quality test.
"""
from copy import deepcopy
import json
import multiprocessing as mp
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from tests.scripts.test_fada_official_appo_consumers import (
    distill_cfg, native_factory, play_arguments,
)
from tests.scripts.test_fada_official_appo_training import configuration, script, check_clean
from unilab.algos.torch.fada_official_appo import checkpoint as cp
from unilab.algos.torch.distill.fada import async_runtime, persistent_workflow
from unilab.algos.torch.distill.fada.checkpoint import load_fada_policy_checkpoint
from unilab.algos.torch.distill.fada.model import FADA_SCENARIO_IDS
from unilab.algos.torch.distill.fada.source_artifact import open_fada_source_artifact
from unilab.algos.torch.distill.fada.workflow_setup import build_fada_architecture_config
from unilab.training import fada_official_appo as training
from unilab.training.backend_adapter import BackendAdapter
from unilab.training.rsl_rl import RslRlVecEnvWrapper
from unilab.visualization import playback_viewer as viewer
from unilab.visualization.playback_distill_sessions import create_fada_playback_session
from unilab.visualization.playback_sessions import RslRlPlaybackConfig

ROOT = Path(__file__).resolve().parents[2]


def collector_factory(env_name, *, num_envs, sim_backend, env_cfg_override):
    """Importable spawn seam: leave the official environment/reward owner intact."""
    assert mp.current_process().name != "MainProcess"
    assert env_name == "G1OfficialOracleFlat" and sim_backend == "mujoco"
    return native_factory(None, num_envs=num_envs, env_cfg_override=env_cfg_override)


def distillation_worker_factory(**kwargs):
    torch.set_num_threads(1)
    torch.manual_seed(43)
    np.random.seed(43)
    return async_runtime.PersistentFADACollectorWorker(**kwargs, env_factory=native_factory)


def changed(before, after, *, prefix=""):
    return any(not torch.equal(value, after[key]) for key, value in before.items()
               if key.startswith(prefix) and value.is_floating_point())


def test_formal_train_resume_distill_two_iterations_and_replay(tmp_path, monkeypatch, capsys):
    from scripts import train_distill

    initial_threads = torch.get_num_threads()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state()
    previous_children = {child.pid for child in mp.active_children()}
    monkeypatch.setenv("WANDB_MODE", "disabled")
    torch.set_num_threads(1)
    torch.manual_seed(43)
    np.random.seed(43)
    runners, runtimes, activations, publications = [], [], [], []
    admissions = []
    names = set()
    actual_runner = training.OfficialFadaAPPORunner
    actual_build_runtime = async_runtime.build_persistent_fada_runtime
    actual_save = persistent_workflow.save_fada_checkpoint
    actual_admit = persistent_workflow._admit_fada_artifact

    def construct(*args, **kwargs):
        runner = actual_runner(*args, **kwargs, env_factory=collector_factory)
        runners.append(runner)
        return runner

    def build_runtime(**kwargs):
        runtime = actual_build_runtime(**kwargs)
        runtimes.append(runtime)
        activate = runtime.activate_checkpoint

        def observe_activation(path):
            version = activate(path)
            names.add(runtime._weight_sync.name)
            activations.append(version)
            return version

        runtime.activate_checkpoint = observe_activation
        return runtime

    def observe_save(path, policy, trainer, **kwargs):
        result = actual_save(path, policy, trainer, **kwargs)
        publications.append((kwargs["completed_iterations"],
                             deepcopy(policy.state_dict())))
        return result

    def observe_admission(cfg, **kwargs):
        admitted = actual_admit(cfg, **kwargs)
        main_windows, _, identity = admitted
        main = {key: value[:main_windows] for key, value in identity.items()}
        steady_walk = (main["planner_eligible"] & ~main["cold_start"]
                       & (main["command_scenario"] == FADA_SCENARIO_IDS["walk"]))
        speed = torch.linalg.vector_norm(main["command"][steady_walk, :2], dim=1)
        counts = {"slow": int((speed < .25).sum()),
                  "medium": int(((speed >= .25) & (speed < .6)).sum()),
                  "high": int((speed >= .6).sum())}
        scenarios = {name: int((main["command_scenario"] == value).sum())
                     for name, value in FADA_SCENARIO_IDS.items()}
        result = kwargs["result"]
        record = {"iteration": kwargs["request"].iteration, "speed_bins": counts,
                  "scenarios": scenarios, "main_windows": main_windows,
                  "total_windows": result.num_samples}
        assert all(count > 0 for count in counts.values()), record
        assert main["planner_eligible"].all(), record
        assert not identity["planner_eligible"][main_windows:].any(), record
        assert result.observed_weight_version == kwargs["request"].expected_weight_version
        admissions.append(record)
        return admitted  # Observe the real admission; never supply audit-only rows.

    monkeypatch.setattr(training, "OfficialFadaAPPORunner", construct)
    monkeypatch.setattr(async_runtime, "_build_persistent_fada_worker", distillation_worker_factory)
    monkeypatch.setattr(train_distill, "build_persistent_fada_runtime", build_runtime)
    monkeypatch.setattr(persistent_workflow, "save_fada_checkpoint", observe_save)
    monkeypatch.setattr(persistent_workflow, "_admit_fada_artifact", observe_admission)
    try:
        first_cfg = configuration(tmp_path / "first", [
            "algo.max_iterations=21", "algo.save_interval=1", "training.logger=none",
        ])
        first = script().main.__wrapped__(first_cfg)
        assert first["learner_updates"] == 21
        first_path = Path(first["last_checkpoint"])
        intermediates = [first_path.parent / f"appo_update_{step}.pt" for step in range(1, 21)]
        assert all(path.is_file() for path in intermediates)
        first_payload = cp.read_oracle_checkpoint(first_path)
        earliest = cp.read_oracle_checkpoint(intermediates[0])
        assert changed(earliest["learner"]["actor"], first_payload["learner"]["actor"])
        second_cfg = deepcopy(first_cfg)
        second_cfg.algo.max_iterations = 22
        second_cfg.algo.resume_path = str(first_path)
        second_cfg.training.log_dir = str(tmp_path / "second")
        second = script().main.__wrapped__(second_cfg)
        assert second["learner_updates"] == 22
        final = Path(second["last_checkpoint"])
        final_payload = cp.read_oracle_checkpoint(final)
        identity = final_payload["official_appo_oracle"]
        prior_identity = first_payload["official_appo_oracle"]
        assert identity["role"] == "final" and identity["completed_updates"] == 22
        assert identity["oracle_lineage_id"] == prior_identity["oracle_lineage_id"]
        assert identity["session_id"] != prior_identity["session_id"]
        assert final_payload["cursor"]["unique_used_samples"] > first_payload["cursor"]["unique_used_samples"]
        assert changed(first_payload["learner"]["actor"], final_payload["learner"]["actor"])
        for runner in runners:
            check_clean(runner)

        cfg = distill_cfg(final, tmp_path / "distill")
        fada = cfg.training.fada
        fada.iterations, fada.windows_per_iteration, fada.batch_size = 2, 512, 40
        fada.num_envs = 128
        fada.idm_updates, fada.planner_updates = 1, 1
        # Bound the unchanged transition schedule; more parallel commands also
        # make all three steady-walk speed strata observable in a short run.
        fada.max_env_steps = 384
        fada.async_request_timeout_seconds = 30
        fada.intermediate_oracle_checkpoint_paths = [str(path) for path in intermediates]
        assert fada.v005_replay.enabled and fada.oracle_shadow_enabled
        assert fada.stand_transition_curriculum.enabled and fada.suboptimal_data_ratio == 2
        main_windows = int(fada.windows_per_iteration)
        source_windows = main_windows * (1 + int(fada.suboptimal_data_ratio))
        expected_samples = int(fada.iterations) * source_windows
        train_distill.main.__wrapped__(cfg)
        summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert summary["completed_iterations"] == fada.iterations
        assert summary["samples_seen"] == expected_samples
        assert summary["execution_mode"] == "persistent_async"
        assert np.isfinite(summary["last_idm_loss"]) and np.isfinite(summary["last_planner_loss"])
        assert activations == [1, 2]
        assert [record["iteration"] for record in admissions] == [0, 1]
        for record in admissions:
            assert record["main_windows"] == main_windows
            assert record["total_windows"] == source_windows
            assert record["scenarios"] == {
                "walk": main_windows // 2, "static_stand": main_windows // 4,
                "walk_to_stand": main_windows // 4,
            }
            assert sum(record["speed_bins"].values()) == main_windows // 4
        assert [iteration for iteration, _ in publications] == [0, 1, 2]
        for (_, before), (_, after) in zip(publications, publications[1:]):
            assert changed(before, after, prefix="idm.")
            assert changed(before, after, prefix="planner.")
        assert len(cfg.training.fada.intermediate_oracle_checkpoint_paths) == 20
        for iteration in range(2):
            artifact = open_fada_source_artifact(
                Path(fada.async_artifact_dir) / f"iteration_{iteration:04d}.pt",
                config=build_fada_architecture_config(cfg))
            assert artifact.num_samples == source_windows
            assert artifact.metadata["main_windows"] == main_windows
            assert artifact.metadata["expected_weight_version"] == iteration + 1
            assert artifact.metadata["official_oracle_identity"] == identity
            assert len([item for item in artifact.metadata["collections"]
                        if item["source"] == "intermediate_oracle"]) == 20

        student = load_fada_policy_checkpoint(fada.checkpoint_path)
        assert student.checkpoint["completed_iterations"] == 2
        assert student.checkpoint["samples_seen"] == expected_samples
        assert student.checkpoint["runtime_config"]["official_oracle_identity"] == identity
        assert student.policy.config.obs_dim == 66 and student.policy.config.action_dim == 29
        monkeypatch.setattr(viewer, "create_env", native_factory)
        play_cfg, args = play_arguments(second_cfg, final)
        teacher_session, _, _ = viewer._create_interactive_session(
            args, play_cfg, algo="appo", device=torch.device("cpu"),
            available_backends=("mujoco",), fada_session_factory=None)
        try:
            teacher_session.set_external_command(np.array([.6, 0, 0], np.float32))
            assert teacher_session.env.state.obs["obs"].shape == (1, 322)
            expected = teacher_session.policy.actions_from_env_observation(teacher_session.env.state.obs)
            np.testing.assert_allclose(teacher_session._build_actions().numpy(), expected, rtol=1e-5, atol=1e-6)
            assert np.isfinite(expected).all() and expected.shape == (1, 29)
            teacher_session.step_once()
            teacher_session.reset()
        finally:
            teacher_session.env.close()
        cfg.training.play_only = True
        cfg.training.play_checkpoint_path = str(fada.checkpoint_path)
        playback_cfg = RslRlPlaybackConfig(
            task="G1OfficialOracleFlat", load_run="-1", checkpoint=str(fada.checkpoint_path),
            action_mode="policy", policy_obs_mode="auto", algo_log_name="distill", log_root=None, num_envs=2)
        session, _, _ = create_fada_playback_session(
            playback_cfg=playback_cfg, cfg=cfg, root_dir=ROOT, device="cpu", deps={
                "resolve_checkpoint": lambda *a: Path(fada.checkpoint_path),
                "load_fada_policy": load_fada_policy_checkpoint, "create_env": native_factory,
                "build_env_cfg_override": lambda c: BackendAdapter(c, root_dir=ROOT).build_play_env_cfg_override(),
                "wrapper_cls": RslRlVecEnvWrapper,
            })
        try:
            session.set_external_command(np.array([.6, 0, 0], np.float32))
            session.step_once()
            assert session.controller._observation_history is not None
            assert np.isfinite(session.env.state.obs["obs"]).all()
            session.reset()
            assert session.controller._observation_history is None
        finally:
            session.env.close()
        for runtime in runtimes:
            assert runtime._closed and runtime._weight_sync is None
            assert runtime._runner._collector_process is None
            assert runtime._runner._error_recv is None and runtime._runner._error_send is None
        for name in names:
            with pytest.raises(FileNotFoundError):
                shared_memory.SharedMemory(name=name, create=False)
        assert {child.pid for child in mp.active_children()} <= previous_children
    finally:
        for runtime in runtimes:
            runtime.close()
        for runner in runners:
            runner.close()
        torch.set_num_threads(initial_threads)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
