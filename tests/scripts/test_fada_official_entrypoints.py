"""Bounded formal-route proof. Native physics/GUI/CUDA are never started."""
import copy
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.algos.torch.fada_official import checkpoint_selection as selection
from unilab.algos.torch.fada_official.checkpoint import (
    build_environment_contract, make_identity, read_oracle_checkpoint, save_oracle_checkpoint,
)
from unilab.algos.torch.fada_official.metrics import EpisodeMetrics, REWARD_STATISTIC
from unilab.algos.torch.fada_official.runner import OfficialFadaPPORunner
from unilab.algos.torch.rsl_rl_runtime import resolve_rsl_rl_ppo_runtime
from unilab.algos.torch.distill.fada.checkpoint import save_fada_checkpoint
from unilab.algos.torch.distill.fada.collection_contract import FADACollectionSpec
from unilab.algos.torch.distill.fada.collection_transaction import collect_fada_source_windows
from unilab.algos.torch.distill.fada.oracle import load_fada_oracle_policy, validate_loaded_fada_oracle_lineage
from unilab.algos.torch.distill.fada.source_artifact import save_fada_source_batch, open_fada_source_artifact
from unilab.algos.torch.distill.fada.trainer import FADATrainer
from unilab.algos.torch.distill.fada.model import FADAPlannerIDMPolicy
from unilab.algos.torch.distill.fada.workflow_setup import (
    build_fada_architecture_config, assert_fada_source_route_contract, _paper_source_plan,
)
from unilab.algos.torch.distill.workflows.entry_training import build_teacher_spec
from unilab.algos.torch.distill.workflows.entry_collection import _require_teacher_policy_collection_route
from unilab.base.registry import apply_cfg_overrides
from unilab.envs.locomotion.g1.official_oracle import env as env_owner
from unilab.envs.locomotion.g1.official_oracle.config import G1OfficialOracleFlatCfg, STUDENT_SCHEMA
from unilab.training.backend_adapter import BackendAdapter
from unilab.training.fada_official import OracleTrainingArtifacts, run_training
from unilab.training.rsl_rl import RslRlVecEnvWrapper

ROOT = Path(__file__).resolve().parents[2]


def helper_module():
    spec = importlib.util.spec_from_file_location("official_fake_native",
        ROOT / "tests/envs/locomotion/g1/test_official_oracle_flat.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def fake_native_env(overrides, num_envs=2):
    helper = helper_module()
    backend = helper.FakeBackend(num_envs)
    cfg = G1OfficialOracleFlatCfg()
    apply_cfg_overrides(cfg, overrides)
    original = env_owner.create_backend
    env_owner.create_backend = lambda *a, **kw: backend
    try:
        return env_owner.G1OfficialOracleFlatEnv(cfg, num_envs=num_envs)
    finally:
        env_owner.create_backend = original


def collector_env_factory(env_name, *, num_envs, sim_backend, env_cfg_override):
    assert mp.current_process().name != "MainProcess", "parent must not create a physics env"
    assert env_name == "G1OfficialOracleFlat" and sim_backend == "mujoco"
    return fake_native_env(env_cfg_override, num_envs)


def composed(family="ppo", extra=()):
    task = "mujoco" if family == "ppo" else "mujoco_fada_planner"
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "conf" / family)):
        return compose(config_name="config", overrides=[f"task=g1_official_oracle_flat/{task}", *extra])


@pytest.fixture(autouse=True)
def cpu_budget():
    threads = torch.get_num_threads()
    numpy_rng = np.random.get_state()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        np.random.seed(19)
        yield
    torch.set_num_threads(threads)
    np.random.set_state(numpy_rng)


def small_training(extra=()):
    return composed(extra=["algo.num_envs=2", "algo.actor.hidden_dims=[8,4]",
        "algo.critic.hidden_dims=[8,4]", "algo.algorithm.num_learning_epochs=1",
        "training.device=cpu", "training.logger=none", "env.max_episode_seconds=0.1", *extra])


def make_learner_and_identity(cfg):
    overrides = BackendAdapter(cfg, root_dir=ROOT).build_task_env_cfg_override()
    env_cfg = G1OfficialOracleFlatCfg()
    apply_cfg_overrides(env_cfg, overrides)
    env_cfg.validate()
    rl_cfg = OmegaConf.to_container(cfg.algo, resolve=True)
    runner = OfficialFadaPPORunner("G1OfficialOracleFlat", overrides, rl_cfg, device="cpu")
    learner = runner._build_learner()
    runner.close()
    identity = make_identity(environment_contract=build_environment_contract(env_cfg),
        lineage_id="offline-entrypoint-test", completed_iterations=2600, role="final",
        actor_config=rl_cfg["actor"], critic_config=rl_cfg["critic"],
        reward_statistic=REWARD_STATISTIC, reward_value=5.)
    return learner, identity


def test_formal_training_owner_runs_one_round_then_resumes_one_round(tmp_path):
    cfg = small_training(["algo.max_iterations=1", f"training.log_dir={tmp_path / 'first'}"])
    result = run_training(cfg, root_dir=ROOT, env_factory=collector_env_factory)
    assert result["completed_iterations"] == 1 and result["samples_seen"] == 48
    first = read_oracle_checkpoint(result["last_checkpoint"])
    assert first["official_oracle"]["role"] == "final"
    curve = selection.read_reward_curve(Path(result["log_dir"]) / "reward_curve.pt", first["official_oracle"])
    assert curve["points"][0]["episode_window_count"] > 0
    assert curve["points"][0]["value"] is not None
    cfg2 = small_training(["algo.max_iterations=2", f"algo.resume_path={result['last_checkpoint']}",
        f"training.log_dir={tmp_path / 'second'}"])
    second = run_training(cfg2, root_dir=ROOT, env_factory=collector_env_factory)
    final = read_oracle_checkpoint(second["last_checkpoint"])
    assert second["completed_iterations"] == 2 and second["samples_seen"] == 96
    assert final["official_oracle"]["oracle_lineage_id"] == first["official_oracle"]["oracle_lineage_id"]
    assert [p["iteration"] for p in selection.read_reward_curve(
        Path(second["log_dir"]) / "reward_curve.pt", final["official_oracle"])["points"]] == [1, 2]
    # Already complete must not allocate IPC/start a collector or create a run dir.
    cfg2.algo.resume_path = second["last_checkpoint"]
    cfg2.training.log_dir = str(tmp_path / "never-created")
    skipped = run_training(cfg2, root_dir=ROOT, env_factory=collector_env_factory)
    assert skipped["status"] == "already_complete" and not Path(cfg2.training.log_dir).exists()


def test_script_dispatches_to_owner_without_parent_environment(monkeypatch):
    from scripts import train_rsl_rl
    cfg = small_training()
    called = []
    from unilab.training import fada_official
    monkeypatch.setattr(fada_official, "run_training", lambda cfg, **kw: called.append(cfg))
    monkeypatch.setattr(train_rsl_rl, "create_env", lambda *a, **k: pytest.fail("parent env created"))
    train_rsl_rl.main.__wrapped__(cfg)
    assert called == [cfg]


@pytest.mark.parametrize("override", ["algo.collection.obs_dim=66", "algo.num_steps_per_env=2",
    "algo.collection.mode=async", "training.sim_backend=motrix", "training.num_timesteps=47"])
def test_invalid_combinations_fail_before_runner_or_environment(override, tmp_path):
    cfg = small_training([override, f"training.log_dir={tmp_path / 'invalid'}"])
    with pytest.raises(ValueError):
        run_training(cfg, root_dir=ROOT, runner_factory=lambda *a, **kw: pytest.fail("runner created"))
    assert not (tmp_path / "invalid").exists()


def test_episode_returns_are_raw_cross_rollout_and_last100():
    reducer = EpisodeMetrics(2)
    result = reducer.consume(torch.tensor([[1., 2.], [4., 5.]]), torch.tensor([[0, 0], [0, 1]]))
    assert result["Train/mean_reward"] == 9 and result["episode_window_count"] == 1
    result = reducer.consume(torch.tensor([[3.], [6.]]), torch.tensor([[1], [1]]))
    assert result["Train/mean_reward"] == 7 and result["episodes_completed"] == 2
    assert EpisodeMetrics(1).consume(torch.zeros(1, 2), torch.zeros(1, 2))["Train/mean_reward"] is None
    reducer.consume(torch.ones(2, 60), torch.ones(2, 60))
    assert len(reducer.recent_returns) == 100 and set(reducer.recent_returns) == {1.}


def test_initial_clock_is_the_actual_phase_and_timeout_owner():
    cfg = small_training()
    env = fake_native_env(BackendAdapter(cfg, root_dir=ROOT).build_task_env_cfg_override())
    try:
        wrapper = RslRlVecEnvWrapper(env, device="cpu")
        before = env._backend._physics_state.copy()
        env.initialize_training_progress()
        np.testing.assert_equal(before, env._backend._physics_state)
        assert ((env.state.info["steps"] >= 0) & (env.state.info["steps"] < 5)).all()
        expected = copy.deepcopy(env.command)
        expected.update_phase(env.state.info["steps"])
        np.testing.assert_equal(expected.phase, env.command.phase)
        env.state.info["steps"][0] = 4
        _, _, done, info = wrapper.step(torch.zeros(2, 29))
        assert done[0] and info["time_outs"][0]
        assert env.state.info["steps"][0] == 0
    finally:
        env.close()


def test_save100_final25000_and_callback_failure_do_not_overwrite(tmp_path):
    learner, identity = make_learner_and_identity(small_training())
    metrics = {"iteration": 1, "samples": 48, "Train/mean_reward": 3., "episode_window_count": 2}
    callback = OracleTrainingArtifacts(tmp_path, learner, identity, target=25000, save_interval=100,
                                      completed=99, samples_seen=1000)
    callback(metrics)
    assert read_oracle_checkpoint(tmp_path / "model_100.pt")["official_oracle"]["role"] == "intermediate"
    callback = OracleTrainingArtifacts(tmp_path, learner, identity, target=25000, save_interval=100,
                                      completed=24999, samples_seen=2000, curve=callback.curve)
    callback(metrics)
    assert read_oracle_checkpoint(tmp_path / "model_25000.pt")["samples_seen"] == 2048
    assert read_oracle_checkpoint(tmp_path / "model_25000.pt")["official_oracle"]["role"] == "final"
    duplicate = OracleTrainingArtifacts(tmp_path, learner, identity, target=25000, save_interval=100, completed=24999)
    with pytest.raises(FileExistsError):
        duplicate(metrics)


def frozen_official_reference(steps, rewards, n=20):
    # Fixed-source reference, independent path-index implementation.
    def logspace(items):
        if len(items) <= n:
            return list(items)
        idx = sorted(set([int(round(x - 1)) for x in np.geomspace(1, len(items), n)] + [len(items)-1]))
        result = [items[i] for i in idx]
        while len(result) > n:
            result.pop(-2)
        while len(result) < n:
            candidates = [v for v in items if v not in result]
            result.append(max(candidates, key=lambda x: min(abs(x-y) for y in result)))
            result.sort()
        return result
    chosen = set(logspace(steps))
    for threshold in np.quantile(list(rewards.values()), [.5, .7, .85, .95, 1.]):
        reached = next(step for step, reward in sorted(rewards.items()) if reward >= threshold)
        chosen.add(min(steps, key=lambda step: abs(step-reached)))
    return logspace(sorted(chosen))


@pytest.fixture
def campaign(tmp_path):
    cfg = small_training(["env.max_episode_seconds=20.0"])
    learner, identity = make_learner_and_identity(cfg)
    rewards = {i: float(np.sin(i / 137.) + i / 3000.) for i in range(25, 2601, 25)}
    records = []
    for i in range(100, 2601, 100):
        current = {**identity, "completed_iterations": i,
            "role": "final" if i == 2600 else "intermediate",
            "reward": {"value": rewards[i], "statistic": REWARD_STATISTIC}}
        path = save_oracle_checkpoint(tmp_path / f"model_{i}.pt", learner, identity=current)
        records.append({"path": str(path), "identity": current})
    curve = {"schema": "fada_official_reward_curve_v1", "identity": records[-1]["identity"],
        "statistic": REWARD_STATISTIC, "window": 100,
        "points": [{"iteration": i, "value": r, "episode_window_count": 100} for i, r in rewards.items()],
        "checkpoints": records}
    torch.save(curve, tmp_path / "reward_curve.pt")
    return cfg, Path(records[-1]["path"]), records, rewards


def test_reward_selection_is_official_and_missing_rewards_are_explicit(campaign, tmp_path):
    _, final, records, rewards = campaign
    paths, record = selection.select_source_checkpoints(final)
    expected = frozen_official_reference(list(range(100, 2600, 100)), rewards)
    assert [read_oracle_checkpoint(p)["official_oracle"]["completed_iterations"] for p in paths] == expected
    assert record["sampling_mode"] == "reward_aware" and final not in paths
    curve = torch.load(tmp_path / "reward_curve.pt", weights_only=True)
    curve["points"] = []
    torch.save(curve, tmp_path / "missing.pt")
    with pytest.warns(RuntimeWarning, match="falling back"):
        paths, record = selection.select_source_checkpoints(final, curve_path=tmp_path / "missing.pt")
    assert len(paths) == 20 and record["sampling_mode"] == "logspace"
    curve["checkpoints"] = records[:19]
    torch.save(curve, tmp_path / "too-few.pt")
    with pytest.raises(ValueError, match="20 unique"):
        selection.select_source_checkpoints(final, curve_path=tmp_path / "too-few.pt")


def test_composed_source_route_collects_trains_and_plays_both_families(campaign, tmp_path, monkeypatch):
    train_cfg, final, _, _ = campaign
    cfg = composed("distill", [f"teacher.checkpoint_path={final}", "training.device=cpu",
        "training.logger=none", "training.fada.history_length=2", "training.fada.prediction_horizon=2",
        "training.fada.hidden_dim=8", "training.fada.feedforward_dim=16", "training.fada.num_heads=2",
        "training.fada.planner_layers=1", "training.fada.idm_encoder_layers=1", "training.fada.idm_decoder_layers=1",
        "training.fada.windows_per_iteration=20"])
    _require_teacher_policy_collection_route(cfg)
    config = build_fada_architecture_config(cfg)
    assert_fada_source_route_contract(cfg, config)
    selection.configure_source_selection(cfg, final)
    plan = _paper_source_plan(cfg)
    assert len(plan.source_allocations) == 20 and sum(n for _, n in plan.source_allocations) == 40
    spec = build_teacher_spec(cfg)
    teacher = load_fada_oracle_policy(final, spec)
    validate_loaded_fada_oracle_lineage(teacher, [load_fada_oracle_policy(p, spec) for p in plan.checkpoint_paths])
    env = fake_native_env(BackendAdapter(cfg, root_dir=ROOT).build_task_env_cfg_override())
    try:
        assert env.get_official_oracle_contract() == teacher.checkpoint_identity["environment_contract"]
        env._backend._isolated_rollout_pool = helper_module().FakePool(env._backend)
        env.reset_all()
        env.set_external_command([.6, 0, 0])
        batch = collect_fada_source_windows(env, teacher_policy=teacher, config=config, num_windows=2,
            spec=FADACollectionSpec(student_projection=STUDENT_SCHEMA, collect_oracle_shadow=True,
                                    max_env_steps=20)).batch
        metadata = {"official_oracle_identity": teacher.checkpoint_identity}
        path = save_fada_source_batch(tmp_path / "source.pt", batch, config=config, metadata=metadata)
        loaded_batch = open_fada_source_artifact(path, config=config).load_batch(0)
        student = FADAPlannerIDMPolicy(config)
        trainer = FADATrainer(student, idm_optimizer=torch.optim.Adam(student.idm.parameters()),
                              planner_optimizer=torch.optim.Adam(student.planner.parameters()))
        stats = trainer.update(loaded_batch)
        assert np.isfinite(stats.idm_loss) and np.isfinite(stats.planner_loss)
        student_path = save_fada_checkpoint(tmp_path / "student.pt", student, trainer,
            completed_iterations=1, samples_seen=2, runtime_config=metadata)
    finally:
        env.close()
    from unilab.visualization import playback_viewer as viewer
    from unilab.visualization.playback_cli import _build_play_args
    from unilab.visualization.playback_distill_sessions import create_fada_playback_session
    # Only substitute native env construction; resolver, loaders, factories and sessions are real.
    def native_factory(cfg, *, num_envs, env_cfg_override, **kw):
        return fake_native_env(env_cfg_override, num_envs)
    monkeypatch.setattr(viewer, "create_env", native_factory)
    play_cfg = copy.deepcopy(train_cfg)
    play_cfg.training.play_only = True
    OmegaConf.update(play_cfg, "training.play_checkpoint_path", str(final), force_add=True)
    play_cfg.interactive.action_mode = "policy"
    args = _build_play_args(play_cfg, algo="ppo")
    args.checkpoint_path = str(final)
    session, _, _ = viewer._create_interactive_session(args, play_cfg, algo="ppo", device=torch.device("cpu"),
        available_backends=("mujoco",), fada_session_factory=None)
    try:
        assert session.env.cfg.evaluating and not session.env.cfg.randomization_enabled
        session.set_external_command(np.array([.6, 0, 0], np.float32))
        np.testing.assert_allclose(session._build_actions().numpy(), teacher.actions_from_env_observation(session.env.state.obs),
                                   rtol=1e-5, atol=1e-6)
        session.step_once()
    finally:
        session.env.close()
    from unilab.visualization.playback_sessions import RslRlPlaybackConfig
    from unilab.algos.torch.distill.fada.checkpoint import load_fada_policy_checkpoint
    cfg.training.play_only = True
    cfg.training.play_checkpoint_path = str(student_path)
    pcfg = RslRlPlaybackConfig(task="G1OfficialOracleFlat", load_run="-1", checkpoint=str(student_path),
        action_mode="policy", policy_obs_mode="auto", algo_log_name="distill", log_root=None, num_envs=2)
    session, _, _ = create_fada_playback_session(playback_cfg=pcfg, cfg=cfg, root_dir=ROOT, device="cpu", deps={
        "resolve_checkpoint": lambda *a: student_path, "load_fada_policy": load_fada_policy_checkpoint,
        "create_env": native_factory, "build_env_cfg_override": lambda c: BackendAdapter(c, root_dir=ROOT).build_play_env_cfg_override(),
        "wrapper_cls": RslRlVecEnvWrapper})
    try:
        assert session.env.cfg.evaluating and not session.env.cfg.randomization_enabled
        session.set_external_command(np.array([.6, 0, 0], np.float32))
        session.env.state.info["steps"][0] = session.env.cfg.max_episode_steps - 1
        session.step_once()
        assert session.controller._observation_history is not None
        session.reset()
        assert session.controller._observation_history is None
    finally:
        session.env.close()


def test_legacy_resolver_keeps_wrapper_only_contract(monkeypatch):
    from types import SimpleNamespace
    import rsl_rl.utils
    monkeypatch.setattr(rsl_rl.utils, "resolve_callable", lambda _: lambda cfg: SimpleNamespace(wrapper_cls=RslRlVecEnvWrapper))
    runtime = resolve_rsl_rl_ppo_runtime({"runtime_resolver": "old:resolver"}, default_wrapper_cls=RslRlVecEnvWrapper)
    assert runtime.wrapper_cls is RslRlVecEnvWrapper and runtime.train is None and runtime.playback_factory is None
