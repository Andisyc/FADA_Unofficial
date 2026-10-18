"""One temporary synthetic teacher campaign; no GUI, physics, CUDA or long training."""
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from tests.scripts.test_fada_official_appo_checkpoint import build_learner, cold_environment
from tests.scripts.test_fada_official_appo_training import configuration, script
from tests.scripts.test_fada_official_entrypoints import ROOT, fake_native_env, helper_module
from unilab.algos.torch.fada_official_appo import checkpoint as cp, checkpoint_selection as selection
from unilab.algos.torch.appo.runtime import resolve_appo_runtime
from unilab.algos.torch.distill.fada.oracle import (
    load_fada_oracle_policy, reload_fada_oracle_policy_, validate_loaded_fada_oracle_lineage,
    validate_fada_oracle_environment_contract,
)
from unilab.algos.torch.distill.fada.workflow_setup import (
    build_fada_architecture_config, assert_fada_source_route_contract, _paper_source_plan,
)
from unilab.algos.torch.distill.workflows.entry_collection import _require_teacher_policy_collection_route
from unilab.algos.torch.distill.workflows.entry_training import build_teacher_spec
from unilab.algos.torch.distill.learning.teacher import DistillationTeacherSpec
from unilab.algos.torch.distill.fada.model import FADAPlannerIDMPolicy
from unilab.algos.torch.distill.fada.trainer import FADATrainer
from unilab.algos.torch.distill.fada.checkpoint import save_fada_checkpoint, load_fada_policy_checkpoint
from unilab.algos.torch.distill.fada.async_runtime import PersistentFADACollectorWorker
from unilab.algos.torch.distill.fada.persistent_workflow import _admit_fada_artifact
from unilab.algos.torch.distill.fada.source_artifact import open_fada_source_artifact
from unilab.algos.torch.distill.runtime.async_runtime import DaggerCollectRequest
from unilab.ipc.weight_sync import SharedWeightSync
from unilab.training.backend_adapter import BackendAdapter
from unilab.training.rsl_rl import RslRlVecEnvWrapper
from unilab.visualization import playback_viewer as viewer
from unilab.visualization.playback_cli import _build_play_args
from unilab.visualization.playback_policy_sessions import create_appo_playback_session
from unilab.visualization.playback_sessions import RslRlPlaybackConfig


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu():
    threads = torch.get_num_threads()
    numpy_rng = np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        torch.set_num_threads(1)
        torch.manual_seed(29)
        yield
    torch.set_num_threads(threads)
    np.random.set_state(numpy_rng)


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    """Synthetic metadata and distinct actors, not 21 actually trained policies."""
    directory = tmp_path_factory.mktemp("synthetic-appo-teachers")
    cfg = configuration(directory)
    algo = OmegaConf.to_container(cfg.algo, resolve=True)
    learner = build_learner(algo)
    identity = dict(schema=cp.ORACLE_SCHEMA, algorithm=cp.ALGORITHM, oracle_lineage_id="synthetic-appo",
        session_id="synthetic-session-1", parent_checkpoint=None, role="intermediate", completed_updates=1,
        environment_contract=cold_environment(cfg), actor_config=algo["actor"], critic_config=algo["critic"],
        training_contract=cp.training_contract(algo, num_envs=2, replay_queue_size=3))
    paths = []
    for update in range(1, 22):
        learner._update_counter = update
        with torch.no_grad():
            learner.actor.mlp[-1].bias.fill_(2 + update / 100)
        session = "synthetic-session-1" if update <= 10 else "synthetic-session-2"
        record = {**identity, "completed_updates": update, "role": "final" if update == 21 else "intermediate",
            "session_id": session, "reward": dict(value=update / 10, window_count=100,
                statistic="raw_episode_return_last100", session_id=session)}
        cursor = dict(generated_samples=update*48, published_samples=update*48, accepted_samples=update*48,
            unique_used_samples=update*48, reused_samples=0, stale_dropped_samples=0, abandoned_samples=0,
            learner_updates=update, optimizer_steps=update*2, sample_exposures=update*48)
        name = f"appo_update_{update}.pt" if update < 21 else "appo_final_update_21.pt"
        paths.append(cp.save_oracle_checkpoint(directory / name, learner, identity=record, cursor=cursor))
    return cfg, paths[-1], paths[:-1]


def distill_cfg(final, tmp_path):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "conf/distill")):
        return compose(config_name="config", overrides=[
            "task=g1_official_oracle_flat/mujoco_fada_appo_planner", f"teacher.checkpoint_path={final}",
            "training.device=cpu", "training.logger=none", "training.fada.history_length=2",
            "training.fada.prediction_horizon=2", "training.fada.hidden_dim=8",
            "training.fada.feedforward_dim=16", "training.fada.num_heads=2",
            "training.fada.planner_layers=1", "training.fada.idm_encoder_layers=1",
            "training.fada.idm_decoder_layers=1", "training.fada.windows_per_iteration=20",
            "training.fada.num_envs=2", "training.fada.max_env_steps=256",
            f"training.fada.checkpoint_path={tmp_path / 'student.pt'}",
            f"training.fada.async_artifact_dir={tmp_path / 'sources'}"])


def native_factory(cfg, *, num_envs, env_cfg_override, **kwargs):
    env = fake_native_env(env_cfg_override, num_envs)
    env._backend._isolated_rollout_pool = helper_module().FakePool(env._backend)
    return env


def play_arguments(cfg, path):
    cfg = deepcopy(cfg)
    cfg.training.play_only = True
    OmegaConf.update(cfg, "training.play_checkpoint_path", str(path), force_add=True)
    cfg.interactive.action_mode = "policy"
    args = _build_play_args(cfg, algo="appo")
    args.checkpoint_path = str(path)
    return cfg, args


def test_selection_uses_appp_metadata_and_20_unique_same_lineage_teachers(campaign, tmp_path):
    _, final, intermediates = campaign
    paths, record = selection.select_source_checkpoints(final)
    assert paths == intermediates and record["sampling_mode"] == "reward_aware"
    assert {p["session_id"] for p in record["intermediates"]} == {"synthetic-session-1", "synthetic-session-2"}
    cfg = distill_cfg(final, tmp_path)
    selection.configure_source_selection(cfg, final)
    _require_teacher_policy_collection_route(cfg)
    architecture = build_fada_architecture_config(cfg)
    assert_fada_source_route_contract(cfg, architecture)
    plan = _paper_source_plan(cfg)
    assert len(plan.source_allocations) == 20 and sum(n for _, n in plan.source_allocations) == 40
    spec = build_teacher_spec(cfg)
    teacher = load_fada_oracle_policy(final, spec)
    loaded = [load_fada_oracle_policy(p, spec) for p in paths]
    validate_loaded_fada_oracle_lineage(teacher, loaded, expected_behavior_profile="official_oracle_flat_v1")
    assert all(not p.requires_grad for p in teacher.parameters())
    object_id = id(teacher)
    reload_fada_oracle_policy_(teacher, paths[0], spec)
    assert id(teacher) == object_id and teacher.checkpoint_identity["completed_updates"] == 1
    torch.testing.assert_close(teacher.actor.mlp[-1].bias, torch.full((29,), 2.01))


def test_explicit_selection_cross_directories_and_missing_reward_fallback(campaign, tmp_path):
    _, final, paths = campaign
    # Explicit paths can cross sessions/directories; no parent-directory crawling.
    payload = cp.read_oracle_checkpoint(paths[0])
    moved = tmp_path / "other-run" / paths[0].name
    moved.parent.mkdir()
    torch.save(payload, moved)
    selected, record = selection.select_source_checkpoints(final, intermediate_paths=[moved, *paths[1:]])
    assert selected[0] == moved and record["sampling_mode"] == "explicit"
    for path in [*paths, final]:
        payload = cp.read_oracle_checkpoint(path)
        payload["official_appo_oracle"]["reward"].update(value=None, window_count=0)
        torch.save(payload, tmp_path / path.name)
    with pytest.warns(RuntimeWarning, match="logspace"):
        selected, record = selection.select_source_checkpoints(tmp_path / final.name)
    assert len(selected) == 20 and record["sampling_mode"] == "logspace"


@pytest.mark.parametrize("damage", ["too_few", "duplicate", "lineage", "environment", "algorithm", "progress"])
def test_selection_rejects_bad_candidates_before_environment(campaign, tmp_path, damage):
    _, final, paths = campaign
    candidates = list(paths)
    if damage == "too_few":
        candidates.pop()
    elif damage == "duplicate":
        candidates[-1] = candidates[0]
    else:
        payload = cp.read_oracle_checkpoint(paths[0])
        identity = payload["official_appo_oracle"]
        if damage == "lineage":
            identity["oracle_lineage_id"] = "other-lineage"
        elif damage == "environment":
            identity["environment_contract"]["config_sha256"] = "0" * 64
        elif damage == "algorithm":
            identity["algorithm"] = "fada_official_ppo"
        else:
            payload["cursor"]["unique_used_samples"] = 21 * 48
        broken = tmp_path / "bad.pt"
        torch.save(payload, broken)
        candidates[0] = broken
    with pytest.raises(ValueError):
        selection.select_source_checkpoints(final, intermediate_paths=candidates)


def test_actual_interactive_dispatch_command_refresh_and_raw_teacher_equality(campaign, monkeypatch):
    cfg, final, _ = campaign
    cfg, args = play_arguments(cfg, final)
    monkeypatch.setattr(viewer, "create_env", native_factory)
    session, mode, observed = viewer._create_interactive_session(args, cfg, algo="appo", device=torch.device("cpu"),
        available_backends=("mujoco",), fada_session_factory=None)
    try:
        assert observed == str(final) and mode == "actor"
        assert session.env.cfg.evaluating and not session.env.cfg.randomization_enabled
        session.set_external_command(np.array([.6, 0, 0], np.float32))
        expected = session.policy.actions_from_env_observation(session.env.state.obs)
        np.testing.assert_allclose(session._build_actions().numpy(), expected, rtol=1e-5, atol=1e-6)
        assert (expected > 1).all()  # No hidden tanh/clipping/PD scaling.
        session.step_once()
    finally:
        session.env.close()


@pytest.mark.parametrize("kind", ["missing", "bare", "ppo"])
def test_playback_rejects_bad_checkpoint_without_env_allocation(campaign, tmp_path, kind):
    cfg, final, _ = campaign
    path = tmp_path / "bad.pt"
    if kind == "bare":
        torch.save(cp.read_oracle_checkpoint(final)["learner"], path)
    elif kind == "ppo":
        torch.save({"official_oracle": {"schema": "fada_official_oracle_v1"}}, path)
    cfg, _ = play_arguments(cfg, path)
    def forbidden(*args):
        raise AssertionError("checkpoint validation must precede env construction")
    pcfg = RslRlPlaybackConfig(task="G1OfficialOracleFlat", load_run="-1", checkpoint=str(path),
        action_mode="policy", policy_obs_mode="actor", algo_log_name="fada_official_appo", log_root=None)
    with pytest.raises((FileNotFoundError, ValueError)):
        create_appo_playback_session(playback_cfg=pcfg, cfg=cfg,
            rl_cfg=OmegaConf.to_container(cfg.algo, resolve=True), root_dir=ROOT, env_factory=forbidden,
            device="cpu", wrapper_cls=RslRlVecEnvWrapper)


def test_formal_play_only_uses_runtime_without_learner_or_logs(campaign, tmp_path, monkeypatch):
    import unilab.training as training
    from unilab.training import fada_official_appo as owner
    cfg, final, _ = campaign
    cfg, _ = play_arguments(cfg, final)
    cfg.training.log_dir = str(tmp_path / "never-created")
    cfg.training.play_env_num = 2
    observed = []
    def factory(*args, **kwargs):
        env = native_factory(*args, **kwargs)
        observed.append(env)
        # Replace only the GUI/render loop, while exercising the actual session callbacks.
        def render(**kw):
            kw["initialize"]()
            kw["step"](0)
            return "synthetic-no-video"
        env.run_playback_mode = render
        return env
    monkeypatch.setattr(training, "create_env", factory)
    def forbidden(*args, **kwargs):
        raise AssertionError("play_only must not construct a training learner/collector")
    monkeypatch.setattr(owner, "OfficialFadaAPPORunner", forbidden)
    runtime = resolve_appo_runtime(OmegaConf.to_container(cfg.algo, resolve=True), default_play_fn=forbidden)
    from unilab.algos.torch.fada_official_appo.playback import play_checkpoint
    assert runtime.play_fn is play_checkpoint
    assert script().main.__wrapped__(cfg) == "synthetic-no-video"
    assert len(observed) == 1 and not Path(cfg.training.log_dir).exists()


def test_worker_source_writer_admission_real_student_update_and_playback(campaign, tmp_path):
    from unilab.visualization.playback_distill_sessions import create_fada_playback_session
    cfg = distill_cfg(campaign[1], tmp_path)
    selection.configure_source_selection(cfg, campaign[1])
    config = build_fada_architecture_config(cfg)
    spec = build_teacher_spec(cfg)
    teacher = load_fada_oracle_policy(campaign[1], spec)
    OmegaConf.update(cfg, "training.fada.official_oracle_identity", teacher.checkpoint_identity, force_add=True)
    student = FADAPlannerIDMPolicy(config)
    trainer = FADATrainer(student, idm_optimizer=torch.optim.Adam(student.idm.parameters()),
                          planner_optimizer=torch.optim.Adam(student.planner.parameters()))
    initial = save_fada_checkpoint(tmp_path / "initial.pt", student, trainer, completed_iterations=0,
        samples_seen=0, runtime_config={"official_oracle_identity": teacher.checkpoint_identity})
    weights = SharedWeightSync.from_state_dict(student.state_dict())
    worker = None
    try:
        plan = _paper_source_plan(cfg)
        worker = PersistentFADACollectorWorker(root_dir=str(ROOT),
            cfg_payload=OmegaConf.to_container(cfg, resolve=True), standing_curriculum_enabled=True,
            architecture=asdict(config), final_teacher_checkpoint=str(campaign[1]),
            source_allocations=[(str(p), n) for p, n in plan.source_allocations],
            initial_checkpoint_path=str(initial), device="cpu", weight_sync_name=weights.name,
            weight_sync_lock=weights.lock, weight_param_shapes={k: v.shape for k, v in student.state_dict().items()},
            env_factory=native_factory)
        request = DaggerCollectRequest(request_id="synthetic-source", scenario="fada_iteration", iteration=0,
            checkpoint_path=str(initial), output_path=str(tmp_path / "source.pt"), expected_weight_version=weights.version)
        result = worker.collect(request)
        loaded = open_fada_source_artifact(request.output_path, config=config)
        main, summaries, _ = _admit_fada_artifact(cfg, loaded=loaded, result=result, request=request,
            training_schedule="alternating_idm_then_planner")
        assert main == 20 and result.num_samples == 60
        assert len([s for s in summaries if s["source"] == "intermediate_oracle"]) == 20
        assert loaded.metadata["official_oracle_identity"] == teacher.checkpoint_identity
        batch = loaded.load_batch(0)
        stats = trainer.update(batch)
        assert np.isfinite(stats.idm_loss) and np.isfinite(stats.planner_loss)
        student_path = save_fada_checkpoint(tmp_path / "updated.pt", student, trainer,
            completed_iterations=1, samples_seen=batch.command.shape[0],
            runtime_config={"official_oracle_identity": teacher.checkpoint_identity})
        saved_identity = cfg.training.fada.official_oracle_identity.oracle_lineage_id
        cfg.training.fada.official_oracle_identity.oracle_lineage_id = "other-lineage"
        with pytest.raises(ValueError, match="identity"):
            _admit_fada_artifact(cfg, loaded=loaded, result=result, request=request,
                training_schedule="alternating_idm_then_planner")
        cfg.training.fada.official_oracle_identity.oracle_lineage_id = saved_identity
    finally:
        if worker is not None:
            worker.close()
        weights.cleanup()
    cfg.training.play_only = True
    cfg.training.play_checkpoint_path = str(student_path)
    pcfg = RslRlPlaybackConfig(task="G1OfficialOracleFlat", load_run="-1", checkpoint=str(student_path),
        action_mode="policy", policy_obs_mode="auto", algo_log_name="distill", log_root=None, num_envs=2)
    session, _, _ = create_fada_playback_session(playback_cfg=pcfg, cfg=cfg, root_dir=ROOT, device="cpu", deps={
        "resolve_checkpoint": lambda *a: student_path, "load_fada_policy": load_fada_policy_checkpoint,
        "create_env": native_factory, "build_env_cfg_override": lambda c: BackendAdapter(c, root_dir=ROOT).build_play_env_cfg_override(),
        "wrapper_cls": RslRlVecEnvWrapper})
    try:
        session.set_external_command(np.array([.6, 0, 0], np.float32))
        session.step_once()
        assert session.controller._observation_history is not None
        session.reset()
        assert session.controller._observation_history is None
    finally:
        session.env.close()


def test_wrong_algorithm_spec_environment_and_reload_rejected_without_mutation(campaign, tmp_path):
    _, final, paths = campaign
    spec = DistillationTeacherSpec(322, 29, algo_type="fada_official_appo", critic_obs_dim=322)
    for algorithm in ("sac", "fada_official_ppo", "privileged_locomotion_sac"):
        with pytest.raises(ValueError, match="algo_type"):
            load_fada_oracle_policy(final, replace(spec, algo_type=algorithm))
    policy = load_fada_oracle_policy(final, spec)
    before = deepcopy(policy.actor.state_dict())
    bad = cp.read_oracle_checkpoint(paths[0])
    bad["official_appo_oracle"]["oracle_lineage_id"] = "wrong-lineage"
    path = tmp_path / "wrong.pt"
    torch.save(bad, path)
    with pytest.raises(ValueError, match="lineage"):
        reload_fada_oracle_policy_(policy, path, spec)
    for name, value in before.items():
        torch.testing.assert_close(value, policy.actor.state_dict()[name], rtol=0, atol=0)
    cfg = distill_cfg(final, tmp_path)
    env = native_factory(cfg, num_envs=2,
        env_cfg_override=BackendAdapter(cfg, root_dir=ROOT).build_task_env_cfg_override())
    try:
        validate_fada_oracle_environment_contract(policy.checkpoint_identity, env, cfg)
        mismatch = deepcopy(policy.checkpoint_identity)
        mismatch["environment_contract"]["asset_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="environment"):
            cp.validate_environment(mismatch, env)
    finally:
        env.close()


def test_legacy_resolver_and_ppo_loader_stay_separate(campaign):
    from unilab.algos.torch.fada_official.checkpoint import read_oracle_checkpoint as read_ppo
    from unilab.algos.torch.appo.runner import APPORunner
    play = lambda *a, **k: None
    runtime = resolve_appo_runtime({}, default_play_fn=play)
    assert runtime.runner_cls is APPORunner and runtime.play_fn is play and runtime.train_fn is None
    with pytest.raises(ValueError):
        read_ppo(campaign[1])


@pytest.mark.parametrize("damage", ["intermediate", "algorithm", "missing"])
def test_student_binding_rejects_incomplete_or_wrong_appp_identity(campaign, tmp_path, damage):
    from unilab.algos.torch.distill.fada.official_oracle import validate_student_binding
    config = build_fada_architecture_config(distill_cfg(campaign[1], tmp_path))
    identity = cp.read_oracle_checkpoint(campaign[1])["official_appo_oracle"]
    if damage == "intermediate":
        identity["role"] = "intermediate"
    elif damage == "algorithm":
        identity["algorithm"] = "fada_official_ppo"
    metadata = {} if damage == "missing" else {"official_oracle_identity": identity}
    with pytest.raises(ValueError):
        validate_student_binding(config, metadata)
