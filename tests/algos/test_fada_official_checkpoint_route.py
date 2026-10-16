"""One bounded offline proof for Oracle persistence and the student-facing route."""
import copy
import importlib.util
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensordict import TensorDict

from unilab.algos.torch.fada_official.checkpoint import (
    LoadedOfficialOracle, build_environment_contract, load_oracle_policy,
    make_identity, make_selection_record, read_oracle_checkpoint,
    restore_oracle_checkpoint, save_oracle_checkpoint, validate_environment, validate_lineage,
)
from unilab.algos.torch.fada_official_ppo import OfficialFadaPPO
from unilab.algos.torch.distill.fada.oracle import (
    load_fada_oracle_policy, reload_fada_oracle_policy_, validate_loaded_fada_oracle_lineage,
)
from unilab.algos.torch.distill.learning.teacher import DistillationTeacherSpec
from unilab.algos.torch.distill.fada.collection_contract import FADACollectionSpec
from unilab.algos.torch.distill.fada.collection_transaction import collect_fada_source_windows
from unilab.algos.torch.distill.fada.collection_io import _oracle_shadow_pair
from unilab.algos.torch.distill.fada.collection_views import student_view
from unilab.algos.torch.distill.fada.model import FADAArchitectureConfig, FADAPlannerIDMPolicy
from unilab.algos.torch.distill.fada.playback import FADAPlaybackController
from unilab.algos.torch.distill.fada.checkpoint import save_fada_checkpoint, load_fada_policy_checkpoint
from unilab.algos.torch.distill.fada.source_artifact import (
    FADAShardedSourceWriter, open_fada_source_artifact, save_fada_source_batch,
)
from unilab.algos.torch.distill.fada.trainer import FADATrainer
from unilab.envs.locomotion.g1.official_oracle import env as env_owner
from unilab.envs.locomotion.g1.official_oracle.config import (
    G1OfficialOracleFlatCfg, OfficialOracleRewardConfig, STUDENT_SCHEMA,
)
from unilab.training.rsl_rl import normalize_ppo_train_cfg, RslRlVecEnvWrapper
from unilab.visualization.playback_sessions import RslRlPlaybackConfig, FADAPlaybackSession
from unilab.visualization.playback_policy_sessions import create_official_oracle_playback_session

ROOT = Path(__file__).resolve().parents[2]
SPEC = DistillationTeacherSpec(322, 29, algo_type="fada_official_ppo", critic_obs_dim=322)


@pytest.fixture(autouse=True)
def bounded_cpu():
    threads = torch.get_num_threads()
    numpy_rng = np.random.get_state()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        np.random.seed(17)
        yield
    np.random.set_state(numpy_rng)
    torch.set_num_threads(threads)


@pytest.fixture
def route(monkeypatch):
    # Reuse the step-2 scripted native-pool replacement, never compile MjModel.
    spec = importlib.util.spec_from_file_location("step2_fake_env",
        ROOT / "tests/envs/locomotion/g1/test_official_oracle_flat.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    backend = helper.FakeBackend()
    monkeypatch.setattr(env_owner, "create_backend", lambda *a, **kw: backend)
    cfg = G1OfficialOracleFlatCfg(reward_config=OfficialOracleRewardConfig(helper.SCALES),
                                 randomization_enabled=False, evaluating=True)
    env = env_owner.G1OfficialOracleFlatEnv(cfg, num_envs=2)
    yield env, cfg, helper
    env.close()


def learner_and_identity(cfg):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "conf/ppo")):
        composed = compose(config_name="config", overrides=["task=g1_official_oracle_flat/mujoco"])
    train = normalize_ppo_train_cfg(OmegaConf.to_container(composed.algo, resolve=True))
    train["multi_gpu"] = None
    train["num_steps_per_env"] = 2
    train["actor"]["hidden_dims"] = [8, 4]
    train["critic"]["hidden_dims"] = [8, 4]
    obs = TensorDict({key: torch.zeros(2, 322) for key in ("actor", "policy", "critic")}, [2])
    algo = OfficialFadaPPO.construct_algorithm(obs, SimpleNamespace(num_envs=2, num_actions=29), copy.deepcopy(train), "cpu")
    identity = make_identity(environment_contract=build_environment_contract(cfg),
        lineage_id="offline-test-run", completed_iterations=25000, role="final",
        actor_config=train["actor"], critic_config=train["critic"],
        reward_value=3.5, reward_statistic="episode_return_mean")
    return algo, identity


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for aa, bb in zip(a, b, strict=True):
            assert_nested_equal(aa, bb)
    else:
        assert a == b


def test_sealed_save_restore_and_raw_actor_equality(route, tmp_path):
    env, cfg, _ = route
    learner, identity = learner_and_identity(cfg)
    # Materialize both AdamW moment states with a finite synthetic gradient.
    for name in ("actor", "critic"):
        optimizer = getattr(learner, name + "_optimizer")
        optimizer.zero_grad()
        sum(p.square().sum() for p in getattr(learner, name).parameters()).backward()
        optimizer.step()
    with torch.no_grad():
        learner.actor.mlp[-1].bias.fill_(3.)
    path = save_oracle_checkpoint(tmp_path / "oracle.pt", learner, identity=identity, samples_seen=96)
    restored, _ = learner_and_identity(cfg)
    cursor = restore_oracle_checkpoint(path, restored, expected_identity=identity)
    assert cursor == {"completed_iterations": 25000, "samples_seen": 96}
    assert_nested_equal(learner.save(), restored.save())
    loaded = load_oracle_policy(path)
    teacher = load_fada_oracle_policy(path, SPEC)
    obs = torch.linspace(-.2, .2, 644).reshape(2, 322)
    # Independent plain MLP forward; proves no hidden tanh, clipping or .25 scaling.
    expected = learner.actor.mlp(obs)
    assert (expected > 1).all()
    torch.testing.assert_close(loaded(obs), expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(teacher.actions_from_env_observation(
        {"obs": obs.numpy(), "critic": obs.numpy()}), expected.detach().numpy(), rtol=1e-5, atol=1e-6)
    assert all(not p.requires_grad for p in loaded.parameters())
    with pytest.raises(FileExistsError):
        save_oracle_checkpoint(path, learner, identity=identity)
    assert not list(tmp_path.glob("*.tmp"))
    validate_environment(identity, env)


def test_resident_reload_and_cross_run_rejection(route, tmp_path):
    _, cfg, _ = route
    learner, final = learner_and_identity(cfg)
    first = dict(final, completed_iterations=100, role="intermediate")
    a = save_oracle_checkpoint(tmp_path / "a.pt", learner, identity=first)
    resident = load_fada_oracle_policy(a, SPEC)
    module_id, parameter_id = id(resident), id(next(resident.parameters()))
    with torch.no_grad():
        learner.actor.mlp[-1].bias.add_(.5)
    second = dict(first, completed_iterations=200)
    b = save_oracle_checkpoint(tmp_path / "b.pt", learner, identity=second)
    reload_fada_oracle_policy_(resident, b, SPEC)
    assert (id(resident), id(next(resident.parameters()))) == (module_id, parameter_id)
    assert resident.checkpoint_identity["completed_iterations"] == 200
    wrong = save_oracle_checkpoint(tmp_path / "other.pt", learner,
        identity=dict(second, oracle_lineage_id="another-run"))
    before = copy.deepcopy(resident.state_dict())
    with pytest.raises(ValueError, match="lineage"):
        reload_fada_oracle_policy_(resident, wrong, SPEC)
    assert_nested_equal(before, resident.state_dict())


def test_twenty_selected_teachers_have_independent_lineage(route):
    _, cfg, _ = route
    _, final = learner_and_identity(cfg)
    intermediate = [dict(final, role="intermediate", completed_iterations=n * 100,
                         reward={"value": float(n), "statistic": "episode_return_mean"})
                    for n in range(1, 21)]
    assert validate_lineage(final, intermediate) == final
    record = make_selection_record(final, intermediate, sampling_mode="reward_aware")
    assert len(record["intermediates"]) == 20
    policies = [LoadedOfficialOracle(item) for item in intermediate]
    validate_loaded_fada_oracle_lineage(LoadedOfficialOracle(final), policies)
    for bad in (intermediate[:-1], [intermediate[0]] * 20,
                intermediate[:-1] + [final],
                intermediate[:-1] + [dict(intermediate[-1], oracle_lineage_id="bad")]):
        with pytest.raises(ValueError):
            validate_lineage(final, bad)
    with pytest.raises(ValueError, match="mixes"):
        validate_loaded_fada_oracle_lineage(LoadedOfficialOracle(final), [torch.nn.Linear(1, 1)])


@pytest.mark.parametrize("kind", ["native", "wrong_teacher_type", "wrong_action_identity", "old_student"])
def test_reject_incompatible_checkpoint_and_student_projection(route, tmp_path, kind):
    _, cfg, _ = route
    learner, identity = learner_and_identity(cfg)
    path = tmp_path / "bad.pt"
    if kind == "native":
        torch.save(learner.save(), path)
        with pytest.raises(ValueError, match="sealed"):
            load_fada_oracle_policy(path, SPEC)
    elif kind == "wrong_teacher_type":
        save_oracle_checkpoint(path, learner, identity=identity)
        with pytest.raises(ValueError, match="algo_type"):
            load_fada_oracle_policy(path, replace(SPEC, algo_type="sac"))
    elif kind == "wrong_action_identity":
        identity["environment_contract"]["action_schema"] = "sac_tanh"
        with pytest.raises(ValueError, match="identity"):
            save_oracle_checkpoint(path, learner, identity=identity)
    else:
        from unilab.algos.torch.distill.fada.observation import project_fada_observation_tensor
        with pytest.raises(ValueError):
            project_fada_observation_tensor(torch.zeros(2, 322), observation_contract=STUDENT_SCHEMA)


def test_command_refresh_terminal_reset_and_snapshot_views(route):
    env, _, _ = route
    env.reset_all()
    commands = np.array([[.6, 0, 0], [.3, 0, 0]], np.float32)
    from unilab.algos.torch.distill.collection.transition import set_transition_input_rows
    obs, info = set_transition_input_rows(env, command_info_key="commands", command_rows=commands)
    steps = env.step_counter
    phase = env.command.phase.copy()
    rng = copy.deepcopy(np.random.get_state())
    env.refresh_state()
    np.testing.assert_equal(env.command.phase, phase)
    np.testing.assert_equal(np.random.get_state()[1], rng[1])
    assert env.step_counter == steps
    np.testing.assert_equal(info["commands"], commands)
    np.testing.assert_equal(student_view(obs["obs"], info, projection=STUDENT_SCHEMA,
        expected_student_obs_dim=66), env.get_student_observation())
    # Hold across the periodic resample boundary and an autoreset.
    env.state.info["steps"][:] = 499
    env.step(np.zeros((2, 29), np.float32))
    np.testing.assert_equal(env.command.velocity, commands)
    snap = env.capture_rollout_snapshot()
    env.set_external_command([0, 0, 0])
    env.restore_rollout_snapshot(snap)
    np.testing.assert_equal(env.command.velocity, commands)
    np.testing.assert_equal(env._external_command_rows, commands)
    env.state.info["steps"][0] = env.cfg.max_episode_steps - 1
    state = env.step(np.full((2, 29), .1, np.float32))
    assert state.info["final_student_observation_mask"][0]
    assert state.info["final_student_observation"].shape == (2, 66)
    np.testing.assert_equal(state.info["student_observation"], env.get_student_observation())
    np.testing.assert_equal(state.info["commands"], commands)
    assert env.obs_groups_spec == {"obs": 322, "critic": 322}
    with pytest.raises(ValueError, match="identity"):
        student_view(state.obs["obs"], {}, projection=STUDENT_SCHEMA, expected_student_obs_dim=66)


def student_config():
    return FADAArchitectureConfig(66, 29, 3, observation_contract=STUDENT_SCHEMA,
        history_length=2, prediction_horizon=2, hidden_dim=8, num_heads=2,
        planner_layers=1, idm_encoder_layers=1, idm_decoder_layers=1, feedforward_dim=16)


def test_shadow_student_views_restore_primary_state_even_on_failure(route, tmp_path):
    env, cfg, helper = route
    learner, identity = learner_and_identity(cfg)
    path = save_oracle_checkpoint(tmp_path / "teacher.pt", learner, identity=identity)
    teacher = load_fada_oracle_policy(path, SPEC)
    env.reset_all()
    env.set_external_command([.6, 0, 0])
    # Replace only the native sibling pool; exercise the real isolation owner.
    backend = env._backend
    primary = backend._pool
    sibling = helper.FakePool(backend)
    backend._isolated_rollout_pool = sibling
    before = env.get_student_observation().copy()
    physics = backend._physics_state.copy()
    commands = env.command.velocity.copy()
    phase = env.command.phase.copy()
    primary_calls = len(primary.calls)
    initial_actions = teacher.actions_from_env_observation(env.state.obs)
    kwargs = dict(initial_oracle_actions=initial_actions, initial_command=commands,
        config=student_config(), observation_key="obs", teacher_projection="identity",
        student_projection=STUDENT_SCHEMA, student_drop_index=None,
        command_info_keys=("commands",))
    futures, actions, valid = _oracle_shadow_pair(env, teacher_policy=teacher, **kwargs)
    assert futures.shape == (2, 2, 66) and actions.shape == (2, 2, 29)
    assert np.isfinite(futures).all() and valid.all()
    assert len(primary.calls) == primary_calls and len(sibling.calls) == 8
    np.testing.assert_equal(backend._physics_state, physics)
    np.testing.assert_equal(env.get_student_observation(), before)
    np.testing.assert_equal(env.state.info["student_observation"], before)
    np.testing.assert_equal(env.command.velocity, commands)
    np.testing.assert_equal(env.command.phase, phase)
    class FailingTeacher:
        obs_dim = 322
        def actions_from_env_observation(self, *args, **kwargs):
            raise RuntimeError("injected shadow failure")
    with pytest.raises(RuntimeError, match="injected shadow failure"):
        _oracle_shadow_pair(env, teacher_policy=FailingTeacher(), **kwargs)
    assert backend._pool is primary and not backend._isolated_rollout_active
    np.testing.assert_equal(backend._physics_state, physics)
    np.testing.assert_equal(env.get_student_observation(), before)
    np.testing.assert_equal(env.state.info["student_observation"], before)


def test_source_windows_artifacts_and_student_checkpoint_roundtrip(route, tmp_path):
    env, cfg, _ = route
    learner, identity = learner_and_identity(cfg)
    path = save_oracle_checkpoint(tmp_path / "teacher.pt", learner, identity=identity)
    teacher = load_fada_oracle_policy(path, SPEC)
    env.reset_all()
    env.set_external_command([.6, 0, 0])
    config = student_config()
    collected = collect_fada_source_windows(env, teacher_policy=teacher, config=config,
        num_windows=3, spec=FADACollectionSpec(student_projection=STUDENT_SCHEMA, max_env_steps=16))
    batch = collected.batch
    assert batch.observation_history.shape == (3, 2, 66)
    assert batch.oracle_action_chunk.shape == (3, 2, 29)
    metadata = {"official_oracle_identity": identity}
    source = save_fada_source_batch(tmp_path / "source.pt", batch, config=config, metadata=metadata)
    loaded = open_fada_source_artifact(source, config=config)
    assert_nested_equal(loaded.load_batch(0).observation_history, batch.observation_history)
    with FADAShardedSourceWriter(tmp_path / "manifest.pt", config=config) as writer:
        writer.append(batch)
        writer.commit(metadata=metadata)
    assert open_fada_source_artifact(tmp_path / "manifest.pt", config=config).num_samples == 3
    with pytest.raises(ValueError, match="identity"):
        save_fada_source_batch(tmp_path / "unbound.pt", batch, config=config, metadata={})
    legacy = replace(config, observation_contract="g1_fada_state_v2")
    with pytest.raises(ValueError, match="architecture mismatch"):
        open_fada_source_artifact(source, config=legacy)
    student = FADAPlannerIDMPolicy(config)
    trainer = FADATrainer(student, idm_optimizer=torch.optim.Adam(student.idm.parameters()),
        planner_optimizer=torch.optim.Adam(student.planner.parameters()))
    student_path = save_fada_checkpoint(tmp_path / "student.pt", student, trainer,
        completed_iterations=0, samples_seen=3, runtime_config=metadata)
    restored = load_fada_policy_checkpoint(student_path)
    assert restored.policy.config == config
    assert_nested_equal(student.state_dict(), restored.policy.state_dict())
    # New model cannot silently lose its action/run identity on disk.
    payload = torch.load(student_path, weights_only=True)
    payload["runtime_config"] = {}
    torch.save(payload, tmp_path / "unbound-student.pt")
    with pytest.raises(ValueError, match="identity"):
        load_fada_policy_checkpoint(tmp_path / "unbound-student.pt")


def test_oracle_playback_and_student_session_use_same_views(route, tmp_path):
    env, cfg, _ = route
    learner, identity = learner_and_identity(cfg)
    path = save_oracle_checkpoint(tmp_path / "oracle.pt", learner, identity=identity)
    playback = RslRlPlaybackConfig(task="G1OfficialOracleFlat", load_run="-1", checkpoint=str(path),
        action_mode="policy", policy_obs_mode="auto", algo_log_name="fada_official_ppo", log_root=None,
        num_envs=2)
    session, mode, _ = create_official_oracle_playback_session(playback_cfg=playback,
        env_factory=lambda n: env, checkpoint_resolver=lambda *a: str(path),
        wrapper_cls=RslRlVecEnvWrapper, device="cpu", log=lambda _: None)
    session.reset()
    session.set_external_command(np.array([.6, 0, 0], np.float32))
    expected = load_fada_oracle_policy(path, SPEC).actions_from_env_observation(env.state.obs)
    np.testing.assert_allclose(session._build_actions().numpy(), expected, rtol=1e-5, atol=1e-6)
    assert mode == "actor"
    policy = FADAPlannerIDMPolicy(student_config()).eval()
    controller = FADAPlaybackController(policy, device="cpu")
    planner_session = FADAPlaybackSession(env=env, wrapped_env=session.wrapped_env,
        device="cpu", action_mode="policy", controller=controller, num_envs=2)
    planner_session.reset()
    planner_session.set_external_command(np.array([.6, 0, 0], np.float32))
    reference = FADAPlaybackController(policy, device="cpu")
    expected = reference.act_projected(env.get_student_observation(), env.command.velocity)
    torch.testing.assert_close(planner_session._build_actions(), expected)
    planner_session.step_once()
    planner_session.reset()
    assert controller._observation_history is None


def test_legacy_sac_dispatch_still_uses_original_loader(monkeypatch, tmp_path):
    from unilab.algos.torch.distill.fada import oracle
    path = tmp_path / "legacy.pt"
    torch.save({"actor_state_dict": {}}, path)
    sentinel = torch.nn.Linear(98, 29)
    calls = []
    def load(*args, **kwargs):
        calls.append(args)
        return sentinel
    monkeypatch.setattr(oracle, "load_sac_teacher_policy", load)
    assert oracle.load_fada_oracle_policy(path, DistillationTeacherSpec(98, 29)) is sentinel
    assert len(calls) == 1
