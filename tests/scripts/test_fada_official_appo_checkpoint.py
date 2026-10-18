"""Bounded CPU persistence proof; all artifacts temporary, physics replaced once."""
from copy import deepcopy

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from tests.scripts.test_fada_official_appo_training import (
    ROOT, check_clean, configuration, cpu_budget, fake_env_factory, script, settings,
)
from unilab.algos.torch.fada_official_appo import checkpoint as artifacts
from unilab.algos.torch.fada_official_appo import runner as runner_owner
from unilab.base.registry import apply_cfg_overrides
from unilab.envs.locomotion.g1.official_oracle.config import (
    G1OfficialOracleFlatCfg, build_environment_contract,
)
from unilab.training import fada_official_appo as assembly
from unilab.training.backend_adapter import BackendAdapter


def assert_same(one, two):
    if isinstance(one, torch.Tensor):
        torch.testing.assert_close(one, two, rtol=0, atol=0)
    elif isinstance(one, dict):
        assert one.keys() == two.keys()
        for key in one:
            assert_same(one[key], two[key])
    elif isinstance(one, (list, tuple)):
        assert len(one) == len(two)
        for a, b in zip(one, two, strict=True):
            assert_same(a, b)
    else:
        assert one == two


def cold_environment(cfg):
    env = G1OfficialOracleFlatCfg()
    apply_cfg_overrides(env, BackendAdapter(cfg, root_dir=ROOT, algo_name="appo").build_task_env_cfg_override())
    env.validate()
    return build_environment_contract(env)


def build_learner(algo):
    runner = runner_owner.OfficialFadaAPPORunner("G1OfficialOracleFlat", {}, algo, device="cpu")
    try:
        return runner._build_learner()
    finally:
        runner.close()


def raw_batch(learner):
    obs = torch.randn(24, 2, 322) * .05
    td = TensorDict({"policy": obs.flatten(0, 1)}, [48])
    with torch.no_grad():
        actions = learner.actor(td, stochastic_output=True)
        logp = learner.actor.get_output_log_prob(actions)
    return dict(observations=obs, critic=obs.clone(), actions=actions.reshape(24, 2, 29).clone(),
        actions_log_prob=logp.reshape(24, 2).clone(), rewards=torch.randn(24, 2) * .1 + 1,
        dones=torch.zeros(24, 2), last_obs=obs[-1].clone(), last_critic=obs[-1].clone())


def advance(learner, raw):
    batch = deepcopy(raw)
    learner.train_mode()
    learner.process_batch(batch)
    learner.update(batch)


@pytest.fixture
def saved(tmp_path):
    cfg = configuration(tmp_path, ["algo.algorithm.tau=0.3", "algo.algorithm.target_update_freq=2"])
    algo = OmegaConf.to_container(cfg.algo, resolve=True)
    learner = build_learner(algo)
    advance(learner, raw_batch(learner))
    assert learner._update_counter == 1
    assert learner.learning_rate != algo["algorithm"]["learning_rate"]
    assert any(not torch.equal(v, learner.target_actor.state_dict()[k])
               for k, v in learner.actor.state_dict().items())
    static = artifacts.training_contract(algo, num_envs=2, replay_queue_size=3)
    identity = dict(schema=artifacts.ORACLE_SCHEMA, algorithm=artifacts.ALGORITHM,
        oracle_lineage_id="synthetic-lineage", session_id="session-one", parent_checkpoint=None,
        completed_updates=1, role="intermediate", environment_contract=cold_environment(cfg),
        training_contract=static, actor_config=algo["actor"], critic_config=algo["critic"],
        reward=dict(value=None, window_count=0, statistic="raw_episode_return_last100", session_id="session-one"))
    cursor = dict(generated_samples=96, published_samples=96, accepted_samples=48, unique_used_samples=48,
        reused_samples=0, stale_dropped_samples=0, abandoned_samples=0, learner_updates=1,
        optimizer_steps=2, sample_exposures=48)
    path = artifacts.save_oracle_checkpoint(tmp_path / "oracle.pt", learner, identity=identity, cursor=cursor)
    return learner, algo, identity, cursor, path


def test_roundtrip_restores_target_adam_lr_counter_and_next_real_update(saved):
    learner, algo, identity, cursor, path = saved
    assert set(learner.get_state_dict()) == {"actor", "critic", "optimizer"}
    restored = build_learner(algo)
    payload = artifacts.read_oracle_checkpoint(path)
    observed = artifacts.restore_oracle_checkpoint(payload, restored,
        environment_contract=identity["environment_contract"], static_contract=identity["training_contract"],
        lineage_id="synthetic-lineage")
    assert observed == cursor
    assert_same(learner.get_state_dict(complete=True), restored.get_state_dict(complete=True))
    assert not restored.target_actor.training
    assert all(not p.requires_grad for p in restored.target_actor.parameters())
    batch = raw_batch(learner)
    rng = torch.get_rng_state()
    advance(learner, batch)
    torch.set_rng_state(rng)
    advance(restored, batch)
    assert learner._update_counter == 2
    assert_same(learner.get_state_dict(complete=True), restored.get_state_dict(complete=True))


def test_frozen_loader_raw_actions_rng_and_legacy_isolation(saved, tmp_path):
    from unilab.algos.torch.fada_official.checkpoint import read_oracle_checkpoint as read_ppo
    learner, _, identity, cursor, _ = saved
    with torch.no_grad():
        learner.actor.mlp[-1].bias.fill_(3)
    path = artifacts.save_oracle_checkpoint(tmp_path / "raw.pt", learner, identity=identity, cursor=cursor)
    rng = torch.get_rng_state()
    policy = artifacts.load_oracle_policy(path)
    assert torch.equal(rng, torch.get_rng_state())
    obs = torch.zeros(2, 322)
    expected = learner.actor.mlp(obs)
    assert (expected > 1).all()
    assert_same(policy(obs), expected)
    assert not policy.training and all(not p.requires_grad for p in policy.parameters())
    with pytest.raises(ValueError):
        read_ppo(path)
    with pytest.raises(ValueError):
        policy(torch.zeros(2, 66))


@pytest.mark.parametrize("damage", ["target_nan", "critic_shape", "moment_shape", "moment_missing",
                                     "lr", "counter", "step", "mapping", "schema"])
def test_invalid_complete_state_does_not_mutate_live_learner(saved, damage):
    _, algo, identity, _, path = saved
    learner = build_learner(algo)
    before = artifacts.cpu_snapshot(learner.get_state_dict(complete=True))
    payload = artifacts.read_oracle_checkpoint(path)
    state = payload["learner"]
    moments = state["optimizer"]["state"]
    first = next(iter(moments))
    if damage == "target_nan":
        next(iter(state["target_actor"].values())).fill_(float("nan"))
    elif damage == "critic_shape":
        state["critic"][next(iter(state["critic"]))] = torch.zeros(1)
    elif damage == "moment_shape":
        moments[first]["exp_avg"] = torch.zeros(1)
    elif damage == "moment_missing":
        del moments[first]
    elif damage == "lr":
        state["learning_rate"] *= 2
    elif damage == "counter":
        state["update_counter"] = 99
    elif damage == "step":
        moments[first]["step"].fill_(-1)
    elif damage == "mapping":
        state["optimizer"]["param_groups"][0]["params"].reverse()
    else:
        state["learner_schema"] = "old-bare-appo"
    with pytest.raises(ValueError):
        artifacts.restore_oracle_checkpoint(payload, learner,
            environment_contract=identity["environment_contract"], static_contract=identity["training_contract"])
    assert_same(before, learner.get_state_dict(complete=True))


@pytest.mark.parametrize("damage", ["algorithm", "lineage", "environment", "model", "training", "cursor"])
def test_identity_or_cursor_mismatch_is_rejected_before_mutation(saved, damage):
    _, algo, identity, _, path = saved
    payload = artifacts.read_oracle_checkpoint(path)
    observed = payload["official_appo_oracle"]
    if damage == "algorithm":
        observed["algorithm"] = "fada_official_ppo"
    elif damage == "lineage":
        observed["oracle_lineage_id"] = "different-lineage"
    elif damage == "environment":
        observed["environment_contract"]["config_sha256"] = "0" * 64
    elif damage == "model":
        observed["actor_config"]["hidden_dims"] = [8]
    elif damage == "training":
        observed["training_contract"]["algorithm"]["gamma"] = .5
    else:
        payload["cursor"]["unique_used_samples"] = 999
    learner = build_learner(algo)
    before = artifacts.cpu_snapshot(learner.get_state_dict(complete=True))
    with pytest.raises(ValueError):
        artifacts.restore_oracle_checkpoint(payload, learner, environment_contract=identity["environment_contract"],
            static_contract=identity["training_contract"], lineage_id="synthetic-lineage")
    assert_same(before, learner.get_state_dict(complete=True))


def test_atomic_save_never_overwrites_and_publication_failure_cleans_tmp(saved, tmp_path, monkeypatch):
    learner, _, identity, cursor, path = saved
    contents = path.read_bytes()
    with pytest.raises(FileExistsError):
        artifacts.save_oracle_checkpoint(path, learner, identity=identity, cursor=cursor)
    assert path.read_bytes() == contents
    def fail(*args):
        raise OSError("synthetic publish failure")
    monkeypatch.setattr(artifacts.os, "link", fail)
    with pytest.raises(OSError, match="publish failure"):
        artifacts.save_oracle_checkpoint(tmp_path / "failed.pt", learner, identity=identity, cursor=cursor)
    assert not (tmp_path / "failed.pt").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_formal_entry_saves_and_resumes_cumulative_budget_with_fresh_session(tmp_path, monkeypatch):
    original = assembly.OfficialFadaAPPORunner
    runners, batches, pre_resume = [], [], []
    def construct(env_name, overrides, algo, **kwargs):
        opts = settings()
        runner = original(env_name, {**overrides, **opts}, algo, env_factory=fake_env_factory, **kwargs)
        runners.append(runner)
        old_restore, old_learn = runner.restore_checkpoint, runner.learn
        def restore(*args, **kw):
            old_restore(*args, **kw)
            pre_resume.append(runner._sample_counts())
            assert runner.latest_episode_return is None and runner.episode_window_count == 0
            assert runner.ring is None and runner._last_rollout_id == 0
        runner.restore_checkpoint = restore
        def learn(**kw):
            callback = kw.pop("on_update")
            def updated(event):
                batches.append(event)
                callback(event)
            return old_learn(**kw, on_update=updated)
        runner.learn = learn
        return runner
    monkeypatch.setattr(assembly, "OfficialFadaAPPORunner", construct)
    first_cfg = configuration(tmp_path / "first", ["algo.max_iterations=1", "algo.save_interval=1"])
    first = script().main.__wrapped__(first_cfg)
    first_path = first["last_checkpoint"]
    first_payload = artifacts.read_oracle_checkpoint(first_path)
    assert first_payload["official_appo_oracle"]["role"] == "final"
    assert first_payload["cursor"]["learner_updates"] == 1
    assert first["persistence_available"]
    assert (tmp_path / "first/run/checkpoints/appo_update_1.pt").exists()
    assert batches[0]["rollouts"][0]["policy_version"] == 1
    count_before_resume = len(batches)
    second_cfg = configuration(tmp_path / "second", [f"algo.resume_path={first_path}", "algo.save_interval=0"])
    second = script().main.__wrapped__(second_cfg)
    final = artifacts.read_oracle_checkpoint(second["last_checkpoint"])
    assert second["learner_updates"] == 2 and second["optimizer_steps"] == 4
    assert len(batches) == count_before_resume + 1 and batches[-1]["iteration"] == 2
    assert batches[-1]["rollouts"][0]["policy_version"] == 1
    assert batches[-1]["rollouts"][0]["rollout_id"] == 1
    cursor = first_payload["cursor"]
    assert pre_resume[0]["abandoned_samples"] == (cursor["generated_samples"]
        - cursor["unique_used_samples"] - cursor["stale_dropped_samples"])
    assert second["generated_samples"] == (second["unique_used_samples"] + second["stale_dropped_samples"]
        + second["abandoned_samples"] + second["shutdown_unused_samples"])
    assert final["official_appo_oracle"]["session_id"] != first_payload["official_appo_oracle"]["session_id"]
    assert final["official_appo_oracle"]["oracle_lineage_id"] == first_payload["official_appo_oracle"]["oracle_lineage_id"]
    assert final["official_appo_oracle"]["parent_checkpoint"] == first_path
    assert not list((tmp_path / "second/run/checkpoints").glob("appo_update_*.pt"))
    # Already-complete paths validate state, but never start IPC or allocate logs.
    already_cfg = configuration(tmp_path / "already", [f"algo.resume_path={second['last_checkpoint']}"])
    already = script().main.__wrapped__(already_cfg)
    assert already["status"] == "already_complete" and not (tmp_path / "already/run").exists()
    assert runners[-1].ring is None
    # Unique-sample target is cumulative too, independent of the update ceiling.
    sample_cfg = configuration(tmp_path / "sample_done", [f"algo.resume_path={second['last_checkpoint']}",
        "algo.max_iterations=10", f"algo.max_unique_samples={second['unique_used_samples']}"])
    assert script().main.__wrapped__(sample_cfg)["status"] == "already_complete"
    assert not (tmp_path / "sample_done/run").exists()
    for runner in runners:
        check_clean(runner)


def test_formal_failure_retains_periodic_but_never_seals_partial_final(tmp_path, monkeypatch):
    original = assembly.OfficialFadaAPPORunner
    observed = []
    def construct(env_name, overrides, algo, **kwargs):
        runner = original(env_name, {**overrides, **settings()}, algo, env_factory=fake_env_factory, **kwargs)
        observed.append(runner)
        learner = runner.learner = runner._build_learner()
        update = learner.update
        def fail_second(batch):
            if learner._update_counter == 1:
                raise RuntimeError("synthetic second update failure")
            return update(batch)
        learner.update = fail_second
        return runner
    monkeypatch.setattr(assembly, "OfficialFadaAPPORunner", construct)
    cfg = configuration(tmp_path, ["algo.save_interval=1"])
    with pytest.raises(RuntimeError, match="second update failure"):
        script().main.__wrapped__(cfg)
    assert (tmp_path / "run/checkpoints/appo_update_1.pt").exists()
    assert not list(tmp_path.rglob("appo_final*.pt"))
    check_clean(observed[0])
