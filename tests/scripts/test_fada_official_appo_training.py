"""One bounded formal-entry proof: real APPO CPU updates, fake native boundary."""
from copy import deepcopy
import importlib.util
import json
import multiprocessing as mp
from multiprocessing import shared_memory
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensordict import TensorDict

from unilab.algos.torch.appo.runner import APPORunner
from unilab.algos.torch.appo.runtime import resolve_appo_runtime
from unilab.algos.torch.appo.staging import RolloutStagingPool
from unilab.algos.torch.fada_official_appo import runner as runner_owner
from unilab.algos.torch.fada_official_appo.config import validate_training_contract
from unilab.training import fada_official_appo as assembly

ROOT = Path(__file__).resolve().parents[2]
SPAWN = mp.get_context("spawn")


class ScriptedEnv:
    def __init__(self, num_envs, settings):
        self.num_envs, self.settings, self.tick = num_envs, settings, 0

    def reset(self, indices):
        self.state = SimpleNamespace(
            obs={key: np.zeros((self.num_envs, 322), np.float32) for key in ("obs", "critic")},
            reward=np.ones(self.num_envs, np.float32), terminated=np.zeros(self.num_envs, bool),
            truncated=np.zeros(self.num_envs, bool), info={}, final_observation=None)
        return self.state.obs, {}

    def initialize_training_progress(self):
        self.state.obs = {key: np.full((self.num_envs, 322), .07, np.float32) for key in ("obs", "critic")}

    def step(self, actions):
        np.testing.assert_array_equal(self.state.reward, 1)
        self.tick += 1
        if self.tick == self.settings.get("pause_at"):
            self.settings["entered"].set()
            if not self.settings["release"].wait(10):
                raise TimeoutError("scripted pause expired")
        if self.tick == self.settings.get("fail_at"):
            raise RuntimeError("scripted training collector failure")
        state = self.state
        for value in state.obs.values():
            value.fill(self.tick / 1000)
        state.terminated.fill(False)
        state.truncated.fill(False)
        state.final_observation = None
        if self.tick % 24 == 0:
            state.truncated[0], state.terminated[1] = True, True
            state.final_observation = {
                key: np.full((self.num_envs, 322), -.5, np.float32) for key in ("obs", "critic")}
            for value in state.obs.values():
                value.fill(-1)
        return state

    def close(self):
        self.settings["closed"].set()


def fake_env_factory(env_name, *, num_envs, sim_backend, env_cfg_override):
    assert mp.current_process().name != "MainProcess"
    assert env_name == "G1OfficialOracleFlat" and sim_backend == "mujoco"
    return ScriptedEnv(num_envs, env_cfg_override)


@pytest.fixture(autouse=True)
def cpu_budget(monkeypatch):
    threads = torch.get_num_threads()
    monkeypatch.setenv("WANDB_MODE", "disabled")
    try:
        with torch.random.fork_rng(devices=[]):
            torch.set_num_threads(1)
            torch.manual_seed(17)
            yield
    finally:
        torch.set_num_threads(threads)


def configuration(tmp_path, extra=()):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "conf/appo")):
        return compose(config_name="config", overrides=[
            "task=g1_official_oracle_flat/mujoco", "algo.num_envs=2", "algo.max_iterations=2",
            "algo.algorithm.num_learning_epochs=1", "algo.algorithm.num_mini_batches=2",
            "training.device=cpu", "training.nan_guard.enabled=false",
            f"training.log_dir={tmp_path / 'run'}", "algo.collection.wait_timeout_s=15", *extra])


def script():
    spec = importlib.util.spec_from_file_location("appo_training_entry", ROOT / "scripts/train_appo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def settings(**extra):
    return {"entered": SPAWN.Event(), "release": SPAWN.Event(), "closed": SPAWN.Event(), **extra}


def no_models_written(*args, **kwargs):
    raise AssertionError("step 3 must never read or write a checkpoint")


def check_clean(runner):
    assert runner._collector_process is None and runner._error_recv is None
    assert runner._error_send is None and runner.metrics_queue is None
    for resource in runner._shared_resources:
        for name in resource.name.values() if isinstance(resource.name, dict) else [resource.name]:
            with pytest.raises(FileNotFoundError):
                shared_memory.SharedMemory(name=name, create=False)
    runner.close()


def wait_for(predicate, runner):
    deadline = time.monotonic() + 10
    while not predicate():
        runner._assert_running()
        if time.monotonic() >= deadline:
            raise AssertionError("collector did not make bounded progress")
        time.sleep(.005)


@pytest.mark.parametrize("max_lag", [0, 2])
def test_formal_entry_real_updates_overlap_and_use_versioned_fresh_batches(tmp_path, monkeypatch, max_lag):
    cfg = configuration(tmp_path, [f"algo.collection.max_policy_lag={max_lag}"])
    opts = settings(pause_at=25)
    observed, batches, records, overlap = [], [], [], []
    actual_runner = assembly.OfficialFadaAPPORunner

    def construct(env_name, env_cfg_overrides, rl_cfg, **kwargs):
        runner = actual_runner(env_name, {**env_cfg_overrides, **opts}, rl_cfg,
                               **kwargs, env_factory=fake_env_factory)
        observed.append(runner)
        learner = runner.learner = runner._build_learner()
        initial_actor = deepcopy(learner.actor)
        initial_parameters = [value.detach().clone() for value in learner.actor.parameters()]
        original_process = learner.process_batch
        original_update = learner.update
        original_learn = runner.learn

        def process(batch):
            assert "returns" not in batch and "_critic_obs_flat" not in batch
            batches.append({key: value.detach().clone() for key, value in batch.items()})
            if len(batches) == 1:
                assert batch["observations"].shape == (24, 2, 322)
                obs = batch["observations"].flatten(0, 1)
                with torch.no_grad():
                    initial_actor(TensorDict({"policy": obs}, [obs.shape[0]]), stochastic_output=True)
                    expected = initial_actor.get_output_log_prob(batch["actions"].flatten(0, 1)).view(24, 2)
                    torch.testing.assert_close(batch["actions_log_prob"], expected)
                    final_value = learner.critic(TensorDict({"policy": torch.full((2, 322), -.5)}, [2])).squeeze(-1)
                expected_timeout = 1 + learner.gamma * final_value[0]
                torch.testing.assert_close(batch["rewards"][-1, 0], expected_timeout)
                torch.testing.assert_close(batch["rewards"][-1, 1], torch.tensor(1.))
            result = original_process(batch)
            if len(batches) == 1:
                # At done, V-trace cannot add reset-state bootstrap a second time.
                torch.testing.assert_close(result["returns"][-1], result["rewards"][-1], atol=1e-5, rtol=1e-5)
            return result

        def slow_optimizer(optimizer, args, kwargs):
            if overlap:
                return
            assert opts["entered"].wait(10)
            before = runner.generated_samples.value
            opts["release"].set()
            # This hook runs inside the real learner.update -> optimizer.step.
            wait_for(lambda: runner.ring.available() == 4, runner)
            assert runner.generated_samples.value > before
            overlap.append(True)
        learner.optimizer.register_step_pre_hook(slow_optimizer)
        learner.process_batch = process

        def update(batch):
            result = original_update(batch)
            assert all(np.isfinite(value) for value in result.values())
            return result
        learner.update = update

        def learn(**kw):
            callback = kw.pop("on_update")
            def updated(event):
                records.append(event)
                callback(event)
            result = original_learn(**kw, on_update=updated)
            assert any(not torch.equal(before, after) for before, after in zip(initial_parameters, learner.actor.parameters()))
            return result
        runner.learn = learn
        return runner

    monkeypatch.setattr(assembly, "OfficialFadaAPPORunner", construct)
    try:
        summary = script().main.__wrapped__(cfg)
    finally:
        opts["release"].set()
        for runner in observed:
            runner.close()
    runner = observed[0]
    assert overlap and opts["closed"].is_set()
    assert summary["status"] == "completed" and summary["stop_reason"] == "max_iterations"
    assert summary["learner_updates"] == 2 and summary["optimizer_steps"] == 4
    assert summary["reused_samples"] == 0
    assert summary["accepted_samples"] == summary["unique_used_samples"]
    assert summary["sample_exposures"] == summary["unique_used_samples"]
    ids = [meta["rollout_id"] for record in records for meta in record["rollouts"]]
    assert len(ids) == len(set(ids))
    assert all(len(record["rollouts"]) <= 3 for record in records)
    assert summary["unique_used_samples"] == sum(batch["actions"].shape[0] * batch["actions"].shape[1] for batch in batches)
    assert summary["generated_samples"] >= summary["published_samples"]
    assert summary["generated_samples"] == (summary["unique_used_samples"] + summary["stale_dropped_samples"]
                                             + summary["shutdown_unused_samples"])
    if max_lag == 0:
        assert summary["stale_dropped_samples"] >= 4 * 48
        assert all(meta["policy_version"] == 2 for meta in records[1]["rollouts"])
    else:
        assert summary["stale_dropped_samples"] == 0
        assert all(meta["policy_version"] == 1 for meta in records[1]["rollouts"])
    assert summary["episode_window_count"] > 0 and summary["raw_mean_episode_return"] == 24
    assert summary["last_checkpoint"] is not None and summary["persistence_available"] is True
    assert all(value >= 0 for value in summary["timing_totals"].values())
    assert list(tmp_path.rglob("appo_final_update_2.pt"))
    assert list(tmp_path.rglob("run_config.json"))
    check_clean(runner)


@pytest.mark.parametrize("budget", [48, 96])
def test_unique_sample_budget_limits_aggregation_without_counting_epochs_again(tmp_path, budget):
    cfg = configuration(tmp_path, [f"algo.max_unique_samples={budget}", "algo.max_iterations=10",
                                  "algo.algorithm.num_learning_epochs=2"])
    algo = OmegaConf.to_container(cfg.algo, resolve=True)
    opts = settings()
    runner = runner_owner.OfficialFadaAPPORunner("G1OfficialOracleFlat", opts, algo,
        device="cpu", env_factory=fake_env_factory)
    try:
        summary = runner.learn()
    finally:
        runner.close()
    assert summary["stop_reason"] == "max_unique_samples"
    assert summary["unique_used_samples"] == summary["accepted_samples"] == budget
    assert summary["sample_exposures"] == 2 * budget
    assert summary["reused_samples"] == 0
    assert summary["learner_updates"] <= budget // 48
    assert opts["closed"].is_set()
    check_clean(runner)


@pytest.mark.parametrize("failure", ["update", "collector"])
def test_failures_release_resources_and_do_not_claim_failed_batch_as_used(tmp_path, failure):
    cfg = configuration(tmp_path)
    algo = OmegaConf.to_container(cfg.algo, resolve=True)
    opts = settings(**({"fail_at": 3} if failure == "collector" else {}))
    runner = runner_owner.OfficialFadaAPPORunner("G1OfficialOracleFlat", opts, algo,
        device="cpu", env_factory=fake_env_factory)
    if failure == "update":
        runner.learner = runner._build_learner()
        def fail(batch):
            raise RuntimeError("scripted learner update failure")
        runner.learner.update = fail
    with pytest.raises(RuntimeError, match="scripted"):
        runner.learn()
    assert runner.last_run_summary["status"] == "failed"
    assert runner.last_run_summary["unique_used_samples"] == 0
    assert runner.last_run_summary["learner_updates"] == 0
    assert runner.last_run_summary["optimizer_steps"] == 0
    assert opts["closed"].is_set()
    check_clean(runner)


@pytest.mark.parametrize("override,message", [
    ("algo.save_interval=-1", "save_interval"), ("algo.load_run=some_user_run", "explicit"),
    ("algo.max_unique_samples=49", "multiple"), ("algo.algorithm.num_mini_batches=5", "divisible"),
    ("algo.algorithm.enable_compile=true", "compile"), ("algo.collection.max_policy_lag=-1", "max_policy_lag"),
])
def test_formal_gates_reject_before_logs_or_checkpoint_resolution(tmp_path, monkeypatch, override, message):
    cfg = configuration(tmp_path, [override])
    monkeypatch.setattr(torch, "save", no_models_written)
    monkeypatch.setattr(torch, "load", no_models_written)
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid request must fail before runner/log allocation")
    monkeypatch.setattr(assembly, "OfficialFadaAPPORunner", forbidden)
    with pytest.raises(ValueError, match=message):
        script().main.__wrapped__(cfg)
    assert not (tmp_path / "run").exists()


def test_legacy_runtime_and_staging_keep_default_behavior_and_reset_reuses_storage(tmp_path):
    play = lambda *args, **kwargs: None
    legacy = resolve_appo_runtime({}, default_play_fn=play)
    assert legacy.runner_cls is APPORunner and legacy.play_fn is play and legacy.train_fn is None
    algo = OmegaConf.to_container(configuration(tmp_path).algo, resolve=True)
    selected = resolve_appo_runtime(algo, default_play_fn=play)
    assert selected.runner_cls is runner_owner.OfficialFadaAPPORunner and selected.train_fn is assembly.run_training
    from unilab.algos.torch.fada_official_appo.playback import play_checkpoint
    assert selected.play_fn is play_checkpoint
    pool = RolloutStagingPool(capacity=3, num_envs=2, slot_shapes={"obs": (2, 24, 322)}, device="cpu")
    for value in (1, 2, 3, 4):
        pool.stage_numpy_views({"obs": np.full((2, 24, 322), value, np.float32)})
    assert pool.active_count == 3 and pool.slot_versions == (3, 1, 2)
    pointer = pool.batch()["observations"].data_ptr()
    pool.reset()
    with pytest.raises(RuntimeError, match="no active"):
        pool.batch()
    pool.stage_numpy_views({"obs": np.full((2, 24, 322), 7, np.float32)})
    batch = pool.batch()["observations"]
    assert batch.data_ptr() == pointer and batch.shape == (24, 2, 322)
    torch.testing.assert_close(batch, torch.full_like(batch, 7))
