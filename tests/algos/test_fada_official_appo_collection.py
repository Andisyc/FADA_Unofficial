"""Small, bounded transport proof: actual spawn/worker, no physics or optimizer steps."""
from copy import deepcopy
import multiprocessing as mp
from multiprocessing import shared_memory
from pathlib import Path
from queue import Empty
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.algos.torch.fada_official_appo.config import validate_collection_contract
from unilab.algos.torch.fada_official_appo.rollout import HEADERS, split_rollout
from unilab.algos.torch.fada_official_appo.runner import OfficialFadaAPPORunner
from unilab.algos.torch.fada_official_appo.runtime import resolve_runtime
from unilab.ipc.memory_budget import estimate_fada_appo_bytes
from unilab.ipc.rollout_ring_buffer import (
    FADA_APPO_PROTOCOL, FADA_PPO_PROTOCOL, RolloutRingBuffer, rollout_field_specs,
)
from unilab.ipc.weight_sync import SharedWeightSync

ROOT = Path(__file__).resolve().parents[2]
SPAWN = mp.get_context("spawn")


class ScriptedEnv:
    def __init__(self, num_envs, settings):
        self.num_envs, self.settings, self.tick = num_envs, settings, 0

    def reset(self, indices):
        self.settings["resets"].value += 1
        self.state = SimpleNamespace(
            obs={key: np.zeros((self.num_envs, 322), np.float32) for key in ("obs", "critic")},
            reward=np.ones(self.num_envs, np.float32), terminated=np.zeros(self.num_envs, bool),
            truncated=np.zeros(self.num_envs, bool), info={}, final_observation=None)
        return self.state.obs, {}

    def initialize_training_progress(self):
        assert self.tick == 0
        self.settings["initialized"].value += 1
        # Replace, rather than mutate, the observation returned by reset.
        self.state.obs = {key: np.full((self.num_envs, 322), 7, np.float32) for key in ("obs", "critic")}

    def step(self, actions):
        # Critic timeout correction must never mutate the environment's reward array.
        np.testing.assert_array_equal(self.state.reward, 1)
        self.tick += 1
        with self.settings["counter"].get_lock():
            self.settings["counter"].value += 1
        if self.tick == self.settings.get("pause_at"):
            self.settings["entered"].set()
            if not self.settings["release"].wait(10):
                raise TimeoutError("scripted pause expired")
        if self.tick == self.settings.get("fail_at"):
            raise RuntimeError("scripted APPO step failure")
        state = self.state
        for value in state.obs.values():
            value.fill(self.tick)
        state.terminated.fill(False)
        state.truncated.fill(False)
        state.final_observation = None
        if self.tick % 24 == 0:
            state.truncated[0], state.terminated[1] = True, True
            if not self.settings.get("missing_final"):
                state.final_observation = {
                    key: np.full((self.num_envs, 322), 10, np.float32) for key in ("obs", "critic")}
            for value in state.obs.values():
                value.fill(-100)  # autoreset input, not the terminal critic input
        return state

    def close(self):
        self.settings["closed"].set()


def scripted_env_factory(env_name, *, num_envs, sim_backend, env_cfg_override):
    assert env_name == "G1OfficialOracleFlat" and sim_backend == "mujoco"
    if env_cfg_override.get("construct_fail"):
        raise RuntimeError("scripted APPO construction failure")
    return ScriptedEnv(num_envs, env_cfg_override)


@pytest.fixture(autouse=True)
def cpu_budget():
    threads = torch.get_num_threads()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.set_num_threads(1)
            torch.manual_seed(7)
            yield
    finally:
        torch.set_num_threads(threads)


def config():
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "conf/appo")):
        cfg = OmegaConf.to_container(compose(config_name="config",
            overrides=["task=g1_official_oracle_flat/mujoco"]).algo, resolve=True)
    cfg["num_envs"] = 2
    cfg["collection"]["wait_timeout_s"] = 15.
    return cfg


def settings(**extra):
    return {"counter": SPAWN.Value("i", 0), "resets": SPAWN.Value("i", 0),
            "initialized": SPAWN.Value("i", 0), "closed": SPAWN.Event(),
            "entered": SPAWN.Event(), "release": SPAWN.Event(), **extra}


def make_runner(opts):
    runner = OfficialFadaAPPORunner("G1OfficialOracleFlat", opts, config(), device="cpu",
                                  env_factory=scripted_env_factory)
    learner = runner._build_learner()
    set_markers(learner, 1)
    return runner, learner


def set_markers(learner, marker):
    with torch.no_grad():
        for model in (learner.actor, learner.critic):
            for parameter in model.mlp.parameters():
                parameter.zero_()
        learner.actor.mlp[-1].bias.fill_(marker)
        learner.critic.mlp[-1].bias.fill_(10 * marker)


def wait_until(predicate, runner):
    deadline = time.monotonic() + 15
    while not predicate():
        runner._assert_running()
        if time.monotonic() >= deadline:
            raise AssertionError("bounded collector condition timed out")
        SPAWN.Event().wait(.01)


def names(runner):
    return [name for resource in runner._shared_resources
            for name in (resource.name.values() if isinstance(resource.name, dict) else [resource.name])]


def assert_closed(runner, allocated):
    assert runner._collector_process is None
    assert runner.metrics_queue is None and runner._error_recv is None and runner._error_send is None
    for name in allocated:
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=name, create=False)
    runner.close()  # idempotent


def owned_copy(payload):
    assert not set(HEADERS).intersection(payload)
    return {key: value.copy() for key, value in payload.items()}


def test_real_spawn_continuous_sampling_joint_versions_raw_returns_and_backpressure():
    opts = settings(pause_at=5)
    runner, learner = make_runner(opts)
    allocated = []
    try:
        runner.start_collection(learner.actor, learner.critic)
        allocated = names(runner)
        assert opts["entered"].wait(15)
        set_markers(learner, 2)
        assert runner.publish_weights(learner.actor, learner.critic) == 2
        set_markers(learner, 3)
        assert runner.publish_weights(learner.actor, learner.critic) == 3
        opts["release"].set()
        wait_until(lambda: runner.ring.available() == 4, runner)
        assert opts["counter"].value == 96
        borrowed = runner.ring.read_numpy_views()
        before = borrowed["obs"].copy()

        def slow_consumer(payload):
            assert not SPAWN.Event().wait(.05)
            assert opts["counter"].value == 96  # no ACK, no overwrite
            np.testing.assert_array_equal(borrowed["obs"], before)
            return owned_copy(payload)

        first, meta = runner.consume_rollout(slow_consumer)
        assert meta == {"rollout_id": 1, "policy_version": 1, "valid_steps": 24}
        assert first["obs"][0, 0, 0] == 7  # refreshed initial phase input
        np.testing.assert_allclose(first["rewards"][0, -1], 1 + .99 * 10)
        np.testing.assert_array_equal(first["rewards"][1], 1)  # termination is not timeout
        wait_until(lambda: opts["counter"].value == 120, runner)
        # Collector advanced with no learner update or additional weight publication.
        np.testing.assert_array_equal(first["obs"], before)
        assert not np.array_equal(borrowed["obs"], before)  # released slot was reused
        second, meta = runner.consume_rollout(owned_copy)
        assert meta["rollout_id"] == 2 and meta["policy_version"] == 3
        np.testing.assert_allclose(second["rewards"][0, -1], 1 + .99 * 30)
        assert second["obs"][0, 0, 0] == -100  # no reset between rollouts
        for payload, mean in ((first, 1.), (second, 3.)):
            reference = torch.distributions.Normal(mean, .8).log_prob(torch.from_numpy(payload["actions"])).sum(-1)
            np.testing.assert_allclose(payload["log_probs"], reference.numpy(), rtol=2e-5, atol=1e-4)
        msg = runner.metrics_queue.get(timeout=5)
        assert msg["mean_ep_reward"] == 24  # excludes gamma * V(final)
        assert opts["resets"].value == opts["initialized"].value == 1
        assert not set(HEADERS).intersection(runner.payload_shapes)
        with pytest.raises(RuntimeError, match="already started"):
            runner.learn(1)
        assert callable(resolve_runtime(config()).train_fn)
    finally:
        opts["release"].set()
        runner.close()
    assert opts["closed"].is_set() and runner.collector_exitcode == 0
    assert_closed(runner, allocated)


@pytest.mark.parametrize("failure,message", [
    ({"construct_fail": True}, "construction failure"),
    ({"fail_at": 3}, "step failure"),
    ({"missing_final": True}, "pre-reset final obs and critic"),
])
def test_real_spawn_failures_propagate_and_release_partial_data(failure, message):
    opts = settings(**failure)
    runner, learner = make_runner(opts)
    try:
        runner.start_collection(learner.actor, learner.critic)
        allocated = names(runner)
        with pytest.raises(RuntimeError, match=message):
            runner.consume_rollout(owned_copy)
    finally:
        runner.close()
    assert_closed(runner, allocated)
    assert runner.collector_exitcode != 0
    if not failure.get("construct_fail"):
        assert opts["closed"].is_set()


def test_stop_inside_rollout_does_not_publish_and_closes_environment():
    opts = settings(pause_at=5)
    runner, learner = make_runner(opts)
    try:
        runner.start_collection(learner.actor, learner.critic)
        allocated = names(runner)
        assert opts["entered"].wait(15)
        runner._stop_event.set()
        opts["release"].set()
        runner._collector_process.join(timeout=5)
        assert not runner._collector_process.is_alive()
        assert runner.ring.available() == 0
    finally:
        opts["release"].set()
        runner.close()
    assert opts["closed"].is_set() and runner.collector_exitcode == 0
    assert_closed(runner, allocated)


def test_consumer_failure_stops_full_collector_and_unlinks_resources():
    opts = settings()
    runner, learner = make_runner(opts)
    def fail(payload):
        raise ValueError("consumer copy failure")
    try:
        runner.start_collection(learner.actor, learner.critic)
        allocated = names(runner)
        wait_until(lambda: runner.ring.available() == 4, runner)
        with pytest.raises(ValueError, match="consumer copy failure"):
            runner.consume_rollout(fail)
    finally:
        runner.close()
    assert opts["closed"].is_set() and runner.collector_exitcode == 0
    assert_closed(runner, allocated)


def test_parent_initial_weight_failure_cleans_already_allocated_resources(monkeypatch):
    runner, learner = make_runner(settings())
    def fail(*args):
        raise ValueError("initial weight failure")
    monkeypatch.setattr(SharedWeightSync, "write_weights", fail)
    with pytest.raises(ValueError, match="initial weight failure"):
        runner.start_collection(learner.actor, learner.critic)
    assert_closed(runner, names(runner))


def test_int64_metadata_ownership_memory_and_legacy_protocol_sentinels():
    args = dict(num_envs=2, num_steps=24, obs_dim=322, action_dim=29, critic_dim=322, num_slots=2)
    ring = RolloutRingBuffer(**args, protocol=FADA_APPO_PROTOCOL)
    try:
        for rollout_id in (1, 2):
            slot = ring.write_buffer
            slot["rollout_id"][0] = rollout_id
            slot["policy_version"][0] = 2**24 + 7
            with pytest.raises(ValueError, match="incomplete"):
                ring.signal_write_done()
            slot["valid_steps"][0] = 24
            ring.signal_write_done()
        with pytest.raises(BufferError):
            _ = ring.write_buffer
        payload = ring.read_numpy_views()
        _, meta = split_rollout(payload, previous_rollout_id=0, published_version=2**24+7, num_steps=24)
        assert meta["policy_version"] == 2**24 + 7
        for overrides in ({"previous_rollout_id": 1}, {"published_version": 1}, {"num_steps": 23}):
            call = dict(previous_rollout_id=0, published_version=2**24+7, num_steps=24)
            call.update(overrides)
            with pytest.raises(ValueError):
                split_rollout(payload, **call)
        bad = dict(payload, policy_version=np.ones(1, np.float32))
        with pytest.raises(ValueError, match="int64"):
            split_rollout(bad, previous_rollout_id=0, published_version=1, num_steps=24)
        specs = rollout_field_specs(**args, protocol=FADA_APPO_PROTOCOL)
        estimate = estimate_fada_appo_bytes(num_envs=2, steps_per_env=24, obs_dim=322,
            action_dim=29, critic_dim=322, num_slots=2, weight_param_shapes={"x": torch.Size([3, 2])})
        assert estimate["ring_buffer"] == sum(int(np.prod(shape)) * dtype.itemsize for shape, dtype in specs.values())
        assert estimate["weights"] == 3*2*4 + 8
        assert estimate["total"] == estimate["ring_buffer"] + estimate["weights"]
    finally:
        ring.cleanup()
    legacy = RolloutRingBuffer(**args)
    ppo = RolloutRingBuffer(**{**args, "num_slots": 1}, protocol=FADA_PPO_PROTOCOL)
    try:
        for _ in range(5):
            legacy.signal_write_done()  # default APPO still permits overwrite
        assert legacy.available() == 2 and legacy.sync_primitives[1].value == 3
        ppo.write_buffer["valid_steps"][0] = 24
        ppo.signal_write_done()
        with pytest.raises(BufferError):
            _ = ppo.write_buffer
        ppo.advance_read()
        assert ppo.available() == 0
    finally:
        legacy.cleanup()
        ppo.cleanup()


@pytest.mark.parametrize("change", [
    {"mode": "synchronous_heterogeneous"}, {"num_slots": 1}, {"num_slots": True},
    {"collector_torch_threads": 0}, {"initial_episode_progress": False},
    {"wait_timeout_s": float("nan")}, {"poll_interval_s": 0},
    {"poll_interval_s": 20, "wait_timeout_s": 1},
])
def test_collection_config_rejects_invalid_boundaries(change):
    cfg = deepcopy(config())
    cfg["collection"].update(change)
    with pytest.raises(ValueError):
        validate_collection_contract(cfg)
