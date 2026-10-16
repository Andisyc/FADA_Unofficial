"""Bounded offline sync/IPC proof: scripted envs, CPU tensors, no physics."""
import copy
import multiprocessing as mp
from multiprocessing import shared_memory
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.algos.torch.fada_official import runner as owner
from unilab.algos.torch.fada_official.rollout import import_round_into_learner, validate_round_payload
from unilab.base.np_env import NpEnvState
from unilab.ipc.memory_budget import estimate_fada_ppo_bytes
from unilab.ipc.rollout_ring_buffer import FADA_PPO_PROTOCOL, RolloutRingBuffer, rollout_field_specs
from unilab.ipc.weight_sync import SharedWeightSync

ROOT = Path(__file__).resolve().parents[2]
SPAWN = mp.get_context("spawn")


class ScriptedEnv:
    """Intentionally reuses arrays to exercise producer/consumer aliasing."""
    def __init__(self, num_envs, settings):
        self.num_envs = num_envs
        self.settings = settings
        self.cfg = SimpleNamespace(max_episode_seconds=20., ctrl_dt=.02)
        self.action_space = SimpleNamespace(shape=(29,))
        self.observation_space = SimpleNamespace(shape=(644,))
        self.obs_groups_spec = {"obs": 322, "critic": 322}
        self.counter = settings.get("counter")
        self.tick = 0
        self.state = None

    def reset_all(self):
        self.state = NpEnvState({"obs": np.zeros((self.num_envs, 322), np.float32),
                                 "critic": np.zeros((self.num_envs, 322), np.float32)},
                                np.ones(self.num_envs, np.float32), np.zeros(self.num_envs, bool),
                                np.zeros(self.num_envs, bool), {})
        return self.state

    def step(self, actions):
        self.tick += 1
        if self.counter is not None:
            with self.counter.get_lock(): self.counter.value += 1
        if self.settings.get("fail_at") == self.tick:
            raise RuntimeError("scripted collector failure")
        state = self.state
        for array in state.obs.values(): array.fill(self.tick)
        state.terminated.fill(False); state.truncated.fill(False)
        state.final_observation = None
        if self.tick == 1:
            state.truncated[0] = True
            state.terminated[1] = True
            state.final_observation = {"obs": np.full((self.num_envs, 322), 10, np.float32),
                                        "critic": np.full((self.num_envs, 322), 10, np.float32)}
            for array in state.obs.values(): array[:2] = -100  # clearly not final obs
        state.reward.fill(1)
        return state

    def close(self):
        if self.settings.get("closed") is not None: self.settings["closed"].set()


def scripted_env_factory(env_name, *, num_envs, sim_backend, env_cfg_override):
    assert env_name == "G1OfficialOracleFlat" and sim_backend == "mujoco"
    return ScriptedEnv(num_envs, env_cfg_override)


@pytest.fixture(autouse=True)
def cpu_budget():
    threads = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7); torch.set_num_threads(1)
        yield
    torch.set_num_threads(threads)


def config():
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "conf/ppo")):
        cfg = OmegaConf.to_container(compose(config_name="config", overrides=["task=g1_official_oracle_flat/mujoco"]).algo, resolve=True)
    cfg["num_envs"] = 2; cfg["num_steps_per_env"] = 2
    cfg["actor"]["hidden_dims"] = [4, 3]; cfg["critic"]["hidden_dims"] = [4, 3]
    cfg["collection"]["round_timeout_s"] = 15.
    cfg["collection"]["poll_interval_s"] = .01
    return cfg


def make_runner(settings=None, cfg=None):
    return owner.OfficialFadaPPORunner("G1OfficialOracleFlat", settings or {}, cfg or config(),
                                      device="cpu", env_factory=scripted_env_factory)


def ring_args():
    return dict(num_envs=2, num_steps=2, obs_dim=3, action_dim=2, critic_dim=3,
                num_slots=1, protocol=FADA_PPO_PROTOCOL)


def valid_payload(version=1):
    specs = rollout_field_specs(1, 2, 2, 3, 2, 3, protocol=FADA_PPO_PROTOCOL)
    payload = {name: np.zeros(shape[1:], dtype) for name, (shape, dtype) in specs.items()}
    payload["policy_version"][0] = version; payload["valid_steps"][0] = 2
    payload["action_std"].fill(.8)
    return payload


def validate(payload, version=1):
    validate_round_payload(payload, expected_version=version, num_envs=2, num_steps=2,
                           obs_dim=3, action_dim=2, critic_dim=3)


def assert_unlinked(names):
    for name in names:
        with pytest.raises(FileNotFoundError): shared_memory.SharedMemory(name=name, create=False)


def shared_names(runner):
    return [name for resource in runner._shared_resources
            for name in (resource.name.values() if isinstance(resource.name, dict) else [resource.name])]


def set_model_markers(learner, value):
    with torch.no_grad():
        for parameter in learner.actor.mlp.parameters(): parameter.zero_()
        for parameter in learner.critic.parameters(): parameter.zero_()
        learner.actor.mlp[-1].bias.fill_(value)
        learner.critic.mlp[-1].bias.fill_(10 * value)


def test_two_real_spawn_rounds_freeze_model_pair_and_release_resources(monkeypatch):
    counter, closed = SPAWN.Value("i", 0), SPAWN.Event()
    runner = make_runner({"counter": counter, "closed": closed})
    runner.learner = runner._build_learner()
    set_model_markers(runner.learner, 1)
    imported, metrics, names = [], [], []
    actual_import = owner.import_round_into_learner
    def observe_import(learner, payload):
        imported.append({k: v.cpu().clone() for k, v in payload.items()})
        actual_import(learner, payload)
        torch.testing.assert_close(learner.storage.actions, payload["actions"].transpose(0, 1))
        torch.testing.assert_close(learner.storage.values[:, :, 0], payload["values"].T)
        torch.testing.assert_close(learner.storage.actions_log_prob[:, :, 0], payload["log_probs"].T)
        for actual, key in zip(learner.storage.distribution_params, ("action_mean", "action_std"), strict=True):
            torch.testing.assert_close(actual, payload[key].transpose(0, 1))
    monkeypatch.setattr(owner, "import_round_into_learner", observe_import)
    original_update = runner.learner.update
    def slow_update():
        before = counter.value
        assert before in (2, 4)
        # Deliberate bounded learner stall: ACK alone cannot release collection.
        assert not SPAWN.Event().wait(.05)
        assert counter.value == before
        return original_update()
    monkeypatch.setattr(runner.learner, "update", slow_update)
    def completed(event):
        metrics.append(dict(event))
        names.extend(shared_names(runner))
        assert counter.value == event["iteration"] * 2
        if event["iteration"] == 1: set_model_markers(runner.learner, 10)
    last = runner.learn(2, on_iteration=completed)
    assert [m["policy_version"] for m in metrics] == [1, 2]
    assert sum(m["samples"] for m in metrics) == 8
    assert all(m["raw_mean_reward"] == 1 for m in metrics)
    assert last["iteration"] == 2 and counter.value == 4
    torch.testing.assert_close(imported[0]["action_mean"], torch.ones(2, 2, 29))
    torch.testing.assert_close(imported[0]["values"], torch.full((2, 2), 10.))
    torch.testing.assert_close(imported[1]["action_mean"], torch.full((2, 2, 29), 10.))
    torch.testing.assert_close(imported[1]["values"], torch.full((2, 2), 100.))
    assert imported[0]["obs"][0, 0, 0] == 0 and imported[0]["obs"][0, 1, 0] == -100
    assert imported[0]["final_critic"][0, 0, 0] == 10
    assert imported[0]["rewards"][0, 0] == 1
    assert closed.is_set() and runner._collector_process is None and runner.collector_exitcode == 0
    assert_unlinked(names)
    runner.close()


def test_real_spawn_worker_error_is_visible_and_cleans_partial_round(monkeypatch):
    closed = SPAWN.Event()
    runner = make_runner({"fail_at": 2, "closed": closed})
    names = []
    start = runner._start_collector
    def record(*args, **kwargs):
        names.extend(shared_names(runner)); return start(*args, **kwargs)
    monkeypatch.setattr(runner, "_start_collector", record)
    with pytest.raises(RuntimeError, match="scripted collector failure"):
        runner.learn(2)
    assert runner.iteration == 0 and runner.learner.storage.step == 0
    assert closed.is_set() and runner._collector_process is None
    assert runner.collector_exitcode not in (0, None)
    assert_unlinked(names)


def test_strict_ring_no_overwrite_large_version_and_default_legacy_behavior():
    ring = RolloutRingBuffer(**ring_args())
    try:
        with pytest.raises(ValueError, match="incomplete"): ring.signal_write_done()
        payload = valid_payload(2**24 + 123)
        for name, value in payload.items(): ring.write_buffer[name][:] = value
        ring.signal_write_done()
        with pytest.raises(BufferError): _ = ring.write_buffer
        with pytest.raises(BufferError): ring.signal_write_done()
        views = ring.read_numpy_views(); validate(views, 2**24 + 123)
        owned = ring.read_torch("cpu")
        assert owned["policy_version"].dtype == torch.int64 and int(owned["policy_version"][0]) == 2**24 + 123
        ring.advance_read()
        ring.write_buffer["obs"].fill(42)
        assert owned["obs"].sum() == 0
        wp, _ = ring.sync_primitives
        wp.value += 2
        with pytest.raises(RuntimeError, match="window"): ring.available()
    finally: ring.cleanup()
    legacy = RolloutRingBuffer(2, 2, 3, 2, num_slots=1)
    try:
        assert "values" not in legacy.name and "policy_version" not in legacy.name
        legacy.signal_write_done(); legacy.signal_write_done()
        assert legacy.available() == 1 and legacy.read_slot == 0
        assert all(value.dtype == np.float32 for value in legacy.read_numpy_views().values())
    finally: legacy.cleanup()


@pytest.mark.parametrize("kind", ["stale", "future", "duplicate", "missing", "partial", "bad_mask", "nan", "std", "dtype"])
def test_payload_corruption_rejected(kind):
    payload = valid_payload()
    version = 1
    if kind == "stale": payload["policy_version"][0] = 0
    elif kind == "future": payload["policy_version"][0] = 2
    elif kind == "duplicate": version = 2
    elif kind == "missing": payload.pop("values")
    elif kind == "partial": payload["valid_steps"][0] = 1
    elif kind == "bad_mask": payload["truncated"][0, 0] = 1
    elif kind == "nan": payload["obs"][0, 0, 0] = np.nan
    elif kind == "std": payload["action_std"][0, 0, 0] = 0
    elif kind == "dtype": payload["values"] = payload["values"].astype(np.float64)
    with pytest.raises(ValueError): validate(payload, version)


def test_import_preserves_old_statistics_and_bootstraps_final_once(monkeypatch):
    cfg = config(); cfg["collection"].update(obs_dim=3, critic_dim=3, action_dim=2)
    runner = make_runner(cfg=cfg)
    try:
        learner = runner._build_learner()
        monkeypatch.setattr(learner.critic, "forward", lambda td, **kwargs: td["critic"][:, :1])
        payload = valid_payload()
        payload["values"][:] = [[1, 3], [2, 4]]
        payload["critic"][:, :, 0] = [[1, 3], [2, 4]]
        payload["rewards"][:] = [[1, 3], [2, 4]]
        payload["dones"][:] = [[0, 1], [1, 0]]
        payload["truncated"][0, 1] = 1
        payload["final_critic"][0, 1, 0] = 10
        payload["last_critic"][:, 0] = [100, 5]
        payload["log_probs"].fill(-7); payload["action_mean"].fill(9)
        validate(payload)
        owned = {name: torch.from_numpy(value.copy()) for name, value in payload.items()}
        import_round_into_learner(learner, owned)
        torch.testing.assert_close(learner.storage.rewards[:, :, 0], torch.tensor([[1., 2.], [12.9, 4.]]))
        torch.testing.assert_close(learner.storage.returns[:, :, 0], torch.tensor([[13.28095, 2.], [12.9, 8.95]]), rtol=1e-5, atol=1e-6)
        assert torch.all(learner.storage.actions_log_prob == -7)
        assert torch.all(learner.storage.distribution_params[0] == 9)
        assert owned["rewards"][0, 1] == 3  # raw transport buffer was not bootstrapped
    finally: runner.close()


def test_weight_pair_uses_one_lock_and_one_version():
    state = {"actor.bias": torch.tensor([1.]), "critic.bias": torch.tensor([10.])}
    sync = SharedWeightSync.from_state_dict(state)
    attached = SharedWeightSync({k: v.shape for k, v in state.items()}, create=False,
                                shm_name=sync.name, lock=sync.lock)
    try:
        output = {key: torch.zeros_like(value) for key, value in state.items()}
        assert attached.read_weights_into(output) == 1
        assert output["critic.bias"] == output["actor.bias"] * 10
        sync.write_weights({"actor.bias": torch.tensor([2.]), "critic.bias": torch.tensor([20.])})
        assert attached.read_weights_into(output) == 2
        assert output["critic.bias"] == output["actor.bias"] * 10
    finally: attached.close(); sync.cleanup()


def test_allocation_and_attachment_failures_leave_no_owned_blocks(monkeypatch):
    import unilab.ipc.rollout_ring_buffer as ipc
    original = shared_memory.SharedMemory
    allocated = []
    def fail_third(*args, **kwargs):
        if len(allocated) == 2: raise MemoryError("allocation sentinel")
        handle = original(*args, **kwargs); allocated.append(handle.name); return handle
    with monkeypatch.context() as patch:
        patch.setattr(ipc.shared_memory, "SharedMemory", fail_third)
        with pytest.raises(MemoryError): RolloutRingBuffer(**ring_args())
    assert_unlinked(allocated)
    allocated.clear()
    def record_allocation(*args, **kwargs):
        handle = original(*args, **kwargs); allocated.append(handle.name); return handle
    def fail_sync(*args, **kwargs): raise RuntimeError("sync allocation sentinel")
    with monkeypatch.context() as patch:
        patch.setattr(ipc.shared_memory, "SharedMemory", record_allocation)
        patch.setattr(ipc._SPAWN_CTX, "Value", fail_sync)
        with pytest.raises(RuntimeError, match="sync allocation"): RolloutRingBuffer(**ring_args())
    assert_unlinked(allocated)
    parent = RolloutRingBuffer(**ring_args())
    try:
        names = dict(parent.name); names["critic"] = names["critic"] + "_missing"
        with pytest.raises(FileNotFoundError): RolloutRingBuffer(**ring_args(), create=False, shm_name_prefix=names)
        handle = original(name=parent.name["obs"], create=False); handle.close()  # child did not unlink parent
    finally: parent.cleanup()
    assert estimate_fada_ppo_bytes(num_envs=2, steps_per_env=2, obs_dim=3, action_dim=2, critic_dim=3)["total"] == 384


@pytest.mark.parametrize("failure", ["early_exit", "timeout", "learner", "callback", "startup"])
def test_parent_failures_stop_without_advancing_and_cleanup(monkeypatch, failure):
    runner = make_runner()
    names = []
    def no_process(*args, **kwargs):
        names.extend(shared_names(runner))
        if failure == "startup": raise RuntimeError("startup sentinel")
        if failure == "early_exit":
            runner._collector_process = SimpleNamespace(pid=1, exitcode=0, is_alive=lambda: False,
                join=lambda **kwargs: None, close=lambda: None)
    monkeypatch.setattr(runner, "_start_collector", no_process)
    if failure in ("learner", "callback"):
        def ready(ring):
            arrays = ring.write_buffer
            for value in arrays.values(): value.fill(0)
            arrays["action_std"].fill(.8); arrays["rewards"].fill(1)
            arrays["valid_steps"][0] = 2; arrays["policy_version"][0] = 1
            ring.signal_write_done()
        monkeypatch.setattr(runner, "_wait_for_round", ready)
        if failure == "learner":
            runner.learner = runner._build_learner()
            def fail_update(): raise RuntimeError("learner sentinel")
            monkeypatch.setattr(runner.learner, "update", fail_update)
    elif failure == "timeout":
        runner.round_timeout_s = .01; runner.poll_interval_s = .001
    def fail_callback(event): raise RuntimeError("callback sentinel")
    with pytest.raises((RuntimeError, TimeoutError)):
        runner.learn(2, on_iteration=fail_callback if failure == "callback" else None)
    assert runner.iteration == (1 if failure == "callback" else 0)
    assert runner._stop_event.is_set() and runner._closed
    assert_unlinked(names)
    runner.close()
