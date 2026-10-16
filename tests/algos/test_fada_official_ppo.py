"""Offline comparison with the pinned FADA formulas, without a physics env."""
import copy
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from unilab.algos.torch.fada_official_ppo import (
    LEARNER_SCHEMA, OfficialFadaGaussianDistribution, OfficialFadaPPO,
)
from unilab.training.rsl_rl import normalize_ppo_train_cfg

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def cpu_budget():
    threads = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        torch.set_num_threads(1)
        yield
    torch.set_num_threads(threads)


def composed(task="g1_official_oracle_flat/mujoco"):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "conf/ppo")):
        cfg = compose(config_name="config", overrides=[f"task={task}"])
    return normalize_ppo_train_cfg(OmegaConf.to_container(cfg.algo, resolve=True))


def make_learner(*, full=False, **overrides):
    cfg = composed()
    cfg["multi_gpu"] = None
    if not full:
        cfg["num_steps_per_env"] = 2
        cfg["actor"]["hidden_dims"] = [4, 3]
        cfg["critic"]["hidden_dims"] = [4, 3]
    cfg["algorithm"].update(overrides)
    dimension, actions = (322, 29) if full else (3, 2)
    obs = TensorDict({"actor": torch.zeros(2, dimension), "critic": torch.zeros(2, dimension)}, [2])
    obs["policy"] = obs["actor"].clone()  # Real RslRlVecEnvWrapper also supplies this alias.
    env = SimpleNamespace(num_envs=2, num_actions=actions)
    return OfficialFadaPPO.construct_algorithm(obs, env, copy.deepcopy(cfg), "cpu"), obs, cfg


def reference_mlp(model):
    # Plain nn.Sequential: no tested MLPModel or learner forward/loss helpers.
    layers = []
    for layer in model.mlp:
        if isinstance(layer, torch.nn.Linear):
            linear = torch.nn.Linear(layer.in_features, layer.out_features)
            linear.load_state_dict(copy.deepcopy(layer.state_dict()))
            layers.append(linear)
        else:
            assert isinstance(layer, torch.nn.ELU)
            layers.append(torch.nn.ELU())
    return torch.nn.Sequential(*layers)


def reference_losses(actor, critic, std_parameter, batch, clip=.2):
    # Reference: FADA agents/ppo/ppo.py:_compute_ppo_loss, commit 290b3a9c.
    mean = actor(batch.observations["actor"])
    std = torch.nan_to_num(std_parameter.clamp(min=1e-6), nan=1e-6, posinf=1., neginf=1e-6)
    logp = -.5 * (((batch.actions - mean) / std).square() + 2 * std.log() + math.log(2 * math.pi)).sum(-1)
    entropy = (std.log() + .5 * (1 + math.log(2 * math.pi))).sum()
    ratio = torch.exp(logp - batch.old_actions_log_prob[:, 0])
    advantage = batch.advantages[:, 0]
    surrogate = -torch.minimum(advantage * ratio, advantage * ratio.clamp(1 - clip, 1 + clip)).mean()
    value = critic(batch.observations["critic"])
    clipped = batch.values + torch.clip(value - batch.values, -clip, clip)
    value_loss = torch.maximum((value - batch.returns)**2, (clipped - batch.returns)**2).mean()
    old_mean, old_std = batch.old_distribution_params
    kl = (torch.log(std / old_std) + (old_std**2 + (old_mean - mean)**2) / (2 * std**2) - .5).sum(-1).mean()
    return {"actor_loss": surrogate - .01 * entropy, "critic_loss": value_loss,
            "surrogate": surrogate, "value": value_loss, "entropy": entropy, "kl": kl,
            "mean": mean, "logp": logp, "predicted_value": value}


def fixed_batch(algo, dimension, actions, *, dtype=torch.float32):
    obs = TensorDict({"actor": torch.linspace(-.5, .5, 4 * dimension, dtype=dtype).reshape(4, dimension),
                      "critic": torch.linspace(.4, -.4, 4 * dimension, dtype=dtype).reshape(4, dimension)}, [4])
    with torch.no_grad():
        mean = algo.actor(obs)
        value = algo.critic(obs)
        action = mean + torch.linspace(-.3, .3, actions, dtype=dtype)
        std = torch.full_like(mean, .8)
        logp = torch.distributions.Normal(mean, std).log_prob(action).sum(-1, keepdim=True)
    return RolloutStorage.Batch(
        observations=obs, actions=action, values=value + torch.tensor([[1.], [-1.], [2.], [-2.]], dtype=dtype),
        advantages=torch.tensor([[10.], [-20.], [30.], [-40.]], dtype=dtype),
        returns=value + torch.tensor([[100.], [-100.], [50.], [-50.]], dtype=dtype),
        old_actions_log_prob=logp + torch.tensor([[-.6], [-.6], [.3], [.3]], dtype=dtype),
        old_distribution_params=(mean + .04, std),
    )


def assert_nested_equal(one, two):
    if isinstance(one, torch.Tensor):
        torch.testing.assert_close(one, two, rtol=1e-5, atol=1e-6)
    elif isinstance(one, dict):
        assert one.keys() == two.keys()
        for key in one: assert_nested_equal(one[key], two[key])
    elif isinstance(one, (list, tuple)):
        assert len(one) == len(two)
        for a, b in zip(one, two, strict=True): assert_nested_equal(a, b)
    else:
        assert one == two


def fill_rollout(algo, obs):
    with torch.inference_mode():
        for i in range(algo.storage.num_transitions_per_env):
            algo.act(obs)
            algo.process_env_step(obs, torch.tensor([1., 2.]) + i, torch.tensor([False, False]), {})
        algo.compute_returns(obs)


def test_config_full_network_probabilities_and_one_adamw_update(monkeypatch):
    algo, _, cfg = make_learner(full=True, num_learning_epochs=1, num_mini_batches=1)
    assert cfg["num_steps_per_env"] == 24 and cfg["max_iterations"] == 25000
    assert algo.actor.obs_dim == algo.critic.obs_dim == 322
    assert [x.out_features for x in algo.actor.mlp if isinstance(x, torch.nn.Linear)] == [512, 256, 128, 29]
    assert [x.out_features for x in algo.critic.mlp if isinstance(x, torch.nn.Linear)] == [512, 256, 128, 1]
    assert not hasattr(algo, "optimizer")
    actor, critic = reference_mlp(algo.actor), reference_mlp(algo.critic)
    std = torch.nn.Parameter(algo.actor.distribution.std_param.detach().clone())
    batch = fixed_batch(algo, 322, 29)
    reference = reference_losses(actor, critic, std, batch)
    observed = algo._compute_losses(batch)
    for name in observed:
        torch.testing.assert_close(observed[name], reference[name], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(algo.actor.output_mean, reference["mean"], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(algo.actor.get_output_log_prob(batch.actions), reference["logp"], rtol=1e-5, atol=1e-6)
    actor_parameters = list(actor.parameters()) + [std]
    ref_actor_opt = torch.optim.AdamW(actor_parameters, lr=1e-5, weight_decay=.001, betas=(.9, .999), eps=1e-8)
    ref_critic_opt = torch.optim.AdamW(critic.parameters(), lr=1e-5, weight_decay=.001, betas=(.9, .999), eps=1e-8)
    (reference["actor_loss"] + reference["critic_loss"]).backward()
    actor_norm = torch.nn.utils.clip_grad_norm_(actor_parameters, 1.)
    critic_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.)
    assert actor_norm > 1 and critic_norm > 1
    before = [p.detach().clone() for p in algo.actor.parameters()]
    ref_actor_opt.step(); ref_critic_opt.step()
    algo.storage.step = algo.storage.num_transitions_per_env
    algo._returns_ready = True
    monkeypatch.setattr(algo.storage, "mini_batch_generator", lambda *args: iter([batch]))
    metrics = algo.update()
    assert algo.storage.step == 0 and not algo._returns_ready
    torch.testing.assert_close(algo.actor.distribution.std_param, std, rtol=1e-5, atol=1e-6)
    for actual, expected in ((algo.actor.mlp, actor), (algo.critic.mlp, critic)):
        assert_nested_equal(actual.state_dict(), expected.state_dict())
    # Compare parameter increments, not just nearly equal large initial weights.
    ref_parameters = [std] + list(actor.parameters())  # distribution registered before mlp
    for initial, actual, expected in zip(before, algo.actor.parameters(), ref_parameters, strict=True):
        torch.testing.assert_close(actual - initial, expected - initial, rtol=1e-5, atol=1e-6)
    for name, model, optimizer in (("actor", actor, ref_actor_opt), ("critic", critic, ref_critic_opt)):
        tested = getattr(algo, f"{name}_optimizer")
        for actual, expected in zip(getattr(algo, name).mlp.parameters(), model.parameters(), strict=True):
            assert_nested_equal(tested.state[actual], optimizer.state[expected])
    assert_nested_equal(algo.actor_optimizer.state[algo.actor.distribution.std_param], ref_actor_opt.state[std])
    assert metrics["actor_learning_rate"] == metrics["critic_learning_rate"] == 1e-5


def test_float64_losses_and_identical_distribution_kl():
    algo, _, _ = make_learner()
    algo.actor.double(); algo.critic.double()
    actor, critic = reference_mlp(algo.actor).double(), reference_mlp(algo.critic).double()
    # reference_mlp constructs float32 layers; reload after casting for float64 fidelity.
    actor.load_state_dict(algo.actor.mlp.state_dict()); critic.load_state_dict(algo.critic.mlp.state_dict())
    std = algo.actor.distribution.std_param.detach().clone()
    batch = fixed_batch(algo, 3, 2, dtype=torch.float64)
    expected = reference_losses(actor, critic, std, batch)
    observed = algo._compute_losses(batch)
    for name in observed: torch.testing.assert_close(observed[name], expected[name], rtol=1e-8, atol=1e-10)
    batch.old_distribution_params = (algo.actor.output_mean.detach().clone(), algo.actor.output_std.detach().clone())
    assert algo._compute_losses(batch)["kl"].item() == 0


def test_std_scale_protection_and_raw_actions():
    distribution = OfficialFadaGaussianDistribution(5)
    with torch.no_grad(): distribution.std_param.copy_(torch.tensor([-1., float("nan"), float("inf"), -float("inf"), .8]))
    distribution.update(torch.zeros(2, 5))
    torch.testing.assert_close(distribution.std[0], torch.tensor([1e-6, 1e-6, 1., 1e-6, .8]))
    assert torch.isfinite(distribution.log_prob(torch.zeros(2, 5))).all()
    assert torch.isfinite(distribution.entropy).all()
    algo, obs, _ = make_learner()
    with torch.no_grad():
        for parameter in algo.actor.mlp.parameters(): parameter.zero_()
        algo.actor.mlp[-1].bias.fill_(200.)
        raw = algo.act(obs)
    assert torch.all(raw > 100)  # no clipping/tanh/scaling at learner boundary
    torch.testing.assert_close(algo.transition.actions, raw)


@pytest.mark.parametrize("kl,expected", [(0., 3e-5), (.005, 3e-5), (.02, 3e-5), (.03, 2e-5), (.001, 4.5e-5)])
def test_learning_rate_thresholds(kl, expected):
    algo, _, _ = make_learner(actor_learning_rate=3e-5, critic_learning_rate=6e-5)
    algo._update_learning_rates(kl)
    assert algo.actor_learning_rate == pytest.approx(expected)
    assert algo.critic_learning_rate == pytest.approx(expected * 2)
    assert algo.actor_optimizer.param_groups[0]["lr"] == algo.learning_rate
    algo.actor_learning_rate = 1e-5
    algo._update_learning_rates(.03)
    assert algo.actor_learning_rate == 1e-5
    algo.actor_learning_rate = 1e-2
    algo._update_learning_rates(.001)
    assert algo.actor_learning_rate == 1e-2


def test_terminal_timeout_gae_uses_final_not_reset_obs(monkeypatch):
    algo, obs, _ = make_learner()
    monkeypatch.setattr(algo.critic, "forward", lambda td, **kwargs: td["critic"][:, :1])
    first, second, final, reset = (obs.clone() for _ in range(4))
    first["critic"][:, 0] = torch.tensor([1., 2.])
    second["critic"][:, 0] = torch.tensor([3., 4.])
    final["critic"][:, 0] = torch.tensor([10., 999.])
    reset["critic"][:, 0] = torch.tensor([100., 5.])
    with torch.inference_mode():
        algo.act(first)
        algo.process_env_step(second, torch.tensor([1., 2.]), torch.tensor([False, True]), {})
        algo.act(second)
        with pytest.raises(ValueError, match="pre-reset"):
            algo.process_env_step(reset, torch.tensor([3., 4.]), torch.tensor([True, False]), {"time_outs": torch.tensor([True, False])})
        assert algo.storage.step == 1
        algo.process_env_step(reset, torch.tensor([3., 4.]), torch.tensor([True, False]),
                              {"time_outs": torch.tensor([True, False]), "time_out_bootstrap_obs": final})
        algo.compute_returns(reset)
    np.testing.assert_allclose(algo.storage.rewards[:, :, 0], [[1, 2], [12.9, 4]], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(algo.storage.returns[:, :, 0], [[13.28095, 2], [12.9, 8.95]], rtol=1e-5, atol=1e-6)
    raw = np.array([[12.28095, 0.], [9.9, 4.95]])
    expected = (raw - raw.mean()) / (raw.std(ddof=1) + 1e-8)
    np.testing.assert_allclose(algo.storage.advantages[:, :, 0], expected, rtol=1e-5, atol=1e-6)


def test_8_by_4_minibatches_cover_full_rollout_and_reuse_permutation(monkeypatch):
    algo, obs, cfg = make_learner()
    assert cfg["algorithm"]["num_learning_epochs"] == 8 and cfg["algorithm"]["num_mini_batches"] == 4
    with torch.inference_mode():
        for i in range(2):
            obs["actor"][:, 0] = torch.tensor([10. + i, 20. + i])
            algo.act(obs)
            algo.process_env_step(obs, torch.ones(2), torch.zeros(2, dtype=torch.bool), {})
        algo.compute_returns(obs)
    with pytest.raises(RuntimeError, match="before collecting"):
        algo.process_env_step(obs, torch.zeros(2), torch.zeros(2, dtype=torch.bool), {})
    seen = []
    original = algo._compute_losses
    def record(batch):
        seen.append(batch.observations["actor"][:, 0].tolist())
        return original(batch)
    monkeypatch.setattr(algo, "_compute_losses", record)
    algo.update()
    assert len(seen) == 32
    assert all(seen[e*4:(e+1)*4] == seen[:4] for e in range(8))
    assert sorted(x for batch in seen[:4] for x in batch) == [10, 11, 20, 21]
    assert {int(s["step"].item()) for s in algo.actor_optimizer.state.values()} == {32}
    assert {int(s["step"].item()) for s in algo.critic_optimizer.state.values()} == {32}
    assert algo.storage.step == 0


def test_save_load_both_optimizers_and_next_update_agree():
    algo, obs, _ = make_learner(num_learning_epochs=1, num_mini_batches=1)
    fill_rollout(algo, obs); algo.update()
    saved = copy.deepcopy(algo.save())
    assert saved["learner_schema"] == LEARNER_SCHEMA and "optimizer_state_dict" not in saved
    clone, clone_obs, _ = make_learner(num_learning_epochs=1, num_mini_batches=1)
    assert clone.load(saved)
    assert_nested_equal(clone.save(), saved)
    for learner, current_obs in ((algo, obs), (clone, clone_obs)):
        torch.manual_seed(50); fill_rollout(learner, current_obs)
        torch.manual_seed(60); learner.update()
    assert_nested_equal(algo.save(), clone.save())
    actor_only, _, _ = make_learner(num_learning_epochs=1, num_mini_batches=1)
    untouched = copy.deepcopy(actor_only.critic.state_dict())
    assert not actor_only.load(saved, {"actor": True, "optimizer": False})
    assert_nested_equal(actor_only.actor.state_dict(), saved["actor_state_dict"])
    assert_nested_equal(actor_only.critic.state_dict(), untouched)
    assert not actor_only.actor_optimizer.state
    before = copy.deepcopy(clone.save())
    for key in ("learner_schema", "critic_optimizer_state_dict", "actor_learning_rate"):
        broken = copy.deepcopy(saved); broken.pop(key)
        with pytest.raises(ValueError): clone.load(broken)
        assert_nested_equal(clone.save(), before)


@pytest.mark.parametrize("kwargs", [{"enable_compile": True}, {"amp": True},
                                   {"normalize_advantage_per_mini_batch": True},
                                   {"num_mini_batches": 3}, {"use_clipped_value_loss": False}])
def test_unsupported_modes_reject_without_fallback(kwargs):
    with pytest.raises(ValueError): make_learner(**kwargs)


def test_partial_rollout_rejected_and_old_config_unchanged():
    algo, obs, _ = make_learner()
    with pytest.raises(RuntimeError, match="full rollout"): algo.compute_returns(obs)
    with pytest.raises(RuntimeError, match="full rollout"): algo.update()
    saved = copy.deepcopy(algo.save())
    with torch.inference_mode(): algo.act(obs)
    with pytest.raises(RuntimeError, match="empty rollout"): algo.load(saved)
    old = composed("g1_walk_flat/mujoco")
    assert old["algorithm"]["class_name"] == "unilab.algos.torch.rsl_rl_ppo:FinalObservationAwarePPO"
    assert "actor_learning_rate" not in old["algorithm"]
    assert old["actor"]["distribution_cfg"]["class_name"] == "rsl_rl.modules.distribution.GaussianDistribution"
