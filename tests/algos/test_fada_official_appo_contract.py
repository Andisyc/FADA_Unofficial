"""CPU-only model/configuration proof; no environment, worker or training step."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.utils import resolve_callable
from tensordict import TensorDict

from unilab.algos.torch.appo.learner import APPOLearner, _distribution_std
from unilab.algos.torch.appo.runner import APPORunner
from unilab.algos.torch.appo.runtime import resolve_appo_runtime
from unilab.algos.torch.fada_official_appo.config import validate_model_contract
from unilab.algos.torch.fada_official_ppo import OfficialFadaGaussianDistribution

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def cpu_budget():
    threads = torch.get_num_threads()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.set_num_threads(1)
            torch.manual_seed(17)
            yield
    finally:
        torch.set_num_threads(threads)


def composed(algo="appo", task="g1_official_oracle_flat/mujoco"):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "conf" / algo)):
        cfg = compose(config_name="config", overrides=[f"task={task}"])
    return OmegaConf.to_container(cfg, resolve=True)


def make_learner():
    cfg = composed()
    # Invoke the actual APPO model/learner constructor, but not runner.__init__:
    # dimension discovery there creates a native environment, outside this test.
    owner = SimpleNamespace(rl_cfg=cfg["algo"], num_envs=2, obs_dim=322,
                            critic_input_dim=322, action_dim=29, device="cpu", seed=1)
    learner = APPORunner._build_learner(owner)
    return learner, cfg


def assert_close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-4)


def test_config_preserves_environment_reward_and_models():
    cfg, ppo = composed(), composed("ppo")
    validate_model_contract(cfg["algo"], task_name=cfg["training"]["task_name"],
                            sim_backend=cfg["training"]["sim_backend"],
                            collector_device=cfg["training"]["collector_device"])
    for section in ("env", "reward", "play_profile"):
        assert cfg[section] == ppo[section]
    for name in ("actor", "critic"):
        assert cfg["algo"][name] == ppo["algo"][name]
    assert cfg["training"]["no_play"] is True
    assert cfg["algo"]["algo_log_name"] == "fada_official_appo"
    assert cfg["algo"]["algorithm"]["enable_compile"] is False
    assert "num_steps_per_env" not in cfg["algo"]
    assert "class_name" not in cfg["algo"]["algorithm"]
    assert cfg["algo"]["collection"]["mode"] == "async"
    assert cfg["algo"]["collection"]["num_slots"] == 4


@pytest.mark.parametrize("change", [
    {"model_contract": {"obs_dim": 66, "critic_dim": 322, "action_dim": 29}},
    {"model_contract": {"obs_dim": 322, "critic_dim": 322, "action_dim": 28}},
    {"obs_groups": {"actor": ["actor"], "critic": ["critic"]}},
    {"steps_per_env": 12},
    {"num_steps_per_env": 24},
    {"runtime_resolver": "unilab.algos.torch.fada_official.runtime:resolve_runtime"},
    {"collection": {"mode": "synchronous_heterogeneous"}},
    {"algorithm": {"actor_learning_rate": 1e-5}},
    {"actor": {"class_name": "rsl_rl.models.MLPModel"}},
])
def test_rejects_mixed_or_invalid_model_configuration(change):
    cfg = composed()["algo"]
    cfg.update(change)
    with pytest.raises(ValueError):
        validate_model_contract(cfg)


@pytest.mark.parametrize("identity", [
    {"task_name": "G1WalkFlat"}, {"sim_backend": "motrix"}, {"collector_device": "cuda:0"},
])
def test_rejects_incompatible_entrypoint_identity(identity):
    with pytest.raises(ValueError):
        validate_model_contract(composed()["algo"], **identity)


def test_selected_runtime_cannot_silently_train_or_fall_back():
    selected = resolve_appo_runtime(composed()["algo"], default_play_fn=lambda *a, **k: None)
    assert selected.runner_cls.__name__ == "OfficialFadaAPPORunner"
    assert callable(selected.train_fn)
    from unilab.algos.torch.fada_official_appo.playback import play_checkpoint
    assert selected.play_fn is play_checkpoint
    default_cfg = composed(task="go1_joystick_flat/mujoco")
    assert resolve_appo_runtime(default_cfg["algo"], default_play_fn=lambda: None).runner_cls is APPORunner
    # The old synchronous runtime remains resolvable, without building an env.
    from unilab.algos.torch.fada_official.runtime import resolve_runtime
    assert callable(resolve_runtime(composed("ppo")["algo"]).train)


def test_real_appo_models_keep_inputs_separate_and_actions_raw():
    learner, cfg = make_learner()
    actor_obs = torch.linspace(-.4, .4, 2 * 322).reshape(2, 322)
    critic_obs = -actor_obs
    actor_td = TensorDict({"policy": actor_obs}, [2])
    critic_td = TensorDict({"policy": critic_obs}, [2])
    assert isinstance(learner.actor, MLPModel)
    assert isinstance(learner.actor.distribution, OfficialFadaGaussianDistribution)
    assert learner.actor.obs_dim == learner.critic.obs_dim == 322
    with torch.no_grad():
        # Make tanh/action scaling observable, rather than testing tiny outputs.
        layers = [layer for layer in learner.actor.mlp if isinstance(layer, torch.nn.Linear)]
        layers[-1].weight.zero_()
        layers[-1].bias.fill_(2.)
        assert_close(learner.actor(actor_td), torch.full((2, 29), 2.))
        assert_close(learner._critic_value(critic_obs), learner.critic(critic_td).squeeze(-1))
        assert learner.actor(actor_td, stochastic_output=True).shape == (2, 29)
        assert_close(learner.actor.output_std, torch.full((2, 29), .8))
    # Build a collector-side model via the same resolved fields used by worker.py.
    actor_cfg = copy.deepcopy(cfg["algo"]["actor"])
    actor_cls = resolve_callable(actor_cfg.pop("class_name"))
    collector = actor_cls(actor_td, cfg["algo"]["obs_groups"], "actor", 29, **actor_cfg)
    collector.load_state_dict(learner.actor.state_dict(), strict=True)
    assert_close(collector(actor_td), learner.actor(actor_td))


@pytest.mark.parametrize("parameter,scale", [
    (.8, .8), (1e-9, 1e-6), (0., 1e-6), (-.2, 1e-6),
    (float("nan"), 1e-6), (float("inf"), 1.), (float("-inf"), 1e-6),
])
def test_collector_and_learner_use_same_probability(parameter, scale):
    learner, cfg = make_learner()
    obs = torch.linspace(-.3, .3, 2 * 322).reshape(2, 322)
    td = TensorDict({"policy": obs}, [2])
    actor_cfg = copy.deepcopy(cfg["algo"]["actor"])
    actor_cls = resolve_callable(actor_cfg.pop("class_name"))
    collector = actor_cls(td, cfg["algo"]["obs_groups"], "actor", 29, **actor_cfg)
    with torch.no_grad():
        learner.actor.distribution.std_param.fill_(parameter)
        collector.load_state_dict(learner.actor.state_dict(), strict=True)
        collector(td, stochastic_output=True)
        mean, std = learner._actor_mean_std(obs)
        assert_close(std, torch.full_like(mean, scale))
        # Fixed actions avoid RNG equality masquerading as probability equality.
        actions = mean + torch.linspace(-.2, .2, 29) * std
        reference = torch.distributions.Normal(mean, torch.full_like(mean, scale))
        reference_logp = reference.log_prob(actions).sum(-1)
        worker_logp = collector.get_output_log_prob(actions)
        learner_logp = learner._gaussian_log_prob(actions, mean, std)
        assert_close(worker_logp, reference_logp)
        assert_close(learner_logp, reference_logp)
        assert_close(learner._gaussian_entropy(std), reference.entropy().sum(-1))
        assert_close(torch.exp(learner_logp - worker_logp), torch.ones(2))


def test_finite_std_gradient_and_original_ppo_distribution_are_preserved():
    distribution = OfficialFadaGaussianDistribution(2)
    mean = torch.tensor([[.2, -.3]], requires_grad=True)
    actions = torch.tensor([[.1, .4]])
    distribution.update(mean)
    # Historical PPO parameter names and the original sampling formula survive.
    assert set(distribution.state_dict()) == {"std_param"}
    old_scale = torch.nan_to_num(distribution.std_param.clamp(min=1e-6),
                                nan=1e-6, posinf=1., neginf=1e-6)
    assert_close(distribution.std, old_scale.expand_as(mean))
    std = _distribution_std(distribution, mean)
    actual = APPOLearner._gaussian_log_prob(actions, mean, std).sum() + .01 * APPOLearner._gaussian_entropy(std).sum()
    actual_grads = torch.autograd.grad(actual, (mean, distribution.std_param))
    ref_mean = mean.detach().clone().requires_grad_()
    ref_std = torch.full((2,), .8, requires_grad=True)
    reference = torch.distributions.Normal(ref_mean, ref_std)
    expected = reference.log_prob(actions).sum() + .01 * reference.entropy().sum()
    for one, two in zip(actual_grads, torch.autograd.grad(expected, (ref_mean, ref_std)), strict=True):
        assert torch.isfinite(one).all()
        assert_close(one, two)
    assert_close(distribution.log_prob(actions),
                 torch.distributions.Normal(mean, old_scale).log_prob(actions).sum(-1))


@pytest.mark.parametrize("std_type", ["scalar", "log"])
def test_default_gaussian_keeps_original_scale_and_state(std_type):
    distribution = GaussianDistribution(3, init_std=.7, std_type=std_type)
    mean = torch.zeros(2, 3)
    distribution.update(mean)
    assert_close(_distribution_std(distribution, mean), distribution.std)
    expected_key = "std_param" if std_type == "scalar" else "log_std_param"
    assert set(distribution.state_dict()) == {expected_key}
    if std_type == "scalar":
        # FADA protection must not silently alter the legacy scalar branch.
        with torch.no_grad():
            distribution.std_param.fill_(-.2)
        assert_close(_distribution_std(distribution, mean), torch.full_like(mean, -.2))
