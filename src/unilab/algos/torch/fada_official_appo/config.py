"""Cold model/configuration checks; no environment or learner construction."""
from collections.abc import Mapping
import math

TASK_NAME = "G1OfficialOracleFlat"
SIM_BACKEND = "mujoco"
RUNTIME_RESOLVER = "unilab.algos.torch.fada_official_appo.runtime:resolve_runtime"
DISTRIBUTION = "unilab.algos.torch.fada_official_ppo:OfficialFadaGaussianDistribution"
MODEL_DIMENSIONS = {"obs_dim": 322, "critic_dim": 322, "action_dim": 29}


def validate_model_contract(rl_cfg, *, task_name=None, sim_backend=None, collector_device=None):
    """Check APPO model wiring, and entrypoint identity when supplied by its caller.

    The generic APPO resolver receives only ``cfg.algo``. Entrypoints can also
    supply their actual task/backend/device, without inventing duplicate task
    identity inside algorithm metadata.
    """
    for actual, expected, label in (
        (task_name, TASK_NAME, "task"),
        (sim_backend, SIM_BACKEND, "backend"),
        (collector_device, "cpu", "collector device"),
    ):
        if actual is not None and actual != expected:
            raise ValueError(f"FADA APPO requires {label}={expected}")
    if not isinstance(rl_cfg, Mapping):
        raise ValueError("FADA APPO algorithm configuration must be a mapping")
    if rl_cfg.get("algo") != "appo" or rl_cfg.get("runtime_resolver") != RUNTIME_RESOLVER:
        raise ValueError("FADA APPO requires its explicit APPO runtime, not a PPO/default runtime")
    dimensions = rl_cfg.get("model_contract")
    if (not isinstance(dimensions, Mapping) or dict(dimensions) != MODEL_DIMENSIONS
            or any(type(value) is not int for value in dimensions.values())):
        raise ValueError("FADA APPO requires teacher322/critic322/action29")
    if rl_cfg.get("obs_groups") != {"actor": {"policy": 322}, "critic": {"policy": 322}}:
        raise ValueError("FADA APPO actor and critic each require their own policy input")
    if type(rl_cfg.get("steps_per_env")) is not int or rl_cfg["steps_per_env"] != 24:
        raise ValueError("FADA APPO requires steps_per_env=24")
    if "num_steps_per_env" in rl_cfg:
        raise ValueError("FADA APPO must not inherit PPO num_steps_per_env")
    collection = rl_cfg.get("collection", {})
    algorithm = rl_cfg.get("algorithm", {})
    if not isinstance(collection, Mapping) or not isinstance(algorithm, Mapping):
        raise ValueError("FADA APPO collection/algorithm configuration must be mappings")
    ppo_only = {"class_name", "actor_learning_rate", "critic_learning_rate",
                "normalize_advantage_per_mini_batch", "rnd_cfg", "symmetry_cfg", "amp"}
    if collection.get("mode") == "synchronous_heterogeneous" or ppo_only.intersection(algorithm):
        raise ValueError("FADA APPO must not inherit synchronous PPO/dual-optimizer configuration")
    for name in ("actor", "critic"):
        model = rl_cfg.get(name)
        if not isinstance(model, Mapping):
            raise ValueError(f"FADA APPO missing {name} model")
        expected = {"class_name": "rsl_rl.models.MLPModel", "hidden_dims": [512, 256, 128],
                    "activation": "elu", "obs_normalization": False}
        if name == "actor":
            expected["distribution_cfg"] = {"class_name": DISTRIBUTION,
                                            "init_std": .8, "std_type": "scalar"}
        if dict(model) != expected:
            raise ValueError(f"FADA APPO {name} model/distribution contract mismatch")


def validate_collection_contract(rl_cfg, **identity):
    """Opt-in bounded asynchronous transport; no native environment construction."""
    validate_model_contract(rl_cfg, **identity)
    cfg = rl_cfg.get("collection", {})
    if cfg.get("mode") != "async" or cfg.get("initial_episode_progress") is not True:
        raise ValueError("FADA APPO requires async collection and initial episode progress")
    for name, lower in (("num_slots", 2), ("collector_torch_threads", 1)):
        if type(cfg.get(name)) is not int or cfg[name] < lower:
            raise ValueError(f"FADA APPO {name} must be an integer >= {lower}")
    for name in ("poll_interval_s", "wait_timeout_s"):
        value = cfg.get(name)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"FADA APPO {name} must be finite and positive")
    if cfg["poll_interval_s"] > cfg["wait_timeout_s"]:
        raise ValueError("FADA APPO poll interval exceeds wait timeout")


def validate_training_contract(rl_cfg, *, num_envs, max_iterations, save_interval=0,
                               replay_queue_size=3, resume_path=None):
    """Finite cumulative budgets and explicit sealed-checkpoint resume only."""
    validate_collection_contract(rl_cfg)
    if type(save_interval) is not int or save_interval < 0:
        raise ValueError("FADA APPO save_interval must be a nonnegative integer")
    if str(rl_cfg.get("load_run", "-1")) != "-1" or rl_cfg.get("resume", False):
        raise ValueError("FADA APPO resume requires only explicit algo.resume_path; load_run must be -1")
    path = resume_path if resume_path is not None else rl_cfg.get("resume_path")
    if path is not None and (not isinstance(path, str) or not path.strip()):
        raise ValueError("FADA APPO resume_path must be a nonempty explicit path")
    for name, value in (("num_envs", num_envs), ("max_iterations", max_iterations),
                        ("replay_queue_size", replay_queue_size)):
        if type(value) is not int or value < 1:
            raise ValueError(f"FADA APPO {name} must be a positive integer")
    lag = rl_cfg["collection"].get("max_policy_lag")
    if type(lag) is not int or lag < 0:
        raise ValueError("FADA APPO max_policy_lag must be a nonnegative integer")
    budget = rl_cfg.get("max_unique_samples")
    if budget is not None and (type(budget) is not int or budget < 1 or budget % (num_envs * 24)):
        raise ValueError("max_unique_samples must be a positive multiple of num_envs * 24")
    algorithm = rl_cfg["algorithm"]
    for name in ("num_learning_epochs", "num_mini_batches", "target_update_freq"):
        value = algorithm.get(name)
        if type(value) is not int or value < 1:
            raise ValueError(f"FADA APPO {name} must be a positive integer")
    if (num_envs * 24) % algorithm["num_mini_batches"]:
        raise ValueError("Each rollout batch must be divisible into nonempty minibatches")
    if algorithm.get("enable_compile") is not False or algorithm.get("amp", False):
        raise ValueError("FADA APPO first training slice disables compile and AMP")
    if rl_cfg.get("multi_gpu") is not None or rl_cfg.get("num_workers", 1) != 1:
        raise ValueError("FADA APPO first training slice uses one learner and one collector")
