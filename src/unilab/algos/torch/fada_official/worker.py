"""CPU collection, one complete rollout per atomically published model pair."""
from contextlib import ExitStack
from copy import deepcopy
import time

import numpy as np
import torch
from rsl_rl.utils import resolve_callable

from unilab.ipc.rollout_ring_buffer import FADA_PPO_PROTOCOL, RolloutRingBuffer
from unilab.ipc.weight_sync import SharedWeightSync
from unilab.training.rsl_rl import RslRlVecEnvWrapper
from unilab.training.seed import apply_training_seed


def _make_env(env_name, **kwargs):
    from unilab.base import registry
    from unilab.base.registry import ensure_registries
    ensure_registries()
    return registry.make(env_name, **kwargs)


def _build_models(cfg, obs, action_dim):
    models = {}
    for name, output in (("actor", action_dim), ("critic", 1)):
        spec = deepcopy(cfg[name])
        cls = resolve_callable(spec.pop("class_name"))
        models[name] = cls(obs, cfg["obs_groups"], name, output, **spec).cpu().eval()
    return torch.nn.ModuleDict(models)


def collect_round(wrapped, models, obs, output, *, num_steps, version, stop_event):
    output["valid_steps"].fill(0)
    output["policy_version"].fill(version)
    output["final_critic"].fill(0)
    actor, critic = models["actor"], models["critic"]
    with torch.inference_mode():
        for step in range(num_steps):
            if stop_event.is_set():
                return None
            actions = actor(obs, stochastic_output=True)
            mean, std = actor.output_distribution_params
            # Copy before env.step: env-owned arrays and TensorDict views can
            # be mutated by autoreset or reused at the next control step.
            for name, tensor in (("obs", obs["actor"]), ("critic", obs["critic"]),
                                 ("actions", actions), ("log_probs", actor.get_output_log_prob(actions)),
                                 ("values", critic(obs).squeeze(-1)),
                                 ("action_mean", mean), ("action_std", std)):
                output[name][:, step] = tensor.numpy()
            obs, reward, done, info = wrapped.step(actions)
            output["rewards"][:, step] = reward.numpy()
            output["dones"][:, step] = done.numpy()
            timeout = info.get("time_outs", torch.zeros_like(done)).bool()
            output["truncated"][:, step] = timeout.numpy()
            if timeout.any():
                final = info.get("time_out_bootstrap_obs")
                if final is None:
                    raise ValueError("PPO collector timeout has no pre-reset critic observation")
                output["final_critic"][timeout.numpy(), step] = final["critic"][timeout].numpy()
            output["valid_steps"][0] = step + 1
        output["last_obs"][:] = obs["actor"].numpy()
        output["last_critic"][:] = obs["critic"].numpy()
    return obs


def official_fada_collector_fn(
    *, stop_event, env_name, env_cfg_overrides, rl_cfg, num_envs, num_steps,
    obs_dim, action_dim, critic_dim, ring_names, sync_primitives,
    weight_name, weight_shapes, weight_lock, initial_version,
    poll_interval_s, round_timeout_s, collector_torch_threads=1,
    seed=None, sim_backend="mujoco", env_factory=None,
):
    torch.set_num_threads(collector_torch_threads)
    apply_training_seed(seed, torch_runtime=True, cuda=False)
    with ExitStack() as stack:
        ring = RolloutRingBuffer(num_envs, num_steps, obs_dim, action_dim, critic_dim=critic_dim,
                                 num_slots=1, protocol=FADA_PPO_PROTOCOL, create=False,
                                 shm_name_prefix=ring_names)
        stack.callback(ring.close)
        ring.attach_sync_primitives(*sync_primitives)
        weights = SharedWeightSync(weight_shapes, create=False, shm_name=weight_name, lock=weight_lock)
        stack.callback(weights.close)
        env = (env_factory or _make_env)(env_name, num_envs=num_envs, sim_backend=sim_backend,
                                        env_cfg_override=env_cfg_overrides)
        stack.callback(env.close)
        if env.obs_groups_spec != {"obs": obs_dim, "critic": critic_dim} or env.action_space.shape != (action_dim,):
            raise ValueError("PPO collector environment schema/dimensions mismatch")
        wrapped = RslRlVecEnvWrapper(env, device="cpu")
        if rl_cfg["collection"].get("initial_episode_progress", False):
            env.initialize_training_progress()
            wrapped.episode_length_buf.copy_(torch.as_tensor(env.state.info["steps"], dtype=torch.float32))
        obs = wrapped.get_observations()
        if any(value.dtype != torch.float32 for value in obs.values()):
            raise ValueError("PPO collector observations must be float32")
        models = _build_models(rl_cfg, obs, action_dim)
        state = models.state_dict()
        if state.keys() != weight_shapes.keys() or any(state[k].shape != weight_shapes[k] for k in state):
            raise ValueError("PPO collector model pair does not match published weights")
        previous = initial_version - 1
        while not stop_event.is_set():
            deadline = time.monotonic() + round_timeout_s
            # Slot acknowledgement is not permission to collect with old weights.
            while weights.version <= previous or ring.available() == ring.num_slots:
                if stop_event.wait(poll_interval_s):
                    return
                if time.monotonic() >= deadline:
                    raise TimeoutError("PPO collector timed out waiting for next weights/empty slot")
            version = weights.read_weights_into(state)  # Same lock for the whole pair.
            if version != previous + 1:
                raise ValueError("PPO collector model pair skipped a policy version")
            obs = collect_round(wrapped, models, obs, ring.write_buffer, num_steps=num_steps,
                                version=version, stop_event=stop_event)
            if obs is None:
                return  # A partial round is never committed.
            ring.signal_write_done()
            previous = version
