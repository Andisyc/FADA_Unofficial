"""APPO Rollout Worker — runs in a subprocess.

Collects rollout payloads and writes them to RolloutRingBuffer.
"""

from __future__ import annotations

import statistics
import sys
import time
from collections import defaultdict, deque
from contextlib import ExitStack
from functools import wraps
from queue import Empty, Full
from typing import Any, Dict

import numpy as np
import torch
from rsl_rl.utils import resolve_callable

from unilab.base.final_observation import resolve_terminal_observation_contract
from unilab.base.observations import split_obs_dict
from unilab.base.registry import ensure_registries
from unilab.training.seed import apply_training_seed


def put_latest_metrics(metrics_queue: Any, msg: dict[str, Any], *, worker_name: str) -> None:
    """Best-effort metrics enqueue that keeps recent data under learner stalls."""
    try:
        metrics_queue.put_nowait(msg)
        return
    except Full:
        pass
    except Exception as e:
        print(f"[{worker_name}] metrics enqueue error: {type(e).__name__}: {e}", file=sys.stderr)
        return

    try:
        metrics_queue.get_nowait()
    except Empty:
        pass
    except Exception as e:
        print(
            f"[{worker_name}] metrics drop stale metrics error: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return

    try:
        metrics_queue.put_nowait(msg)
    except Full:
        pass
    except Exception as e:
        print(f"[{worker_name}] metrics enqueue error: {type(e).__name__}: {e}", file=sys.stderr)


def compute_timeout_bootstrap_correction(
    critic: Any,
    collector_device: str,
    gamma: float,
    timeout_mask: np.ndarray,
    final_obs: np.ndarray,
    final_critic: np.ndarray,
) -> np.ndarray:
    """Compute gamma * V(final_observation) for current timeout envs."""
    corrections = np.zeros(timeout_mask.shape, dtype=np.float32)
    if not np.any(timeout_mask):
        return corrections

    from tensordict import TensorDict

    critic_input_np = final_critic
    critic_input = torch.from_numpy(critic_input_np[timeout_mask]).to(collector_device)
    critic_td = TensorDict(
        {"policy": critic_input},
        batch_size=critic_input.shape[0],
        device=collector_device,
    )
    with torch.no_grad():
        bootstrap = critic(critic_td).squeeze(-1).cpu().numpy().astype(np.float32, copy=False)
    corrections[timeout_mask] = float(gamma) * bootstrap
    return corrections


def _with_collector_resources(fn):
    """Close attached mappings/environment even when construction or collection fails."""
    @wraps(fn)
    def run(*args, **kwargs):
        stop_event = args[0] if args else kwargs["stop_event"]
        try:
            with ExitStack() as resources:
                return fn(*args, **kwargs, _resources=resources)
        except BaseException:
            stop_event.set()
            raise
    return run


@_with_collector_resources
def appo_collector_fn(
    stop_event: Any,
    env_name: str,
    rl_cfg: dict,
    num_envs: int,
    steps_per_env: int,
    shm_rollout_ring_buffer_name: Dict[str, str],
    sync_primitives: tuple,
    obs_dim: int,
    action_dim: int,
    critic_dim: int,
    actor_weight_sync_name: str,
    actor_weight_param_shapes: dict,
    critic_weight_sync_name: str,
    critic_weight_param_shapes: dict,
    metrics_queue: Any,
    collector_device: str = "cpu",
    sim_backend: str = "mujoco",
    env_cfg_override: dict | None = None,
    seed: int | None = None,
    nan_guard_cfg=None,
    protocol="appo",
    num_slots=4,
    joint_weight_sync=None,
    env_factory=None,
    generated_samples=None,
    _resources=None,
):
    """Entry point for the APPO collector subprocess.

    Creates environment + policy, collects rollouts, writes raw payloads
    to the IPC ring buffer. Error handling is provided by the
    ``_collector_entry_wrapper`` in ``async_runner.py``.
    """
    from copy import deepcopy

    from tensordict import TensorDict

    from unilab.base import registry
    from unilab.ipc import RolloutRingBuffer, SharedWeightSync
    from unilab.ipc.rollout_ring_buffer import FADA_APPO_PROTOCOL

    bounded = protocol == FADA_APPO_PROTOCOL
    if protocol not in ("appo", FADA_APPO_PROTOCOL):
        raise ValueError("Unsupported APPO collector protocol")
    collection = rl_cfg.get("collection", {})
    if bounded:
        torch.set_num_threads(collection["collector_torch_threads"])
        if not joint_weight_sync or joint_weight_sync.get("lock") is None:
            raise ValueError("Bounded APPO requires an atomic joint weight snapshot")
    if env_factory is None:
        ensure_registries()
    apply_training_seed(seed, torch_runtime=True, cuda=True)

    # Connect to shared memory
    ring_buffer = RolloutRingBuffer(
        num_envs=num_envs,
        num_steps=steps_per_env,
        obs_dim=obs_dim,
        action_dim=action_dim,
        critic_dim=critic_dim,
        create=False,
        shm_name_prefix=shm_rollout_ring_buffer_name,
        protocol=protocol,
        num_slots=num_slots,
    )
    _resources.callback(ring_buffer.close)
    ring_buffer.attach_sync_primitives(*sync_primitives)  # (write_ptr, read_ptr)
    if bounded:
        pair_sync = SharedWeightSync(joint_weight_sync["shapes"], create=False,
                                    shm_name=joint_weight_sync["name"], lock=joint_weight_sync["lock"])
        _resources.callback(pair_sync.close)
    else:
        actor_weight_sync = SharedWeightSync(
            actor_weight_param_shapes, create=False, shm_name=actor_weight_sync_name
        )
        _resources.callback(actor_weight_sync.close)
        critic_weight_sync = SharedWeightSync(
            critic_weight_param_shapes, create=False, shm_name=critic_weight_sync_name
        )
        _resources.callback(critic_weight_sync.close)

    # Create environment
    env: Any = (env_factory or registry.make)(
        env_name, num_envs=num_envs, sim_backend=sim_backend, env_cfg_override=env_cfg_override
    )
    _resources.callback(env.close)

    if nan_guard_cfg is not None and nan_guard_cfg.enabled:
        from unilab.base.nan_guard import NanGuard

        env.set_nan_guard(
            NanGuard(
                nan_guard_cfg,
                num_envs=env.num_envs,
                supports_state_playback=env.play_capabilities.supports_physics_state_playback,
            )
        )

    # Build actor (stochastic MLPModel — mirrors runner._build_learner)
    cfg = dict(rl_cfg)

    obs_example = torch.zeros((num_envs, obs_dim), device=collector_device)
    td_example = TensorDict({"policy": obs_example}, batch_size=num_envs)

    # deepcopy so MLPModel.__init__'s distribution_cfg.pop("class_name") doesn't
    # mutate the shared rl_cfg dict.
    actor_cfg = deepcopy(cfg["actor"])
    actor_cls = resolve_callable(actor_cfg.pop("class_name"))
    actor_cfg.pop("num_actions", None)
    actor = actor_cls(
        td_example,
        cfg.get("obs_groups", {"actor": {"policy": obs_dim}}),
        "actor",
        action_dim,
        **actor_cfg,
    )
    actor = actor.to(collector_device)
    actor.eval()

    critic_obs_dim = critic_dim if critic_dim > 0 else obs_dim
    critic_obs_example = torch.zeros((num_envs, critic_obs_dim), device=collector_device)
    critic_td_example = TensorDict({"policy": critic_obs_example}, batch_size=num_envs)
    critic_cfg = deepcopy(cfg.get("critic") or cfg.get("actor") or {})
    critic_cls = resolve_callable(critic_cfg.pop("class_name", "rsl_rl.models.MLPModel"))
    critic_cfg.pop("num_actions", None)
    critic_cfg.pop("distribution_cfg", None)
    critic = critic_cls(
        critic_td_example,
        cfg.get("obs_groups", {"critic": {"policy": critic_obs_dim}}),
        "critic",
        1,
        **critic_cfg,
    )
    critic = critic.to(collector_device)
    critic.eval()
    # Load initial weights
    if bounded:
        pair = torch.nn.ModuleDict({"actor": actor, "critic": critic})
        pair_sd = pair.state_dict()
        local_pair_version = pair_sync.read_weights_into(pair_sd)
        if local_pair_version < 1:
            raise ValueError("Bounded APPO has no published initial weights")
    else:
        actor_sd = dict(actor.state_dict())
        actor_weight_sync.read_weights_into(actor_sd)
        actor.load_state_dict(actor_sd)
        local_actor_weight_version = actor_weight_sync.version

        critic_sd = dict(critic.state_dict())
        critic_weight_sync.read_weights_into(critic_sd)
        critic.load_state_dict(critic_sd)
        local_critic_weight_version = critic_weight_sync.version

    # Reset environment
    if bounded:
        # Partial reset returns observations but does not create NpEnvState.
        # Match the official warm-reset lifecycle before scattering its clock.
        env.reset_all()
        env.initialize_training_progress()
        obs_out = env.state.obs  # initialization refreshes the phase observations
    else:
        env_indices = np.arange(num_envs, dtype=np.int32)
        try:
            obs_out, _ = env.reset(env_indices)
        except TypeError:
            obs_out, _ = env.reset()

    def to_float32_np(x):
        if hasattr(x, "cpu"):
            x = x.cpu().numpy()
        return np.asarray(x, dtype=np.float32)

    obs_np, critic_np = split_obs_dict(obs_out)
    obs_np = to_float32_np(obs_np)
    critic_np = to_float32_np(critic_np)

    # Pre-allocate obs TensorDict once; update in-place each step to avoid
    # repeated TensorDict construction overhead in the hot loop.
    obs_torch = torch.zeros((num_envs, obs_dim), dtype=torch.float32, device=collector_device)
    obs_td = TensorDict({"policy": obs_torch}, batch_size=num_envs, device=collector_device)

    total_steps = 0
    ep_rewards = deque(maxlen=100) if bounded else []
    ep_lengths = deque(maxlen=100) if bounded else []
    current_ep_rewards = np.zeros(num_envs, dtype=np.float32)
    current_ep_lengths = np.zeros(num_envs, dtype=np.int32)
    ep_reward_components = defaultdict(lambda: deque(maxlen=100)) if bounded else defaultdict(list)
    rollout_id = 0

    # Episode completion mode counters (reset after each metrics report)
    ep_timeouts = 0
    ep_terminates = 0

    # Collector timing EMA (milliseconds, α=0.1 → slow-moving average)
    _EMA = 0.1
    ema_mlp_infer_ms: float = 0.0
    ema_env_step_ms: float = 0.0

    try:
        while not stop_event.is_set():
            if bounded:
                while ring_buffer.available() == num_slots:
                    if stop_event.wait(collection["poll_interval_s"]):
                        return
                if stop_event.is_set():
                    return
            # Pull latest weights from learner
            if bounded:
                if pair_sync.version > local_pair_version:
                    local_pair_version = pair_sync.read_weights_into(pair_sd)
            elif actor_weight_sync.version > local_actor_weight_version:
                actor_sd = dict(actor.state_dict())
                local_actor_weight_version = actor_weight_sync.read_weights_into(actor_sd)
                actor.load_state_dict(actor_sd)
            if not bounded and critic_weight_sync.version > local_critic_weight_version:
                critic_sd = dict(critic.state_dict())
                local_critic_weight_version = critic_weight_sync.read_weights_into(critic_sd)
                critic.load_state_dict(critic_sd)

            # Collect one rollout of length steps_per_env
            write_buf = ring_buffer.write_buffer
            if bounded:
                write_buf["valid_steps"][0] = 0
                write_buf["policy_version"][0] = local_pair_version
                write_buf["rollout_id"][0] = rollout_id + 1
            for step in range(steps_per_env):
                if bounded and stop_event.is_set():
                    return  # Never publish a partial rollout.
                # --- MLP inference (timed) ---
                t_mlp = time.perf_counter()
                with torch.no_grad():
                    obs_torch.copy_(torch.from_numpy(obs_np))
                    actions_torch = actor(obs_td, stochastic_output=True)
                    log_probs_torch = actor.get_output_log_prob(actions_torch)
                    actions_np = actions_torch.cpu().numpy()
                ema_mlp_infer_ms = (1 - _EMA) * ema_mlp_infer_ms + _EMA * (
                    (time.perf_counter() - t_mlp) * 1000
                )

                write_buf["obs"][:, step, :] = obs_np
                if critic_np is not None:
                    write_buf["critic"][:, step, :] = critic_np
                write_buf["actions"][:, step, :] = actions_np
                write_buf["log_probs"][:, step] = log_probs_torch.cpu().numpy().ravel()

                # --- Env step (timed) ---
                t_env = time.perf_counter()
                state = env.step(actions_np)
                if bounded and generated_samples is not None:
                    with generated_samples.get_lock():
                        generated_samples.value += num_envs
                ema_env_step_ms = (1 - _EMA) * ema_env_step_ms + _EMA * (
                    (time.perf_counter() - t_env) * 1000
                )

                next_obs_raw = state.obs
                reward_raw = np.asarray(state.reward, dtype=np.float32).ravel()
                if bounded:
                    reward_raw = reward_raw.copy()
                truncated_raw = state.truncated.astype(np.float32, copy=False).ravel()
                combined_done_raw = (
                    (state.terminated | state.truncated).astype(np.float32, copy=False).ravel()
                )

                next_actor_obs_np, next_critic_np = split_obs_dict(next_obs_raw)
                next_actor_obs_np = to_float32_np(next_actor_obs_np)
                next_critic_np = to_float32_np(next_critic_np)
                terminal_contract = resolve_terminal_observation_contract(
                    next_obs_batch_size=next_actor_obs_np.shape[0],
                    final_observation=state.final_observation,
                    done=combined_done_raw > 0.5,
                    info=state.info,
                    truncated=truncated_raw,
                )

                if bounded and np.any(truncated_raw > .5):
                    if (terminal_contract.terminal_obs is None or terminal_contract.terminal_critic is None
                            or not np.array_equal(terminal_contract.timeout_terminal_mask, truncated_raw > .5)):
                        raise ValueError("Bounded APPO timeout requires pre-reset final obs and critic")
                correction = compute_timeout_bootstrap_correction(
                    critic=critic,
                    collector_device=collector_device,
                    gamma=float(cfg["algorithm"].get("gamma", 0.99)),
                    timeout_mask=terminal_contract.timeout_terminal_mask,
                    final_obs=(
                        terminal_contract.terminal_obs
                        if terminal_contract.terminal_obs is not None
                        else next_actor_obs_np
                    ),
                    final_critic=(
                        terminal_contract.terminal_critic
                        if terminal_contract.terminal_critic is not None
                        else next_critic_np
                    ),
                )
                if bounded:
                    reward_train = reward_raw + correction
                else:
                    reward_raw += correction  # Preserve the legacy APPO reward/statistics contract.
                    reward_train = reward_raw

                write_buf["rewards"][:, step] = reward_train
                write_buf["dones"][:, step] = combined_done_raw
                write_buf["truncated"][:, step] = truncated_raw

                # Episode tracking (vectorized)
                total_steps += num_envs
                current_ep_rewards += reward_raw
                current_ep_lengths += 1
                reset_indices = np.where(combined_done_raw > 0.5)[0]
                if len(reset_indices) > 0:
                    ep_rewards.extend(current_ep_rewards[reset_indices].tolist())
                    ep_lengths.extend(current_ep_lengths[reset_indices].tolist())
                    current_ep_rewards[reset_indices] = 0.0
                    current_ep_lengths[reset_indices] = 0
                    # Count episode completion modes for timeout/terminated rates
                    ep_timeouts += int(np.sum(truncated_raw[reset_indices] > 0.5))
                    ep_terminates += int(np.sum(truncated_raw[reset_indices] <= 0.5))

                log_info = state.info.get("log", {})
                for k, v in log_info.items():
                    if k.startswith("reward/"):
                        ep_reward_components[k].append(v)

                if metrics_queue is not None and total_steps % (num_envs * 10) == 0 and ep_rewards:
                    try:
                        msg = {
                            "total_steps": total_steps,
                            "mean_ep_reward": statistics.mean(ep_rewards if bounded else ep_rewards[-100:]),
                            "mean_ep_length": statistics.mean(ep_lengths if bounded else ep_lengths[-100:])
                            if ep_lengths
                            else 0.0,
                        }
                        if bounded:
                            msg["episode_window_count"] = len(ep_rewards)
                            msg["reward_statistic"] = "raw_episode_return_last100"
                        # Episode completion mode rates
                        total_ep = ep_timeouts + ep_terminates
                        if total_ep > 0:
                            msg["timeout_rate"] = ep_timeouts / total_ep
                            msg["terminated_rate"] = ep_terminates / total_ep
                            ep_timeouts = 0
                            ep_terminates = 0
                        # Collector-side timing breakdown
                        msg["collector_timing_ms"] = {
                            "mlp_infer_ms": ema_mlp_infer_ms,
                            "env_step_total_ms": ema_env_step_ms,
                        }
                        if ep_reward_components:
                            msg["reward_components"] = {
                                k: statistics.mean(v) for k, v in ep_reward_components.items() if v
                            }
                            ep_reward_components.clear()
                        put_latest_metrics(metrics_queue, msg, worker_name="APPOWorker")
                    except Exception as e:
                        print(
                            f"[APPOWorker] metrics build error: {type(e).__name__}: {e}",
                            file=sys.stderr,
                        )

                obs_np = next_actor_obs_np
                critic_np = next_critic_np
                if bounded:
                    write_buf["valid_steps"][0] = step + 1

            write_buf["last_obs"][:] = obs_np
            if critic_np is not None:
                write_buf["last_critic"][:] = critic_np
            if bounded and stop_event.is_set():
                return
            ring_buffer.signal_write_done()  # atomic increment, non-blocking
            rollout_id += 1

    except Exception:
        stop_event.set()
        raise

