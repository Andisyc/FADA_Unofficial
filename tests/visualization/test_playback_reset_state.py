"""Reset-to-consumer regression using the real NpEnv lifecycle, without physics."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np
import pytest
import torch

from unilab.algos.torch.distill.fada.model import FADAArchitectureConfig
from unilab.algos.torch.distill.fada.playback import FADAPlaybackController
from unilab.algos.torch.hora.rsl_rl import HoraRslRlVecEnvWrapper
from unilab.base.base import EnvCfg
from unilab.base.np_env import NpEnv
from unilab.training.rsl_rl import RslRlVecEnvWrapper
from unilab.visualization import playback_controls
from unilab.visualization.playback_sessions import FADAPlaybackSession, OffPolicyPlaybackSession


class _ResetEnv(NpEnv):
    def __init__(self):
        backend = MagicMock()
        backend.backend_type = "mujoco"
        super().__init__(EnvCfg(max_episode_seconds=1.0), backend, num_envs=1)
        self.reset_count = 0
        self.physical_phase = np.zeros((1, 2), dtype=np.float32)

    @property
    def obs_groups_spec(self):
        return {"obs": 7, "critic": 9}

    @property
    def action_space(self):
        return gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def _observations(self, info):
        actor = np.concatenate(
            [self.physical_phase, info["commands"], info["current_actions"]], axis=1
        )
        return {"obs": actor, "critic": np.concatenate([actor, info["critic_info"]], axis=1)}

    def reset(self, env_indices):
        assert env_indices.tolist() == [0]
        self.reset_count += 1
        self.physical_phase[:] = [self.reset_count, self.reset_count + 0.5]
        info = {
            "gait_phase": self.physical_phase.copy(),
            "commands": np.zeros((1, 3), dtype=np.float32),
            "current_actions": np.zeros((1, 2), dtype=np.float32),
            "critic_info": np.full((1, 2), self.reset_count, dtype=np.float32),
        }
        return self._observations(info), info

    def apply_action(self, actions, state):
        raise AssertionError("reset regression must not advance physics")

    def update_state(self, state):
        return state.replace(obs=self._observations(state.info))


class _Actor:
    def explore(self, obs, priv_info, deterministic):
        assert deterministic
        return obs[:, :2] + obs[:, 2:].sum(dim=1, keepdim=True) + priv_info


class _HistoryPolicy:
    config = FADAArchitectureConfig(obs_dim=7, action_dim=2, command_dim=3, history_length=3)

    def __call__(self, observation_history, action_history, command):
        return SimpleNamespace(
            action=observation_history[:, -1, :2]
            + action_history.sum(dim=1)
            + command.sum(dim=1, keepdim=True)
        )


@pytest.mark.parametrize("route", ["sac", "rsl", "hora", "fada"])
def test_reset_keeps_state_and_refreshed_policy_input_consistent(route):
    env = _ResetEnv()
    if route == "sac":
        session = OffPolicyPlaybackSession(
            env=env, device="cpu", action_mode="policy", actor=_Actor(),
            actor_algo_type="privileged_locomotion_sac", normalizer=None, num_envs=1,
            obs_extractor=lambda obs: obs["obs"],
            priv_info_resolver=lambda **kw: kw["info"]["critic_info"],
        )
        wrapper = None
    else:
        cls = RslRlVecEnvWrapper if route == "rsl" else HoraRslRlVecEnvWrapper
        wrapper = cls(env, device="cpu", policy_obs_mode="actor")
        session = None
        if route == "fada":
            session = FADAPlaybackSession(
                env=env, wrapped_env=wrapper, device="cpu", action_mode="policy", num_envs=1,
                controller=FADAPlaybackController(_HistoryPolicy(), device="cpu"),
            )

    for warm in (False, True):
        if warm:
            # Emulate a previous episode's carriers without taking a physics step.
            env.state.info["gait_phase"][:] = -5.0
            env.state.info["commands"][:] = [0.6, 0.2, 0.1]
            env.state.info["current_actions"][:] = 0.123
            env.state.info["steps"][:] = 19
            if wrapper is not None:
                wrapper.episode_returns[:] = 10.0
                wrapper.episode_lengths[:] = 19

        if session is not None:
            output = session.reset()
        else:
            output, info = wrapper.reset()
            assert info is env.state.info
        actor_obs = output if route == "sac" else output["actor"].cpu().numpy()
        np.testing.assert_array_equal(actor_obs, env.state.obs["obs"])
        np.testing.assert_array_equal(env.state.info["gait_phase"], env.physical_phase)
        np.testing.assert_array_equal(actor_obs[:, :2], env.physical_phase)
        np.testing.assert_array_equal(env.state.info["commands"], 0.0)
        np.testing.assert_array_equal(env.state.info["current_actions"], 0.0)
        np.testing.assert_array_equal(env.state.info["steps"], 0)
        if wrapper is not None:
            assert not wrapper.episode_returns.any()
            assert not wrapper.episode_lengths.any()
        if route in {"hora", "fada"}:
            np.testing.assert_array_equal(output["priv_info"].cpu(), env.state.info["critic_info"])

        if session is None:
            torch.testing.assert_close(output["actor"], wrapper.get_observations()["actor"])
        else:
            if route == "fada":
                assert session.controller._observation_history is None
                assert session.controller._action_history is None
            before = session._build_actions()
            session.refresh_observation()
            if route == "fada":
                # Compare identical cold histories to isolate observation refresh.
                session.controller.reset()
            after = session._build_actions()
            np.testing.assert_array_equal(before, after)
        env._backend.step.assert_not_called()


@pytest.mark.parametrize("route", ["sac", "fada"])
@pytest.mark.parametrize("probe_raises", [False, True])
def test_command_probe_keeps_current_phase_through_first_policy_action(
    route, probe_raises, monkeypatch
):
    env = _ResetEnv()
    original_limits = [[-0.6, -0.4, -0.8], [1.0, 0.4, 0.8]]
    env.cfg.commands = SimpleNamespace(
        vel_limit=original_limits, rel_standing_envs=1.0
    )
    if route == "sac":
        session = OffPolicyPlaybackSession(
            env=env, device="cpu", action_mode="policy", actor=_Actor(),
            actor_algo_type="privileged_locomotion_sac", normalizer=None, num_envs=1,
            obs_extractor=lambda obs: obs["obs"],
            priv_info_resolver=lambda **kw: kw["info"]["critic_info"],
        )
    else:
        wrapper = RslRlVecEnvWrapper(env, device="cpu", policy_obs_mode="actor")
        session = FADAPlaybackSession(
            env=env, wrapped_env=wrapper, device="cpu", action_mode="policy", num_envs=1,
            controller=FADAPlaybackController(_HistoryPolicy(), device="cpu"),
        )
    session.reset()
    old_phase = env.physical_phase.copy()
    if probe_raises:
        force_probe = playback_controls._force_policy_command_probe_obs

        def fail_after_refresh(owner, command):
            force_probe(owner, command)
            raise RuntimeError("injected command probe failure")

        monkeypatch.setattr(
            playback_controls, "_force_policy_command_probe_obs", fail_after_refresh
        )
        with pytest.raises(RuntimeError, match="injected command probe failure"):
            playback_controls._policy_obs_contains_command(env, reset_fn=session.reset)
    else:
        assert playback_controls._policy_obs_contains_command(env, reset_fn=session.reset)

    # Match the viewer: refresh the session, then let it consume its first action.
    session.refresh_observation()
    obs = session.obs if route == "sac" else session.obs["actor"].cpu().numpy()
    assert not np.array_equal(old_phase, env.physical_phase)
    np.testing.assert_array_equal(obs[:, :2], env.physical_phase)
    np.testing.assert_array_equal(obs[:, :2], env.state.info["gait_phase"])
    np.testing.assert_array_equal(env.state.obs["critic"][:, :7], obs)
    np.testing.assert_array_equal(env.state.info["commands"], 0.0)
    assert env.cfg.commands.vel_limit == original_limits
    assert env.cfg.commands.rel_standing_envs == 1.0

    expected = env.physical_phase.copy()
    if route == "sac":
        np.testing.assert_array_equal(session.current_priv_info, env.state.info["critic_info"])
        expected += env.state.info["critic_info"]
    else:
        assert session.controller._observation_history is None
        assert session.controller._action_history is None
    np.testing.assert_array_equal(session._build_actions(), expected)
    env._backend.step.assert_not_called()
