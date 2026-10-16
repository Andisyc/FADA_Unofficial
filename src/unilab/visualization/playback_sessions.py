"""Playback contracts, commanders, and stateful session implementations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol

import numpy as np
import torch


def _external_velocity_command_rows(
    command: np.ndarray,
    owner_rows: np.ndarray,
) -> np.ndarray:
    """Validate and broadcast one external velocity command to owner row shape."""

    if owner_rows.ndim != 2 or owner_rows.shape[1] < 3:
        raise RuntimeError(
            "Playback command synchronization requires env.state.info['commands'] "
            "with shape (num_envs, >=3)."
        )
    command_arr = np.asarray(command, dtype=owner_rows.dtype)
    if command_arr.shape == (3,):
        command_arr = np.broadcast_to(command_arr, (owner_rows.shape[0], 3))
    if command_arr.shape != (owner_rows.shape[0], 3):
        raise ValueError(
            "Playback command synchronization expects command shape "
            f"(3,) or ({owner_rows.shape[0]}, 3), got {command_arr.shape}."
        )
    if not np.all(np.isfinite(command_arr)):
        raise ValueError("Playback command synchronization requires finite command values.")
    return command_arr


@dataclass(frozen=True)
class RslRlPlaybackConfig:
    """Configuration needed to bootstrap an RSL-RL interactive playback session."""

    task: str
    load_run: str
    checkpoint: str | None
    action_mode: str
    policy_obs_mode: str
    algo_log_name: str
    log_root: str | None
    num_envs: int = 1
    speed: float = 1.0
    start_paused: bool = False
    checkpoint_path: str | None = None
    keyboard: bool = False


@dataclass
class PlaybackControls:
    """Viewer-independent playback control state."""

    paused: bool = False
    speed: float = 1.0
    _single_step_requests: int = field(default=0, init=False, repr=False)

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def toggle_pause(self) -> bool:
        self.paused = not self.paused
        return self.paused

    def request_single_step(self, count: int = 1) -> None:
        self._single_step_requests += max(int(count), 0)

    def set_speed(self, value: float) -> None:
        self.speed = max(float(value), 1e-6)

    def consume_step_permission(self) -> bool:
        if self.paused:
            if self._single_step_requests <= 0:
                return False
            self._single_step_requests -= 1
            return True
        if self._single_step_requests > 0:
            self._single_step_requests -= 1
        return True

    def target_dt(self, ctrl_dt: float) -> float:
        return float(ctrl_dt) / max(float(self.speed), 1e-6)


@dataclass
class KeyboardCommander:
    """Mutable ``[vx, vy, vyaw]`` velocity command driven by keyboard nudges.

    Per-axis nudges stack and are clamped to the task's ``commands.vel_limit``.
    """

    low: np.ndarray
    high: np.ndarray
    step_lin: float = 0.1
    step_ang: float = 0.2
    command: np.ndarray = field(init=False)

    AXIS_VX: ClassVar[int] = 0
    AXIS_VY: ClassVar[int] = 1
    AXIS_VYAW: ClassVar[int] = 2

    def __post_init__(self) -> None:
        self.low = np.asarray(self.low, dtype=np.float64).reshape(3)
        self.high = np.asarray(self.high, dtype=np.float64).reshape(3)
        self.command = np.zeros(3, dtype=np.float64)

    @classmethod
    def from_vel_limit(
        cls, vel_limit: Any, *, step_lin: float = 0.1, step_ang: float = 0.2
    ) -> "KeyboardCommander":
        limit = np.asarray(vel_limit, dtype=np.float64)
        if limit.shape != (2, 3):
            raise ValueError(f"commands.vel_limit must have shape (2, 3), got {limit.shape}")
        return cls(low=limit[0], high=limit[1], step_lin=float(step_lin), step_ang=float(step_ang))

    def nudge(self, axis: int, sign: float) -> None:
        base = self.step_lin if axis in (self.AXIS_VX, self.AXIS_VY) else self.step_ang
        delta = base * (1.0 if sign >= 0 else -1.0)
        self.command[axis] = float(
            np.clip(self.command[axis] + delta, self.low[axis], self.high[axis])
        )

    def zero(self) -> None:
        self.command[:] = 0.0

    def describe(self) -> str:
        return (
            f"cmd vx={self.command[0]:+.2f} vy={self.command[1]:+.2f} vyaw={self.command[2]:+.2f}"
        )


@dataclass
class HeightCommander:
    """Mutable target-height command driven by keyboard nudges."""

    low: float
    high: float
    step: float = 0.01
    target: float = field(init=False)

    def __post_init__(self) -> None:
        self.low = float(self.low)
        self.high = float(self.high)
        self.step = float(self.step)
        if not np.isfinite([self.low, self.high, self.step]).all():
            raise ValueError("height command bounds and step must be finite")
        if self.low > self.high:
            raise ValueError(f"height command range must be ordered, got [{self.low}, {self.high}]")
        if self.step <= 0.0:
            raise ValueError(f"height command step must be positive, got {self.step}")
        self.target = self.high

    @classmethod
    def from_height_range(
        cls,
        height_range: Any,
        *,
        initial: float,
        step: float = 0.01,
    ) -> "HeightCommander":
        limits = np.asarray(height_range, dtype=np.float64)
        if limits.shape != (2,):
            raise ValueError(f"commands.height_range must have shape (2,), got {limits.shape}")
        commander = cls(low=float(limits[0]), high=float(limits[1]), step=float(step))
        commander.set(initial)
        return commander

    def set(self, value: float) -> None:
        self.target = float(np.clip(float(value), self.low, self.high))

    def nudge(self, sign: float) -> None:
        self.set(self.target + self.step * (1.0 if sign >= 0.0 else -1.0))

    def describe(self) -> str:
        return f"target_height={self.target:.3f} m range=[{self.low:.3f}, {self.high:.3f}]"


@dataclass(frozen=True)
class MotionOverlaySelection:
    """Cold-path selection of task bodies used by playback overlays."""

    enabled: bool
    selected_indices: np.ndarray


class PlaybackSession(Protocol):
    """Viewer-facing session contract shared by all policy families."""

    env: Any

    def reset(self) -> Any: ...

    def set_autoreset(self, enabled: bool) -> None: ...

    def advance(self, controls: PlaybackControls) -> bool: ...

    def physics_state(self) -> np.ndarray: ...

    @property
    def info(self) -> dict[str, Any]: ...


class RslRlPlaybackSession:
    """Policy/action stepping core shared by native and web viewers."""

    def __init__(
        self,
        *,
        env: Any,
        wrapped_env: Any,
        device: str,
        action_mode: str,
        policy: Callable[[Any], Any] | None,
        num_envs: int,
    ) -> None:
        self.env = env
        self.wrapped_env = wrapped_env
        self.device = device
        self.action_mode = action_mode
        self.policy = policy
        self.num_envs = int(num_envs)
        self.obs: Any | None = None
        self.action_obs: Any | None = None
        self.actions: torch.Tensor | None = None
        self.step_count = 0
        self.autoreset_enabled = True

    def reset(self) -> Any:
        self.obs, _info = self.wrapped_env.reset()
        self.action_obs = None
        self.actions = None
        self.step_count = 0
        return self.obs

    def set_autoreset(self, enabled: bool) -> None:
        """Synchronize environment reset behavior with session history lifecycle."""

        self.env.set_autoreset(bool(enabled))
        self.autoreset_enabled = bool(enabled)

    def refresh_observation(self) -> Any:
        """Reload the current env observation without advancing the session."""

        get_observations = getattr(self.wrapped_env, "get_observations", None)
        if not callable(get_observations):
            raise RuntimeError(
                "Playback observation refresh requires wrapped_env.get_observations()."
            )
        self.obs = get_observations()
        return self.obs

    def set_external_command(self, command: np.ndarray) -> Any:
        """Apply a velocity command and refresh every policy-facing observation."""

        if self.info.get("student_observation_contract") == "g1_official_oracle_student66_v1":
            self.env.set_external_command(command)
            return self.refresh_observation()
        state = getattr(self.env, "state", None)
        info = getattr(state, "info", None)
        commands = info.get("commands") if isinstance(info, dict) else None
        if not isinstance(commands, np.ndarray):
            raise RuntimeError("Playback command synchronization requires command owner rows.")
        command_arr = _external_velocity_command_rows(command, commands)
        if np.array_equal(commands[:, :3], command_arr):
            return self.obs

        commands[:, :3] = command_arr
        refresh_state = getattr(self.env, "refresh_state", None)
        if not callable(refresh_state):
            raise RuntimeError("Playback command synchronization requires env.refresh_state().")
        refresh_state()
        return self.refresh_observation()

    def set_external_height(self, target_height: float) -> Any:
        """Apply an external height target and refresh every policy-facing observation."""

        state = getattr(self.env, "state", None)
        info = getattr(state, "info", None)
        heights = info.get("height_commands") if isinstance(info, dict) else None
        if not isinstance(heights, np.ndarray) or heights.shape != (self.num_envs, 1):
            raise RuntimeError(
                "Playback height synchronization requires env.state.info['height_commands'] "
                f"with shape ({self.num_envs}, 1)."
            )
        target = float(target_height)
        if not np.isfinite(target):
            raise ValueError(f"Playback target height must be finite, got {target_height}")
        if np.all(heights[:, 0] == target):
            return self.obs
        heights[:, 0] = np.asarray(target, dtype=heights.dtype)
        refresh_state = getattr(self.env, "refresh_state", None)
        if not callable(refresh_state):
            raise RuntimeError("Playback height synchronization requires env.refresh_state().")
        refresh_state()
        return self.refresh_observation()

    def step_once(self) -> Any:
        actions = self._build_actions()
        self.actions = actions
        self.obs, _reward, _done, _info = self.wrapped_env.step(actions)
        self.step_count += 1
        return self.obs

    def advance(self, controls: PlaybackControls) -> bool:
        if not controls.consume_step_permission():
            return False
        self.step_once()
        return True

    def physics_state(self) -> np.ndarray:
        return self.env.get_physics_state_snapshot()

    @property
    def info(self) -> dict[str, Any]:
        state = getattr(self.env, "state", None)
        info = getattr(state, "info", None)
        return info if isinstance(info, dict) else {}

    def _build_actions(self) -> torch.Tensor:
        if self.obs is None:
            raise RuntimeError("Playback session must be reset before stepping.")
        self.action_obs = self.obs
        action_space = self.env.action_space
        action_dim = int(action_space.shape[0])
        if self.action_mode == "policy" and self.policy is not None:
            return self.policy(self.obs)
        if self.action_mode == "random":
            actions = np.random.uniform(
                action_space.low,
                action_space.high,
                size=(self.num_envs, action_dim),
            )
            return torch.from_numpy(actions).to(self.device).float()
        return torch.zeros(self.num_envs, action_dim, device=self.device)


class FADAPlaybackSession(RslRlPlaybackSession):
    """Stateful Planner-IDM playback session with episode-aligned histories."""

    def __init__(self, *, controller: Any | None, **kwargs: Any) -> None:
        self.controller = None
        super().__init__(policy=None, **kwargs)
        if controller is not None:
            self.bind_controller(controller)

    def bind_controller(self, controller: Any) -> None:
        """Bind the sole FADA controller through the public playback seam."""

        if self.action_mode != "policy":
            raise ValueError("FADA controller binding requires interactive.action_mode=policy")
        if controller is None:
            raise ValueError("FADA controller binding requires a controller")
        self.controller = controller
        self.policy = self._fada_policy

    def reset(self) -> Any:
        # B1: 环境 reset 与 FADA history reset 共享一个 lifecycle boundary.
        if self.controller is not None:
            self.controller.reset()
        return super().reset()

    def step_once(self) -> Any:
        # B1: 先以当前 observation/command 计算并执行第一动作.
        actions = self._build_actions()
        self.actions = actions
        self.obs, _reward, done, _info = self.wrapped_env.step(actions)
        # B2: 将 episode 边界交回 history owner, 下一查询按 reset observation 初始化对应行.
        if self.controller is not None and self.autoreset_enabled:
            self.controller.reset(done)
        self.step_count += 1
        return self.obs

    def _fada_policy(self, observation: Any) -> torch.Tensor:
        commands = self.info.get("commands")
        if commands is None:
            raise RuntimeError(
                "FADA playback requires env.state.info['commands'] as the complete task command."
            )
        if self.controller is None:
            raise RuntimeError("FADA playback controller is unavailable.")
        if self.controller.policy.config.observation_contract == "g1_official_oracle_student66_v1":
            if self.info.get("student_observation_contract") != "g1_official_oracle_student66_v1":
                raise ValueError("Official Planner-IDM playback student view identity mismatch")
            return self.controller.act_projected(self.env.get_student_observation(), commands)
        if self.info.get("student_observation_contract") == "g1_official_oracle_student66_v1":
            raise ValueError("Legacy Planner-IDM cannot run in the Official Oracle environment")
        return self.controller.act(observation, commands)


class OffPolicyPlaybackSession:
    """Direct env stepping session for SAC-style off-policy actors."""

    def __init__(
        self,
        *,
        env: Any,
        device: str,
        action_mode: str,
        actor: Any | None,
        actor_algo_type: str,
        normalizer: Any | None,
        num_envs: int,
        obs_extractor: Callable[[dict[str, np.ndarray]], np.ndarray],
        priv_info_resolver: Callable[..., np.ndarray | None],
    ) -> None:
        self.env = env
        self.device = device
        self.action_mode = action_mode
        self.actor = actor
        self.actor_algo_type = str(actor_algo_type)
        self.normalizer = normalizer
        self.num_envs = int(num_envs)
        self.obs_extractor = obs_extractor
        self.priv_info_resolver = priv_info_resolver
        self.obs: np.ndarray | None = None
        self.current_priv_info: np.ndarray | None = None
        self.step_count = 0
        self.autoreset_enabled = True

    def reset(self) -> np.ndarray:
        state = self.env.reset_all()
        self.obs = np.asarray(self.obs_extractor(state.obs), dtype=np.float32)
        self.current_priv_info = self._resolve_priv_info(state.obs, state.info)
        self.step_count = 0
        return self.obs

    def set_autoreset(self, enabled: bool) -> None:
        """Synchronize direct environment reset behavior with the playback session."""

        self.env.set_autoreset(bool(enabled))
        self.autoreset_enabled = bool(enabled)

    def step_once(self) -> np.ndarray:
        actions = self._build_actions()
        state = self.env.step(actions)
        self.obs = np.asarray(self.obs_extractor(state.obs), dtype=np.float32)
        self.current_priv_info = self._resolve_priv_info(state.obs, state.info)
        self.step_count += 1
        return self.obs

    def refresh_observation(self) -> np.ndarray:
        """Refresh direct off-policy observations without advancing physics."""

        refresh_state = getattr(self.env, "refresh_state", None)
        if not callable(refresh_state):
            raise RuntimeError("Playback observation refresh requires env.refresh_state().")
        state = refresh_state()
        self.obs = np.asarray(self.obs_extractor(state.obs), dtype=np.float32)
        self.current_priv_info = self._resolve_priv_info(state.obs, state.info)
        return self.obs

    def set_external_command(self, command: np.ndarray) -> np.ndarray:
        """Apply an external velocity command before the next policy action."""

        commands = self.info.get("commands")
        if not isinstance(commands, np.ndarray):
            raise RuntimeError("Playback command synchronization requires command owner rows.")
        command_arr = _external_velocity_command_rows(command, commands)
        if np.array_equal(commands[:, :3], command_arr):
            if self.obs is None:
                return self.refresh_observation()
            return self.obs
        commands[:, :3] = command_arr
        return self.refresh_observation()

    def set_external_height(self, target_height: float) -> np.ndarray:
        """Apply an external height target before the next policy action."""

        heights = self.info.get("height_commands")
        if not isinstance(heights, np.ndarray) or heights.shape != (self.num_envs, 1):
            raise RuntimeError(
                "Playback height synchronization requires env.state.info['height_commands'] "
                f"with shape ({self.num_envs}, 1)."
            )
        target = float(target_height)
        if not np.isfinite(target):
            raise ValueError(f"Playback target height must be finite, got {target_height}")
        if np.all(heights[:, 0] == target):
            if self.obs is None:
                return self.refresh_observation()
            return self.obs
        heights[:, 0] = np.asarray(target, dtype=heights.dtype)
        return self.refresh_observation()

    def advance(self, controls: PlaybackControls) -> bool:
        if not controls.consume_step_permission():
            return False
        self.step_once()
        return True

    def physics_state(self) -> np.ndarray:
        return self.env.get_physics_state_snapshot()

    @property
    def info(self) -> dict[str, Any]:
        state = getattr(self.env, "state", None)
        info = getattr(state, "info", None)
        return info if isinstance(info, dict) else {}

    def _resolve_priv_info(
        self,
        obs_dict: dict[str, np.ndarray],
        info: dict[str, Any] | None,
    ) -> np.ndarray | None:
        from unilab.algos.torch.offpolicy.worker import offpolicy_actor_requires_priv_info

        if not offpolicy_actor_requires_priv_info(self.actor_algo_type):
            return None
        if self.action_mode != "policy" or self.actor is None:
            return None
        from unilab.base.observations import split_obs_dict

        actor_obs_np, critic_np = split_obs_dict(obs_dict)
        priv_info = self.priv_info_resolver(
            algo_type=self.actor_algo_type,
            obs_np=np.asarray(actor_obs_np, dtype=np.float32),
            critic_np=np.asarray(critic_np, dtype=np.float32),
            info=info,
        )
        if priv_info is None:
            raise ValueError(
                f"{self.actor_algo_type} interactive play step is missing privileged info."
            )
        return np.asarray(priv_info, dtype=np.float32)

    def _build_actions(self) -> np.ndarray:
        if self.obs is None:
            raise RuntimeError("Playback session must be reset before stepping.")
        action_space = self.env.action_space
        action_dim = int(action_space.shape[0])
        if self.action_mode == "policy" and self.actor is not None:
            from unilab.algos.torch.offpolicy.worker import offpolicy_actor_requires_priv_info

            obs_torch = torch.from_numpy(self.obs).to(self.device)
            if self.normalizer is not None:
                obs_torch = self.normalizer(obs_torch, update=False)
            if offpolicy_actor_requires_priv_info(self.actor_algo_type):
                if self.current_priv_info is None:
                    raise ValueError(
                        f"{self.actor_algo_type} interactive play step is missing privileged info."
                    )
                priv_info_torch = torch.from_numpy(self.current_priv_info).to(self.device)
                actions = self.actor.explore(
                    obs_torch,
                    priv_info_torch,
                    deterministic=True,
                )
            else:
                actions = self.actor.explore(obs_torch, deterministic=True)
            return actions.detach().cpu().numpy().astype(np.float32)
        if self.action_mode == "random":
            return np.random.uniform(
                action_space.low,
                action_space.high,
                size=(self.num_envs, action_dim),
            ).astype(np.float32)
        return np.zeros((self.num_envs, action_dim), dtype=np.float32)


_HORA_DISTILL_CHECKPOINT_UNAVAILABLE = "hora_distill_checkpoint_unavailable"
