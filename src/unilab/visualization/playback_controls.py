"""Keyboard, height, and command-probe helpers for interactive playback."""

from __future__ import annotations

from typing import Any

import numpy as np

from .playback_sessions import HeightCommander, KeyboardCommander, RslRlPlaybackConfig

_KEY_ENTER, _KEY_KP_ENTER = 257, 335
_KEY_RIGHT, _KEY_LEFT, _KEY_DOWN, _KEY_UP = 262, 263, 264, 265
_COMMAND_OBS_VERIFY_COMMAND = np.array([0.37, -0.23, 0.19], dtype=np.float64)
_VELOCITY_COMMAND_TASK_NAME_MARKERS = ("Joystick", "Walk")


def _build_playback_config(args, *, num_envs: int = 1) -> RslRlPlaybackConfig:
    return RslRlPlaybackConfig(
        task=str(args.task),
        load_run=str(args.load_run),
        checkpoint=getattr(args, "checkpoint", None),
        checkpoint_path=getattr(args, "checkpoint_path", None),
        action_mode=str(args.action_mode),
        policy_obs_mode=str(args.policy_obs_mode),
        algo_log_name=str(getattr(args, "algo_log_name", "rsl_rl_ppo")),
        log_root=getattr(args, "log_root", None),
        num_envs=num_envs,
        speed=float(getattr(args, "speed", 1.0)),
        start_paused=bool(getattr(args, "start_paused", False)),
        keyboard=bool(getattr(args, "keyboard", False)),
    )


def _build_keyboard_commander(env: Any, args) -> KeyboardCommander | None:
    """Set up keyboard velocity teleop, or return None when unsupported/disabled."""
    if not bool(getattr(args, "keyboard", False)):
        return None

    state = getattr(env, "state", None)
    command_arr = state.info.get("commands") if state is not None else None
    if state is not None and state.info.get("student_observation_contract") == "g1_official_oracle_student66_v1":
        commander = KeyboardCommander.from_vel_limit(
            ((-1., -1., -1.), (1., 1., 1.)),
            step_lin=float(getattr(args, "keyboard_step_lin", .1)),
            step_ang=float(getattr(args, "keyboard_step_ang", .2)))
        env.set_external_command(commander.command)
        return commander
    cmds_cfg = getattr(getattr(env, "cfg", None), "commands", None)
    if not isinstance(command_arr, np.ndarray) or cmds_cfg is None:
        print("[play_interactive] interactive.keyboard ignored: task has no velocity 'commands'.")
        return None

    cmds_cfg.heading_command = False
    cmds_cfg.resampling_time = 0.0

    commander = KeyboardCommander.from_vel_limit(
        cmds_cfg.vel_limit,
        step_lin=float(getattr(args, "keyboard_step_lin", 0.1)),
        step_ang=float(getattr(args, "keyboard_step_ang", 0.2)),
    )
    env.state.info["commands"][:] = commander.command
    return commander


def _build_height_commander(env: Any, args) -> HeightCommander | None:
    """Set up external target-height control when the task exposes its owner array."""

    if not bool(getattr(args, "keyboard", False)):
        return None
    state = getattr(env, "state", None)
    info = getattr(state, "info", None)
    height_commands = info.get("height_commands") if isinstance(info, dict) else None
    commands_cfg = getattr(getattr(env, "cfg", None), "commands", None)
    height_range = getattr(commands_cfg, "height_range", None)
    if not isinstance(height_commands, np.ndarray) or height_commands.shape != (1, 1):
        return None
    if height_range is None:
        raise RuntimeError("height_commands is present but commands.height_range is missing")
    commands_cfg.resampling_time = 0.0
    return HeightCommander.from_height_range(
        height_range,
        initial=float(height_commands[0, 0]),
        step=float(getattr(args, "keyboard_step_height", 0.01)),
    )


def _apply_playback_command(playback_session: Any, command: np.ndarray) -> None:
    setter = getattr(playback_session, "set_external_command", None)
    if not callable(setter):
        raise RuntimeError("Keyboard playback requires playback_session.set_external_command().")
    setter(np.asarray(command, dtype=np.float32))


def _apply_playback_height(playback_session: Any, target_height: float) -> None:
    setter = getattr(playback_session, "set_external_height", None)
    if not callable(setter):
        raise RuntimeError("Playback session does not support external target-height input.")
    setter(float(target_height))


def _measured_base_height(env: Any) -> float | None:
    resolver = getattr(env, "_terrain_relative_base_height", None)
    if not callable(resolver):
        return None
    measured = np.asarray(resolver(), dtype=np.float64).reshape(-1)
    return float(measured[0]) if measured.size else None


def _print_height_status(env: Any, commander: HeightCommander) -> None:
    measured = _measured_base_height(env)
    measured_text = "unavailable" if measured is None else f"{measured:.3f} m"
    print(f"[play_interactive] {commander.describe()} measured_height={measured_text}")


def _state_has_velocity_commands(env: Any) -> bool:
    state = getattr(env, "state", None)
    info = getattr(state, "info", None) if state is not None else None
    command_arr = info.get("commands") if isinstance(info, dict) else None
    return (
        isinstance(command_arr, np.ndarray)
        and command_arr.ndim == 2
        and command_arr.shape[0] > 0
        and command_arr.shape[1] >= 3
    )


def _is_locomotion_env(env: Any) -> bool:
    return type(env).__module__.startswith("unilab.envs.locomotion")


def _is_velocity_command_locomotion_task(env: Any) -> bool:
    if not _is_locomotion_env(env):
        return False
    cfg = getattr(env, "cfg", None)
    candidate_names = [
        type(env).__name__,
        type(cfg).__name__ if cfg is not None else "",
        type(env).__module__,
        type(cfg).__module__ if cfg is not None else "",
    ]
    return any(
        marker in candidate
        for candidate in candidate_names
        for marker in _VELOCITY_COMMAND_TASK_NAME_MARKERS
    )


def _should_render_velocity_arrows(env: Any, *, reset_fn=None) -> bool:
    if not _is_velocity_command_locomotion_task(env):
        return False
    if not _state_has_velocity_commands(env):
        return False
    if reset_fn is None:
        return _state_policy_obs_contains_command(env)
    return _policy_obs_contains_command(env, reset_fn=reset_fn)


def _row_contains_contiguous_vector(
    row: np.ndarray,
    vector: np.ndarray,
    *,
    atol: float = 1.0e-6,
) -> bool:
    values = np.asarray(row, dtype=np.float64).reshape(-1)
    target = np.asarray(vector, dtype=np.float64).reshape(-1)
    if target.size == 0 or values.size < target.size:
        return False
    for start in range(values.size - target.size + 1):
        if np.allclose(values[start : start + target.size], target, atol=atol, rtol=0.0):
            return True
    return False


def _state_policy_obs_contains_command(env: Any) -> bool:
    if not _state_has_velocity_commands(env):
        return False

    state = env.state
    obs = getattr(state, "obs", None)
    actor_obs = obs.get("obs") if isinstance(obs, dict) else None
    if not isinstance(actor_obs, np.ndarray) or actor_obs.ndim != 2 or actor_obs.shape[0] == 0:
        return False

    command = np.asarray(state.info["commands"][0, :3], dtype=np.float64)
    if np.linalg.norm(command) <= 1.0e-9:
        return False
    return _row_contains_contiguous_vector(actor_obs[0], command)


def _force_policy_command_probe_obs(env: Any, command: np.ndarray) -> None:
    state = getattr(env, "state", None)
    info = getattr(state, "info", None)
    if not isinstance(info, dict):
        return

    commands = info.get("commands")
    if not isinstance(commands, np.ndarray):
        return
    if commands.ndim != 2 or commands.shape[1] < 3:
        return

    commands[:, :3] = np.asarray(command, dtype=commands.dtype)

    refresh_state = getattr(env, "refresh_state", None)
    if not callable(refresh_state):
        return
    refresh_state()


def _snapshot_policy_command_probe_state(env: Any) -> np.ndarray | None:
    state = getattr(env, "state", None)
    info = getattr(state, "info", None)
    commands = info.get("commands") if isinstance(info, dict) else None
    return np.array(commands, copy=True) if isinstance(commands, np.ndarray) else None


def _restore_policy_command_probe_state(
    env: Any,
    commands: np.ndarray | None,
) -> None:
    state = getattr(env, "state", None)
    info = getattr(state, "info", None)
    if commands is not None and isinstance(info, dict):
        current_commands = info.get("commands")
        if isinstance(current_commands, np.ndarray) and current_commands.shape == commands.shape:
            current_commands[...] = commands
        else:
            info["commands"] = commands.copy()
        refresh_state = getattr(env, "refresh_state", None)
        if callable(refresh_state):
            # reset_fn may have changed physics and gait phase. Both actor and
            # critic observations must describe that current state, not the
            # cached observation from before the probe's resets.
            refresh_state()


def _policy_obs_contains_command(env: Any, *, reset_fn) -> bool:
    if _state_policy_obs_contains_command(env):
        return True

    cmds_cfg = getattr(getattr(env, "cfg", None), "commands", None)
    if cmds_cfg is None or not hasattr(cmds_cfg, "vel_limit"):
        return False

    original_vel_limit = cmds_cfg.vel_limit
    original_rel_standing_envs = getattr(cmds_cfg, "rel_standing_envs", None)
    original_commands = _snapshot_policy_command_probe_state(env)
    probe = _COMMAND_OBS_VERIFY_COMMAND.tolist()
    try:
        cmds_cfg.vel_limit = [probe, probe]
        if original_rel_standing_envs is not None:
            cmds_cfg.rel_standing_envs = 0.0
        reset_fn()
        _force_policy_command_probe_obs(env, _COMMAND_OBS_VERIFY_COMMAND)
        return _state_policy_obs_contains_command(env)
    finally:
        cmds_cfg.vel_limit = original_vel_limit
        if original_rel_standing_envs is not None:
            cmds_cfg.rel_standing_envs = original_rel_standing_envs
        reset_fn()
        _restore_policy_command_probe_state(env, original_commands)


def _handle_command_key(commander: KeyboardCommander, keycode: int) -> None:
    if keycode == _KEY_UP:
        commander.nudge(commander.AXIS_VX, +1.0)
    elif keycode == _KEY_DOWN:
        commander.nudge(commander.AXIS_VX, -1.0)
    elif keycode == _KEY_LEFT:
        commander.nudge(commander.AXIS_VY, +1.0)
    elif keycode == _KEY_RIGHT:
        commander.nudge(commander.AXIS_VY, -1.0)
    elif keycode in (ord("Q"), ord("q")):
        commander.nudge(commander.AXIS_VYAW, +1.0)
    elif keycode in (ord("E"), ord("e")):
        commander.nudge(commander.AXIS_VYAW, -1.0)
    elif keycode in (_KEY_ENTER, _KEY_KP_ENTER):
        commander.zero()
    else:
        return
    print(f"[play_interactive] {commander.describe()}")


def _handle_height_key(commander: HeightCommander, keycode: int) -> bool:
    if keycode == ord("["):
        commander.nudge(-1.0)
    elif keycode == ord("]"):
        commander.nudge(+1.0)
    else:
        return False
    return True


def _print_keyboard_legend(args, *, height_control: bool = False) -> None:
    print("[play_interactive] Keyboard teleop ENABLED (drive style):")
    print("  Up / Down    : forward / backward (vx)")
    print("  Left / Right : translate left / right (vy)")
    print("  Q / E        : turn left / right (vyaw)")
    print("  Enter        : full stop")
    if height_control:
        step = float(getattr(args, "keyboard_step_height", 0.01))
        print(f"  [ / ]        : target height -/+ {step:.3f} m")
    if str(getattr(args, "action_mode", "")) != "policy":
        print("  NOTE: action_mode is not 'policy'; commands will not drive the robot.")
