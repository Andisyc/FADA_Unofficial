"""MuJoCo viewer composition and lifecycle for interactive playback."""

from __future__ import annotations

import tempfile
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from unilab.base import registry
from unilab.base.backend.mujoco.playback import resolve_render_play_model_files
from unilab.base.scene import SceneCfg
from unilab.training import BackendAdapter, create_env, get_entrypoint_log_root
from unilab.training.rsl_rl import RslRlVecEnvWrapper, get_policy_obs_dims, normalize_ppo_train_cfg
from unilab.visualization.interactive_playback import (
    _HORA_DISTILL_CHECKPOINT_UNAVAILABLE,
    HeightCommander,
    KeyboardCommander,
    PlaybackControls,
    _apply_distill_playback_reset_contract,
    create_appo_playback_session,
    create_distill_playback_session,
    create_fada_playback_session,
    create_hora_distill_playback_session,
    create_rsl_rl_playback_session,
    create_sac_playback_session,
    prepare_motion_overlay_selection,
    select_torch_device,
)
from unilab.visualization.playback_checkpoint_contract import (
    _apply_checkpoint_env_contract,
    _infer_checkpoint_actor_input_dim,
    _warn_if_g1_sac_checkpoint_lacks_standing_contract,
    resolve_checkpoint,
)
from unilab.visualization.playback_cli import PlayInteractiveArgs, _algo_config_dict
from unilab.visualization.playback_controls import (
    _apply_playback_command,
    _apply_playback_height,
    _build_height_commander,
    _build_keyboard_commander,
    _build_playback_config,
    _handle_command_key,
    _handle_height_key,
    _policy_obs_contains_command,
    _print_height_status,
    _print_keyboard_legend,
    _should_render_velocity_arrows,
    _state_has_velocity_commands,
)
from unilab.visualization.playback_overlay import (
    _render_motion_targets,
    _render_reward_debug_targets,
    _render_velocity_arrows,
)
from unilab.visualization.playback_trace import (
    _env_flag,
    _env_positive_int,
    _load_trace_standing_teacher,
    _print_distill_action_trace,
)

try:
    from rsl_rl.runners import OnPolicyRunner
except ImportError:
    OnPolicyRunner = None

ROOT_DIR = Path(__file__).parents[3]
_KEY_BACKSPACE = 259
_VELOCITY_ARROW_HEIGHT = 0.6
_VELOCITY_ARROW_SCALE = 0.45
_VELOCITY_ARROW_WIDTH = 0.025
_VELOCITY_ARROW_LATERAL_OFFSET = 0.0
_OFFPOLICY_INTERACTIVE_ALGOS = {"sac", "flashsac"}
_PLAYBACK_ENV_UNAVAILABLE = "playback_env_unavailable"
_DEFAULT_CAMERA_DISTANCE = 2.0
_TERRAIN_FOLLOW_CAMERA_DISTANCE = 3.0
_FOLLOW_CAMERA_MAX_DISTANCE = 6.0


def _resolve_focus_body_id(mj_model, env, preferred_name: str) -> int:
    candidate_names: list[str] = []
    if preferred_name.strip():
        candidate_names.append(preferred_name.strip())

    cfg = getattr(env, "cfg", None)
    asset = getattr(cfg, "asset", None) if cfg is not None else None
    if asset is not None and getattr(asset, "base_name", None):
        candidate_names.append(str(asset.base_name))
    if cfg is not None and getattr(cfg, "base_name", None):
        candidate_names.append(str(cfg.base_name))

    candidate_names.extend(["base", "trunk", "pelvis", "torso", "torso_link"])
    for name in candidate_names:
        try:
            body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        except Exception:
            body_id = -1
        if body_id >= 0:
            return int(body_id)

    nbody = int(getattr(mj_model, "nbody", 1))
    return 1 if nbody > 1 else 0


def _has_generated_terrain(env: Any) -> bool:
    scene = getattr(getattr(env, "cfg", None), "scene", None)
    return getattr(scene, "terrain", None) is not None


def _default_viewer_camera_distance(mj_model, env: Any, *, follow_body: bool) -> float:
    model_extent = float(getattr(getattr(mj_model, "stat", None), "extent", 1.0))
    extent_distance = max(_DEFAULT_CAMERA_DISTANCE, 2.5 * model_extent)
    if not follow_body:
        return extent_distance
    if _has_generated_terrain(env):
        return _TERRAIN_FOLLOW_CAMERA_DISTANCE
    return min(extent_distance, _FOLLOW_CAMERA_MAX_DISTANCE)


def _available_backends_for_task(task_name: str) -> tuple[str, ...]:
    envs = registry.list_registered_envs()
    task_meta = envs.get(task_name, {})
    backends = task_meta.get("available_backends", ())
    if not isinstance(backends, list):
        return ()
    return tuple(str(backend) for backend in backends)


def _can_launch_glfw_viewer() -> bool:
    try:
        import glfw
    except Exception:
        return True

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ok = bool(glfw.init())
    if ok:
        glfw.terminate()
    return ok


def _uses_native_mujoco_viewer_launch() -> bool:
    launch_fn = getattr(mujoco.viewer, "launch_passive", None)
    module_name = str(getattr(launch_fn, "__module__", ""))
    return module_name.startswith("mujoco")


def _backend_adapter(cfg: DictConfig, *, algo_name: str = "ppo"):
    from unilab.base.backend.mujoco.xml import materialize_scene_visual_override

    return BackendAdapter(
        cfg,
        root_dir=ROOT_DIR,
        algo_name=algo_name,
        scene_materializer=materialize_scene_visual_override,
    )


def _select_playback_device(cfg: DictConfig | None) -> str:
    configured = OmegaConf.select(cfg, "training.device") if cfg is not None else None
    if configured not in (None, ""):
        return str(configured)
    return select_torch_device()


def _load_mujoco_model_file_for_viewer(model_file: str):
    if Path(model_file).suffix.lower() == ".mjb":
        return mujoco.MjModel.from_binary_path(str(model_file))
    return mujoco.MjModel.from_xml_path(str(model_file))


def _load_resolved_visual_viewer_model(env: Any):
    try:
        with tempfile.TemporaryDirectory(prefix="unilab-interactive-viewer-") as tmp_dir:
            model_files = resolve_render_play_model_files(env, num_envs=1, tmp_dir=tmp_dir)
            model_file = model_files[0] if isinstance(model_files, list) else model_files
            print(
                f"[play_interactive] Using resolved visual playback model for viewer: {model_file}"
            )
            return _load_mujoco_model_file_for_viewer(str(model_file))
    except Exception as exc:
        print(
            "[play_interactive] WARNING: failed to resolve visual playback model; "
            f"falling back to visual model ({exc})."
        )
        return None


def _load_viewer_model(env: Any, *, use_env_visual_model: bool):
    backend_visual_model_file = env.get_scene_artifacts().visual_model_file
    if backend_visual_model_file:
        resolved = _load_resolved_visual_viewer_model(env)
        if resolved is not None:
            return resolved
        print(
            f"[play_interactive] Using backend visual model for viewer: {backend_visual_model_file}"
        )
        return mujoco.MjModel.from_xml_path(str(backend_visual_model_file))

    if use_env_visual_model:
        cfg_scene = getattr(getattr(env, "cfg", None), "scene", None)
        if cfg_scene is not None and not isinstance(cfg_scene, SceneCfg):
            raise TypeError("env.cfg.scene must be a SceneCfg")
        model_file = None if cfg_scene is None else cfg_scene.model_file
        if model_file:
            try:
                resolved = _load_resolved_visual_viewer_model(env)
                if resolved is not None:
                    return resolved
                print(f"[play_interactive] Using configured visual model for viewer: {model_file}")
                return mujoco.MjModel.from_xml_path(str(model_file))
            except Exception as exc:
                print(
                    "[play_interactive] WARNING: failed to load configured visual model; "
                    f"falling back to playback model ({exc})."
                )

    try:
        playback_model = env.get_playback_model()
    except NotImplementedError as exc:
        raise AttributeError("Environment does not expose a playback model contract") from exc
    if isinstance(playback_model, str):
        print(f"[play_interactive] Using playback model for viewer: {playback_model}")
        return mujoco.MjModel.from_xml_path(playback_model)
    print("[play_interactive] Using backend playback model for viewer.")
    return playback_model


@dataclass(frozen=True)
class _InteractiveViewerRuntime:
    session: Any
    env: Any
    overlay: Any
    model: Any
    data: Any
    state_spec: Any
    ctrl_dt: float
    controls: PlaybackControls
    commander: KeyboardCommander | None
    height_commander: HeightCommander | None
    trace_distill_actions: bool
    trace_interval: int
    trace_standing_teacher: Any
    render_velocity_arrows: bool
    key_callback: Callable[[int], None]


def _build_interactive_env_factory(args, cfg, *, algo: str, available_backends):
    def create(num_envs: int):
        if cfg is None:
            return registry.make(args.task, num_envs=num_envs, sim_backend="mujoco")

        if algo in _OFFPOLICY_INTERACTIVE_ALGOS:
            from train_offpolicy import build_offpolicy_env_cfg_override

            env_cfg_override = _apply_checkpoint_env_contract(
                build_offpolicy_env_cfg_override(algo, cfg), args
            )
            adapter = _backend_adapter(cfg, algo_name=algo)
            env_cfg_override = adapter.build_play_env_cfg_override(env_cfg_override)
            adapter.report_effective_settings(env_cfg_override, mode="回放")
        else:
            adapter_algo = "distill" if algo == "fada" else algo
            adapter = _backend_adapter(cfg, algo_name=adapter_algo)
            env_cfg_override = adapter.build_task_env_cfg_override()
            if str(cfg.training.task_name) == "G1OfficialOracleFlat":
                env_cfg_override = adapter.build_play_env_cfg_override(env_cfg_override)
            if algo == "distill":
                env_cfg_override = _apply_distill_playback_reset_contract(
                    env_cfg_override, args.task
                )
        try:
            return create_env(
                cfg,
                num_envs=num_envs,
                env_cfg_override=env_cfg_override,
                sim_backend="mujoco",
                task_name=args.task,
            )
        except ValueError as exc:
            if "does not support simulation backend 'mujoco'" not in str(exc):
                raise
            print(
                "[play_interactive] Task does not support MuJoCo backend: "
                f"{args.task}. Available backends: {available_backends or ('<none>',)}. "
                "This script only supports MuJoCo viewer mode."
            )
            raise RuntimeError(_PLAYBACK_ENV_UNAVAILABLE) from exc

    return create


def _create_interactive_session(
    args,
    cfg,
    *,
    algo: str,
    device: torch.device,
    available_backends,
    fada_session_factory: Callable[..., Any] | None,
):
    playback_cfg = _build_playback_config(args, num_envs=1)
    env_factory = _build_interactive_env_factory(
        args, cfg, algo=algo, available_backends=available_backends
    )

    def log(message: str) -> None:
        print(f"[play_interactive] {message}")

    if algo == "ppo":
        wrapper_cls = RslRlVecEnvWrapper
        if cfg is not None:
            from unilab.algos.torch.rsl_rl_runtime import resolve_rsl_rl_ppo_runtime

            runtime = resolve_rsl_rl_ppo_runtime(
                _algo_config_dict(cfg), default_wrapper_cls=RslRlVecEnvWrapper
            )
            wrapper_cls = runtime.wrapper_cls
            if runtime.playback_factory is not None:
                return runtime.playback_factory(playback_cfg=playback_cfg,
                    env_factory=env_factory, checkpoint_resolver=resolve_checkpoint,
                    wrapper_cls=wrapper_cls, device=device, log=log)
        return create_rsl_rl_playback_session(
            playback_cfg=playback_cfg,
            env_factory=env_factory,
            algo_config=_algo_config_dict(cfg),
            root_dir=ROOT_DIR,
            device=device,
            checkpoint_resolver=resolve_checkpoint,
            checkpoint_input_dim_reader=_infer_checkpoint_actor_input_dim,
            entrypoint_log_root=get_entrypoint_log_root,
            wrapper_cls=wrapper_cls,
            runner_cls=OnPolicyRunner,
            policy_obs_dims_getter=get_policy_obs_dims,
            train_cfg_normalizer=normalize_ppo_train_cfg,
            log=log,
        )
    config_required_algos = {
        "appo",
        "hora_distill",
        "distill",
        "fada",
        *_OFFPOLICY_INTERACTIVE_ALGOS,
    }
    if cfg is None and algo in config_required_algos:
        labels = {
            "appo": "APPO",
            "hora_distill": "HORA distill",
            "distill": "Generic distill",
            "fada": "FADA",
        }
        label = labels.get(algo, algo)
        raise ValueError(f"{label} interactive playback requires a composed Hydra config.")
    if algo == "appo":
        return create_appo_playback_session(
            playback_cfg=playback_cfg,
            cfg=cfg,
            rl_cfg=_algo_config_dict(cfg),
            env_factory=env_factory,
            root_dir=ROOT_DIR,
            device=device,
            wrapper_cls=RslRlVecEnvWrapper,
            log=log,
        )
    if algo in _OFFPOLICY_INTERACTIVE_ALGOS:
        session = create_sac_playback_session(
            playback_cfg=playback_cfg,
            cfg=cfg,
            env_factory=env_factory,
            root_dir=ROOT_DIR,
            device=device,
            algo_name=algo,
            log=log,
        )
        _warn_if_g1_sac_checkpoint_lacks_standing_contract(
            algo=algo, task_name=str(args.task), checkpoint_path=session[2], log=log
        )
        return session
    factories = {
        "hora_distill": create_hora_distill_playback_session,
        "distill": create_distill_playback_session,
        "fada": fada_session_factory or create_fada_playback_session,
    }
    factory = factories.get(algo)
    if factory is None:
        raise ValueError(f"Unsupported interactive playback algo: {algo}")
    return factory(
        playback_cfg=playback_cfg,
        cfg=cfg,
        root_dir=ROOT_DIR,
        device=device,
        log=log,
    )


create_interactive_session = _create_interactive_session


def _prepare_viewer_runtime(args, cfg, *, algo: str, device, playback_session):
    env = playback_session.env
    overlay = prepare_motion_overlay_selection(
        env,
        show_target_bodies=bool(args.show_target_bodies),
        show_reward_debug=bool(args.show_reward_debug),
        target_body_names=str(args.target_body_names),
        target_max_bodies=int(args.target_max_bodies),
        log=lambda message: print(f"[play_interactive] {message}"),
    )
    if overlay.enabled:
        print(
            "[play_interactive] Target visualization enabled "
            f"({overlay.selected_indices.size} bodies, axes={args.target_show_axes})."
        )
    if args.show_reward_debug:
        print(
            "[play_interactive] Reward debug overlay enabled "
            f"(vel={args.reward_debug_show_velocity}, connectors={args.reward_debug_show_connectors}, "
            f"global_anchor={args.reward_debug_show_global_anchor})."
        )
    model = _load_viewer_model(
        env, use_env_visual_model=bool(getattr(args, "use_env_visual_model", True))
    )
    data = mujoco.MjData(model)
    playback_session.reset()
    render_velocity_arrows = str(args.action_mode) == "policy" and _should_render_velocity_arrows(
        env, reset_fn=playback_session.reset
    )
    if render_velocity_arrows:
        print("[play_interactive] Velocity arrows enabled (green=target, blue=current).")
    if bool(getattr(args, "keyboard", False)) and bool(
        getattr(args, "require_keyboard_command_obs", True)
    ):
        if not _state_has_velocity_commands(env):
            raise RuntimeError(
                "interactive.keyboard unavailable: task state has no velocity 'commands'."
            )
        if not _policy_obs_contains_command(env, reset_fn=playback_session.reset):
            raise RuntimeError(
                "interactive.keyboard unavailable: policy obs does not contain the velocity command."
            )
        playback_session.refresh_observation()
    controls = PlaybackControls(
        paused=bool(getattr(args, "start_paused", False)),
        speed=float(getattr(args, "speed", 1.0)),
    )
    commander = _build_keyboard_commander(env, args)
    height_commander = _build_height_commander(env, args)
    if commander is not None:
        _apply_playback_command(playback_session, commander.command)
        playback_session.set_autoreset(False)
    if height_commander is not None:
        _apply_playback_height(playback_session, height_commander.target)
        playback_session.set_autoreset(False)
        _print_height_status(env, height_commander)
    trace_distill_actions = (
        algo == "distill"
        and str(getattr(args, "action_mode", "")) == "policy"
        and _env_flag("UNILAB_G1_ACTION_TRACE")
    )
    trace_interval = _env_positive_int("UNILAB_G1_ACTION_TRACE_INTERVAL", 20)
    trace_standing_teacher = None
    if trace_distill_actions:
        trace_standing_teacher = _load_trace_standing_teacher(
            cfg, device=device, log=lambda message: print(f"[play_interactive] {message}")
        )
        print(
            "[play_interactive] Distill action trace enabled "
            f"(interval={trace_interval}, standing_teacher={trace_standing_teacher is not None})."
        )

    def on_key(keycode: int) -> None:
        if keycode == ord(" "):
            print(
                f"[play_interactive] {'paused' if controls.toggle_pause() else 'resumed'} (space)"
            )
        elif keycode in (ord("N"), ord("n")):
            controls.request_single_step()
            if not controls.paused:
                controls.pause()
                print("[play_interactive] paused for single-step mode (n)")
            print("[play_interactive] single step requested (n)")
        elif keycode in (ord("+"), ord("=")):
            controls.set_speed(controls.speed * 1.25)
            print(f"[play_interactive] speed={controls.speed:.2f}x")
        elif keycode in (ord("-"), ord("_")):
            controls.set_speed(controls.speed / 1.25)
            print(f"[play_interactive] speed={controls.speed:.2f}x")
        elif (commander is not None or height_commander is not None) and keycode == _KEY_BACKSPACE:
            playback_session.reset()
            if commander is not None:
                commander.zero()
                _apply_playback_command(playback_session, commander.command)
            if height_commander is not None:
                _apply_playback_height(playback_session, height_commander.target)
                _print_height_status(env, height_commander)
            print("[play_interactive] reset (backspace)")
        elif height_commander is not None and _handle_height_key(height_commander, keycode):
            _apply_playback_height(playback_session, height_commander.target)
            _print_height_status(env, height_commander)
        elif commander is not None:
            _handle_command_key(commander, keycode)

    return _InteractiveViewerRuntime(
        session=playback_session,
        env=env,
        overlay=overlay,
        model=model,
        data=data,
        state_spec=mujoco.mjtState.mjSTATE_FULLPHYSICS,
        ctrl_dt=float(env.cfg.ctrl_dt),
        controls=controls,
        commander=commander,
        height_commander=height_commander,
        trace_distill_actions=trace_distill_actions,
        trace_interval=trace_interval,
        trace_standing_teacher=trace_standing_teacher,
        render_velocity_arrows=render_velocity_arrows,
        key_callback=on_key,
    )


def _render_interactive_frame(
    viewer, runtime: _InteractiveViewerRuntime, args, *, focus_body_id: int
) -> None:
    session, env = runtime.session, runtime.env
    advanced = session.advance(runtime.controls)
    phys = session.physics_state()[0].astype(np.float64)
    mujoco.mj_setState(runtime.model, runtime.data, phys, runtime.state_spec)
    mujoco.mj_forward(runtime.model, runtime.data)
    base_pos = runtime.data.xpos[focus_body_id]
    if hasattr(viewer, "cam") and bool(getattr(args, "camera_follow_body", True)):
        viewer.cam.lookat[:] = (
            float(base_pos[0]),
            float(base_pos[1]),
            float(base_pos[2] + float(getattr(args, "camera_height_offset", 0.15))),
        )
    if (
        runtime.trace_distill_actions
        and advanced
        and int(getattr(session, "step_count", 0)) % runtime.trace_interval == 0
    ):
        _print_distill_action_trace(
            session,
            env=env,
            commander=runtime.commander,
            base_height=float(base_pos[2]),
            standing_teacher=runtime.trace_standing_teacher,
        )
    if runtime.overlay.enabled and args.show_reward_debug:
        _render_reward_debug_targets(
            viewer,
            session.info,
            runtime.overlay.selected_indices,
            marker_radius=args.target_marker_radius,
            marker_alpha=args.target_marker_alpha,
            show_axes=args.target_show_axes,
            axis_length=args.target_axis_length,
            show_vel=args.reward_debug_show_velocity,
            lin_vel_scale=args.reward_debug_lin_vel_scale,
            ang_vel_scale=args.reward_debug_ang_vel_scale,
            show_connectors=args.reward_debug_show_connectors,
            show_global_anchor=args.reward_debug_show_global_anchor,
        )
    elif runtime.overlay.enabled:
        _render_motion_targets(
            viewer,
            env.state.info.get("motion_data", None),
            runtime.overlay.selected_indices,
            marker_radius=args.target_marker_radius,
            marker_alpha=args.target_marker_alpha,
            show_axes=args.target_show_axes,
            axis_length=args.target_axis_length,
        )
    else:
        viewer.user_scn.ngeom = 0
    if runtime.render_velocity_arrows:
        _render_velocity_arrows(
            viewer,
            runtime.data,
            focus_body_id,
            env,
            height=_VELOCITY_ARROW_HEIGHT,
            scale=_VELOCITY_ARROW_SCALE,
            width=_VELOCITY_ARROW_WIDTH,
            lateral_offset=_VELOCITY_ARROW_LATERAL_OFFSET,
        )
    viewer.sync()


def _run_interactive_viewer_loop(runtime: _InteractiveViewerRuntime, args) -> None:
    print("[play_interactive] Opening viewer — close the window or press Esc to quit.")
    print("[play_interactive] Controls: Space=pause/resume, N=single-step, +/-=speed")
    if runtime.commander is not None or runtime.height_commander is not None:
        _print_keyboard_legend(args, height_control=runtime.height_commander is not None)
    with mujoco.viewer.launch_passive(
        runtime.model, runtime.data, key_callback=runtime.key_callback
    ) as viewer:
        focus_body_id = _resolve_focus_body_id(
            runtime.model, runtime.env, getattr(args, "camera_focus_body_name", "")
        )
        if hasattr(viewer, "cam"):
            viewer.cam.distance = (
                float(args.camera_distance)
                if getattr(args, "camera_distance", None) is not None
                else _default_viewer_camera_distance(
                    runtime.model,
                    runtime.env,
                    follow_body=bool(getattr(args, "camera_follow_body", True)),
                )
            )
            if getattr(args, "camera_elevation", None) is not None:
                viewer.cam.elevation = float(args.camera_elevation)
            if getattr(args, "camera_azimuth", None) is not None:
                viewer.cam.azimuth = float(args.camera_azimuth)
        with torch.inference_mode():
            while viewer.is_running():
                started = time.perf_counter()
                if runtime.commander is not None and runtime.env.state is not None:
                    _apply_playback_command(runtime.session, runtime.commander.command)
                _render_interactive_frame(viewer, runtime, args, focus_body_id=focus_body_id)
                remaining = runtime.controls.target_dt(runtime.ctrl_dt) - (
                    time.perf_counter() - started
                )
                if remaining > 0:
                    time.sleep(remaining)


def play_interactive(
    args,
    cfg: DictConfig | None = None,
    *,
    algo: str | None = None,
    fada_session_factory: Callable[..., Any] | None = None,
):
    device = _select_playback_device(cfg)
    print(f"[play_interactive] Device: {device}")
    algo = str(algo or getattr(args, "algo", "ppo"))
    available_backends = _available_backends_for_task(args.task)
    if available_backends and "mujoco" not in available_backends:
        print(
            "[play_interactive] Task does not support MuJoCo backend: "
            f"{args.task}. Available backends: {available_backends or ('<none>',)}. "
            "This script only supports MuJoCo viewer mode."
        )
        return
    try:
        session = _create_interactive_session(
            args,
            cfg,
            algo=algo,
            device=device,
            available_backends=available_backends,
            fada_session_factory=fada_session_factory,
        )
    except RuntimeError as exc:
        if str(exc) in {_PLAYBACK_ENV_UNAVAILABLE, _HORA_DISTILL_CHECKPOINT_UNAVAILABLE}:
            return
        raise
    if _uses_native_mujoco_viewer_launch() and not _can_launch_glfw_viewer():
        print(
            "[play_interactive] GLFW viewer initialization failed (no usable display). "
            "Set DISPLAY correctly, or run this command in a desktop session."
        )
        return
    try:
        runtime = _prepare_viewer_runtime(
            args, cfg, algo=algo, device=device, playback_session=session[0]
        )
    except RuntimeError as exc:
        if str(exc).startswith("interactive.keyboard unavailable:"):
            print(f"[play_interactive] {exc}")
            return
        raise
    _run_interactive_viewer_loop(runtime, args)
    print("[play_interactive] Done.")
