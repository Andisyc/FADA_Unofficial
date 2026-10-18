"""Explicit sealed-APPO playback assembly, separate from training resume."""
from copy import deepcopy
from pathlib import Path

from omegaconf import OmegaConf


def resolve_play_checkpoint(cfg, *, root_dir, playback_cfg=None):
    candidates = [getattr(playback_cfg, "checkpoint_path", None),
        OmegaConf.select(cfg, "training.play_checkpoint_path"),
        getattr(playback_cfg, "checkpoint", None), OmegaConf.select(cfg, "algo.checkpoint")]
    value = next((v for v in candidates if v is not None and str(v) not in {"", "-1", "None", "null"}), None)
    if value is None:
        raise FileNotFoundError("FADA APPO playback requires an explicit training.play_checkpoint_path")
    path = Path(str(value))
    path = (path if path.is_absolute() else Path(root_dir) / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"FADA APPO playback checkpoint does not exist: {path}")
    return path


def play_checkpoint(cfg, rl_cfg, *, root_dir=None, resolve_checkpoint_path=None):
    """Use the same session/actor for play_only and post-training playback."""
    from unilab.training import create_env
    from unilab.training.backend_adapter import BackendAdapter
    from unilab.training.rsl_rl import RslRlVecEnvWrapper
    from unilab.visualization.playback_policy_sessions import create_official_appo_playback_session
    from unilab.visualization.playback_sessions import RslRlPlaybackConfig

    root_dir = Path(root_dir) if root_dir is not None else Path(__file__).resolve().parents[5]
    path = resolve_play_checkpoint(cfg, root_dir=root_dir)
    play_cfg = deepcopy(cfg)
    play_cfg.training.play_only = True
    adapter = BackendAdapter(play_cfg, root_dir=root_dir, algo_name="appo")
    playback_cfg = RslRlPlaybackConfig(task=str(cfg.training.task_name), load_run="-1", checkpoint=str(path),
        action_mode="policy", policy_obs_mode="actor", algo_log_name=str(cfg.algo.algo_log_name),
        log_root=None, num_envs=int(cfg.training.play_env_num))
    session, _, _ = create_official_appo_playback_session(playback_cfg=playback_cfg, cfg=play_cfg,
        rl_cfg=rl_cfg, root_dir=root_dir, device=cfg.training.device, wrapper_cls=RslRlVecEnvWrapper,
        env_factory=lambda n: create_env(play_cfg, num_envs=n, env_cfg_override=adapter.build_play_env_cfg_override()))
    try:
        return session.env.run_playback_mode(play_render_mode=cfg.training.play_render_mode,
            play_steps=cfg.training.play_steps, output_video=path.parent / "play_video.mp4",
            render_spacing=float(OmegaConf.select(cfg, "training.render_spacing",
                default=getattr(session.env.cfg, "render_spacing", 1.0))),
            render_offset_mode=str(getattr(session.env.cfg, "render_offset_mode", "grid")),
            initialize=session.reset, step=lambda _: session.step_once())
    finally:
        session.env.close()
