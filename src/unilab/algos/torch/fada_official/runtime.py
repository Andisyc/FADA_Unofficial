"""Owner-selected runtime; invalid official combinations never fall back to PPO."""
from unilab.algos.torch.rsl_rl_runtime import RslRlPPORuntime
from unilab.training.rsl_rl import RslRlVecEnvWrapper


def resolve_runtime(rl_cfg):
    c = rl_cfg.get("collection", {})
    if (rl_cfg.get("algorithm", {}).get("class_name") !=
            "unilab.algos.torch.fada_official_ppo:OfficialFadaPPO"
            or c.get("mode") != "synchronous_heterogeneous"
            or c.get("collector_device") != "cpu" or c.get("num_slots") != 1
            or [c.get(k) for k in ("obs_dim", "critic_dim", "action_dim")] != [322, 322, 29]
            or rl_cfg.get("num_steps_per_env") != 24
            or rl_cfg.get("obs_groups", {}).get("actor") != ["actor"]
            or rl_cfg.get("obs_groups", {}).get("critic") != ["critic"]):
        raise ValueError("Official PPO requires teacher322/critic322/action29, CPU synchronous collection and T=24")
    from unilab.training.fada_official import run_training, play_training_checkpoint
    from unilab.visualization.playback_policy_sessions import create_official_oracle_playback_session
    return RslRlPPORuntime(wrapper_cls=RslRlVecEnvWrapper, train=run_training,
        play=play_training_checkpoint, playback_factory=create_official_oracle_playback_session)
