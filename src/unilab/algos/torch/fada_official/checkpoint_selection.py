"""Metadata-based adaptation of FADA offline_data_utils at 290b3a9c5c0e.

Selection uses completed-update identities, never infers rewards from filenames.
"""
import math
from pathlib import Path
import warnings

import numpy as np
import torch
from omegaconf import OmegaConf

from .checkpoint import make_selection_record, normalize_identity, read_oracle_checkpoint, validate_same_run
from .metrics import REWARD_STATISTIC


def sample_steps_logspace(steps, n_target):
    steps = sorted(steps)
    if n_target <= 0:
        raise ValueError("n_target must be positive")
    if n_target >= len(steps):
        return steps
    indices = {max(0, min(len(steps) - 1, int(round(v - 1))))
               for v in np.geomspace(1, len(steps), num=n_target)}
    indices.add(len(steps) - 1)
    sampled = [steps[i] for i in sorted(indices)]
    while len(sampled) > n_target:
        sampled.pop(-2)
    while len(sampled) < n_target:
        candidates = [s for s in steps if s not in sampled]
        sampled.append(max(candidates, key=lambda candidate: min(abs(candidate - s) for s in sampled)))
        sampled.sort()
    return sampled


def sample_steps_reward_aware(steps, n_target, reward_map):
    steps = sorted(steps)
    if not steps or not reward_map:
        return []
    chosen = set(sample_steps_logspace(steps, n_target))
    rewards = np.asarray(list(reward_map.values()), np.float64)
    for q in (.50, .70, .85, .95, 1.00):
        threshold = float(np.quantile(rewards, q))
        reached = [s for s, r in sorted(reward_map.items()) if r >= threshold]
        if reached:
            chosen.add(min(steps, key=lambda s: abs(s - reached[0])))
    sampled = sorted(chosen)
    return sample_steps_logspace(sampled, n_target) if len(sampled) > n_target else sampled


def read_reward_curve(path, identity):
    curve = torch.load(Path(path), map_location="cpu", weights_only=True)
    if (not isinstance(curve, dict) or curve.get("schema") != "fada_official_reward_curve_v1"
            or curve.get("statistic") != REWARD_STATISTIC or curve.get("window") != 100):
        raise ValueError("Official reward curve schema/statistic mismatch")
    validate_same_run(normalize_identity(curve.get("identity")), normalize_identity(identity))
    seen = set()
    for point in curve["points"]:
        step, value = point["iteration"], point["value"]
        if (type(step) is not int or step < 1 or step in seen
                or (value is not None and not math.isfinite(float(value)))):
            raise ValueError("Official reward curve contains invalid/duplicate iterations")
        seen.add(step)
    for item in curve["checkpoints"]:
        validate_same_run(normalize_identity(item["identity"]), normalize_identity(identity))
    return curve


def select_source_checkpoints(final_path, *, curve_path=None, sampling_mode="reward_aware"):
    final_path = Path(final_path).resolve()
    final = read_oracle_checkpoint(final_path)["official_oracle"]
    curve = read_reward_curve(curve_path or final_path.parent / "reward_curve.pt", final)
    records = [p for p in curve["checkpoints"] if p["identity"]["role"] == "intermediate"
               and p["identity"]["completed_iterations"] < final["completed_iterations"]]
    steps = [p["identity"]["completed_iterations"] for p in records]
    if len(steps) < 20 or len(set(steps)) != len(steps):
        raise ValueError("Official source selection requires at least 20 unique intermediate checkpoints")
    reward_map = {p["iteration"]: float(p["value"]) for p in curve["points"]
                  if p["value"] is not None and p["iteration"] <= final["completed_iterations"]}
    if sampling_mode not in {"reward_aware", "logspace"}:
        raise ValueError("Unsupported Official checkpoint sampling mode")
    actual_mode = sampling_mode
    if sampling_mode == "reward_aware" and not reward_map:
        warnings.warn("Official episode rewards are missing; falling back to logspace checkpoint selection",
                      RuntimeWarning, stacklevel=2)
        actual_mode = "logspace"
    chosen = (sample_steps_reward_aware(steps, 20, reward_map) if actual_mode == "reward_aware"
              else sample_steps_logspace(steps, 20))
    by_step = {p["identity"]["completed_iterations"]: p for p in records}
    selected = [by_step[s] for s in chosen]
    paths = [Path(p["path"]).resolve() for p in selected]
    if len(set(paths)) != 20 or any(not p.is_file() or p == final_path for p in paths):
        raise ValueError("Official selected checkpoints are missing, repeated or include final")
    record = make_selection_record(final, [p["identity"] for p in selected], sampling_mode=actual_mode)
    record["paths"] = [str(p) for p in paths]
    record["curve_path"] = str(Path(curve_path or final_path.parent / "reward_curve.pt").resolve())
    return paths, record


def configure_source_selection(cfg, final_path):
    """Composition seam called before the existing exact source-budget owner."""
    if not cfg.training.fada.intermediate_oracle_checkpoint_paths:
        paths, record = select_source_checkpoints(final_path,
            curve_path=OmegaConf.select(cfg, "training.fada.oracle_reward_curve_path"),
            sampling_mode=str(cfg.training.fada.oracle_checkpoint_sampling_mode))
        OmegaConf.update(cfg, "training.fada.intermediate_oracle_checkpoint_paths", [str(p) for p in paths])
        OmegaConf.update(cfg, "training.fada.oracle_selection", record, force_add=True)
