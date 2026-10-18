"""APPO-only source selection from sealed checkpoint metadata, not PPO curves."""
from pathlib import Path
import warnings

from omegaconf import OmegaConf

from unilab.algos.torch.fada_official.checkpoint_selection import (
    sample_steps_logspace, sample_steps_reward_aware,
)
from .checkpoint import read_oracle_checkpoint, validate_lineage, validate_same_lineage


def _record(path):
    path = Path(path).resolve()
    payload = read_oracle_checkpoint(path)
    return {"path": str(path), "identity": payload["official_appo_oracle"], "cursor": payload["cursor"]}


def select_source_checkpoints(final_path, *, intermediate_paths=None, sampling_mode="reward_aware"):
    final = _record(final_path)
    identity = final["identity"]
    if identity["role"] != "final":
        raise ValueError("FADA APPO source requires a final checkpoint")
    if sampling_mode not in {"reward_aware", "logspace"}:
        raise ValueError("Unsupported APPO checkpoint sampling mode")
    explicit = bool(intermediate_paths)
    paths = (list(intermediate_paths) if explicit else
             sorted(Path(final["path"]).parent.glob("appo_update_*.pt")))
    if len({str(Path(p).resolve()) for p in paths}) != len(paths):
        raise ValueError("FADA APPO intermediate paths are repeated")
    candidates = []
    for path in paths:
        record = _record(path)
        validate_same_lineage(identity, record["identity"])
        if (not explicit and record["identity"]["role"] == "intermediate"
                and record["identity"]["completed_updates"] >= identity["completed_updates"]):
            continue  # A periodic save may share the final update, but is not an earlier teacher.
        candidates.append(record)
    candidates.sort(key=lambda record: record["identity"]["completed_updates"])
    steps = [record["identity"]["completed_updates"] for record in candidates]
    if len(candidates) < 20 or len(set(steps)) != len(steps) or (explicit and len(candidates) != 20):
        raise ValueError("FADA APPO source requires 20 unique intermediate checkpoints before final")
    previous_samples = -1
    for record in candidates:
        step, samples = record["identity"]["completed_updates"], record["cursor"]["unique_used_samples"]
        if (record["identity"]["role"] != "intermediate" or step >= identity["completed_updates"]
                or samples <= previous_samples or samples >= final["cursor"]["unique_used_samples"]):
            raise ValueError("FADA APPO intermediate role/update/sample progress mismatch")
        previous_samples = samples
    rewards = {record["identity"]["completed_updates"]: record["identity"]["reward"]["value"]
               for record in candidates if record["identity"]["reward"]["value"] is not None}
    mode = "explicit" if explicit else sampling_mode
    if not explicit and mode == "reward_aware" and not rewards:
        warnings.warn("APPO raw episode rewards are missing; falling back to logspace checkpoint selection",
                      RuntimeWarning, stacklevel=2)
        mode = "logspace"
    chosen = (steps if explicit else sample_steps_reward_aware(steps, 20, rewards)
              if mode == "reward_aware" else sample_steps_logspace(steps, 20))
    selected = [record for record in candidates if record["identity"]["completed_updates"] in chosen]
    validate_lineage(identity, [record["identity"] for record in selected])
    selection = {"sampling_mode": mode, "final": identity, "final_cursor": final["cursor"],
        "intermediates": [record["identity"] for record in selected],
        "cursors": [record["cursor"] for record in selected],
        "paths": [record["path"] for record in selected]}
    return [Path(record["path"]) for record in selected], selection


def configure_source_selection(cfg, final_path):
    root = Path(__file__).resolve().parents[5]
    configured = [Path(str(p)) for p in cfg.training.fada.intermediate_oracle_checkpoint_paths]
    paths, record = select_source_checkpoints(final_path,
        intermediate_paths=[p if p.is_absolute() else root / p for p in configured],
        sampling_mode=str(cfg.training.fada.oracle_checkpoint_sampling_mode))
    OmegaConf.update(cfg, "training.fada.intermediate_oracle_checkpoint_paths", [str(p) for p in paths])
    OmegaConf.update(cfg, "training.fada.oracle_selection", record, force_add=True)
