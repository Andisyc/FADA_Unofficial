"""CPU headers are identities, not float32 staging inputs or update counters."""
import numpy as np

HEADERS = ("rollout_id", "policy_version", "valid_steps")


def split_rollout(payload, *, previous_rollout_id, published_version, num_steps):
    metadata = {}
    for name in HEADERS:
        value = payload[name]
        if value.shape != (1,) or value.dtype != np.int64:
            raise ValueError(f"APPO {name} must be an int64 scalar header")
        metadata[name] = int(value[0])
    if metadata["rollout_id"] != previous_rollout_id + 1:
        raise ValueError("APPO rollout identity is duplicated, skipped or out of order")
    if not 1 <= metadata["policy_version"] <= published_version:
        raise ValueError("APPO behavior version is unpublished or invalid")
    if metadata["valid_steps"] != num_steps:
        raise ValueError("APPO rollout is incomplete")
    return {name: value for name, value in payload.items() if name not in HEADERS}, metadata
