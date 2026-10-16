"""Student projection from the same environment frame as the teacher label."""
import numpy as np

from unilab.algos.torch.distill.collection.common import project_student_obs

OFFICIAL_STUDENT_CONTRACT = "g1_official_oracle_student66_v1"


def student_view(source, info, *, projection, expected_student_obs_dim, student_drop_index=None):
    if projection != OFFICIAL_STUDENT_CONTRACT:
        return project_student_obs(source, projection=projection,
            expected_student_obs_dim=expected_student_obs_dim, student_drop_index=student_drop_index)
    if student_drop_index is not None or expected_student_obs_dim != 66:
        raise ValueError("Official student view requires its independent 66-D contract")
    if info.get("student_observation_contract") != OFFICIAL_STUDENT_CONTRACT:
        raise ValueError("Official student observation identity is missing")
    value = np.asarray(info.get("student_observation"), dtype=np.float32)
    if value.ndim != 2 or source.shape != (value.shape[0], 322):
        raise ValueError("Official teacher/student frame row mismatch")
    if value.shape != (source.shape[0], 66) or not np.isfinite(value).all():
        raise ValueError("Official student observation must be finite (N,66)")
    return value.copy()
