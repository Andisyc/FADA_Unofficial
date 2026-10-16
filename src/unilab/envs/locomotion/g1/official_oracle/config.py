"""Independent MuJoCo-flat adaptation of FADA's g1_29dof_oracle.

Source: LeCAR-Lab/FADA@290b3a9c5c0ebe2e9851b61350617544be05e9bc,
config_values/robot.py and config_values/loco/g1. Not a legacy SAC schema.
"""
from dataclasses import dataclass, field

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg
from unilab.envs.locomotion.g1.base import G1BaseCfg

SOURCE_COMMIT = "290b3a9c5c0ebe2e9851b61350617544be05e9bc"
OBS_SCHEMA = "g1_official_oracle_flat_v1"
STUDENT_SCHEMA = "g1_official_oracle_student66_v1"
JOINT_NAMES = tuple(["left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint","left_knee_joint","left_ankle_pitch_joint","left_ankle_roll_joint","right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint","right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint","waist_yaw_joint","waist_roll_joint","waist_pitch_joint","left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint","left_elbow_joint","left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint","right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint","right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint"])
LOWER = np.array([-2.5307,-0.5236,-2.7576,-0.087267,-0.87267,-0.2618,-2.5307,-2.9671,-2.7576,-0.087267,-0.87267,-0.2618,-2.618,-0.52,-0.52,-3.0892,-1.5882,-2.618,-1.0472,-1.972222054,-1.61443,-1.61443,-3.0892,-2.2515,-2.618,-1.0472,-1.972222054,-1.61443,-1.61443], dtype=np.float32)
UPPER = np.array([2.8798,2.9671,2.7576,2.8798,0.5236,0.2618,2.8798,0.5236,2.7576,2.8798,0.5236,0.2618,2.618,0.52,0.52,2.6704,2.2515,2.618,2.0944,1.972222054,1.61443,1.61443,2.6704,1.5882,2.618,2.0944,1.972222054,1.61443,1.61443], dtype=np.float32)
EFFORT = np.array([88,139,88,139,50,50,88,139,88,139,50,50,88,50,50,25,25,25,25,25,5,5,25,25,25,25,25,5,5], dtype=np.float32)
ARMATURE = np.array([0.01017752,0.025101925,0.01017752,0.025101925,0.00721945,0.00721945,0.01017752,0.025101925,0.01017752,0.025101925,0.00721945,0.00721945,0.01017752,0.00721945,0.00721945,0.003609725,0.003609725,0.003609725,0.003609725,0.003609725,0.00425,0.00425,0.003609725,0.003609725,0.003609725,0.003609725,0.003609725,0.00425,0.00425], dtype=np.float32)

DEFAULT_ANGLES = np.array(
    [-.312, 0, 0, .669, -.363, 0] * 2 + [0, 0, 0]
    + [.2, .2, 0, .6, 0, 0, 0] + [.2, -.2, 0, .6, 0, 0, 0],
    dtype=np.float32,
)
KP = np.array(
    [40.179238471, 99.098427777, 40.179238471, 99.098427777, 28.501246196, 28.501246196] * 2
    + [40.179238471, 28.501246196, 28.501246196]
    + [14.250623098] * 5 + [16.778327481] * 2
    + [14.250623098] * 5 + [16.778327481] * 2, dtype=np.float32,
)
KD = np.array(
    [2.557889765, 6.308801854, 2.557889765, 6.308801854, 1.814445687, 1.814445687] * 2
    + [2.557889765, 1.814445687, 1.814445687]
    + [.907222843] * 5 + [1.068141502] * 2
    + [.907222843] * 5 + [1.068141502] * 2, dtype=np.float32,
)
POSE_WEIGHTS = np.array([.01, 1, 5, .01, 5, 5] * 2 + [50] * 17, dtype=np.float32)
MASS_BODIES = (
    "pelvis", "left_hip_yaw_link", "left_hip_roll_link", "left_hip_pitch_link",
    "left_knee_link", "right_hip_yaw_link", "right_hip_roll_link",
    "right_hip_pitch_link", "right_knee_link", "torso_link",
)
REWARD_NAMES = (
    "track_xy", "track_yaw", "alive", "lin_z", "ang_xy", "joint_vel", "joint_acc",
    "action_rate", "pos_limits", "energy", "pose", "orientation", "world_base_height",
    "gait", "slide", "clearance", "feet_ori", "close_feet", "undesired",
)


def build_environment_contract(cfg):
    """Cold-path semantic identity, independent of learner and playback packages."""
    import hashlib
    import json
    import xml.etree.ElementTree as ET
    from dataclasses import asdict
    from pathlib import Path

    def plain(value):
        if isinstance(value, dict):
            return {str(k): plain(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [plain(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        return value

    def digest(value):
        return hashlib.sha256(json.dumps(plain(value), sort_keys=True, allow_nan=False).encode()).hexdigest()

    scene = Path(cfg.scene.model_file)
    robot = scene.with_name("robot.xml")
    root = ET.parse(robot).getroot()
    compiler = root.find("compiler")
    meshdir = robot.parent / (compiler.get("meshdir", "") if compiler is not None else "")
    files = [scene, robot] + [meshdir / m.attrib["file"] for m in root.findall(".//asset/mesh")]
    assets = [(p.name, hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
    effective = asdict(cfg)
    for name in ("evaluating", "randomization_enabled", "mujoco_num_threads"):
        effective.pop(name, None)
    effective["scene"] = {"asset_sha256": digest(assets)}
    return {"task_name": "G1OfficialOracleFlat", "backend": "mujoco",
        "source_commit": SOURCE_COMMIT, "observation_schema": OBS_SCHEMA,
        "student_schema": STUDENT_SCHEMA, "action_schema": "g1_official_raw_pd025_v1",
        "dimensions": [322, 322, 66, 29, 3], "joint_names": list(JOINT_NAMES),
        "asset_sha256": digest(assets), "config_sha256": digest(effective)}


@dataclass
class OfficialOracleRewardConfig:
    scales: dict[str, float] = field(default_factory=dict)


@registry.envcfg("G1OfficialOracleFlat")
@dataclass
class G1OfficialOracleFlatCfg(G1BaseCfg):
    scene: SceneCfg = field(default_factory=lambda: SceneCfg(
        model_file=str(ASSETS_ROOT_PATH / "robots/g1/official_oracle/scene_flat.xml")
    ))
    reward_config: OfficialOracleRewardConfig | None = None
    sim_dt: float = .005
    ctrl_dt: float = .02
    max_episode_seconds: float = 20.
    post_step_forward_sensor: bool = True
    evaluating: bool = False
    randomization_enabled: bool = True

    def validate(self):
        super().validate()
        if self.sim_dt != .005 or self.ctrl_dt != .02:
            raise ValueError("Official Oracle requires 5ms physics / 20ms control")
        canonical = ASSETS_ROOT_PATH / "robots/g1/official_oracle/scene_flat.xml"
        from pathlib import Path
        if (self.scene is None or Path(self.scene.model_file).resolve() != Path(canonical).resolve()
                or self.scene.fragment_files or self.scene.terrain is not None):
            raise ValueError("Official Oracle v1 supports only its independent flat scene")
        if not self.post_step_forward_sensor or self.noise_config.level != 0:
            raise ValueError("Official Oracle requires aligned noiseless observations")
        if self.control_config.action_scale != .25 or self.control_config.simulate_action_latency:
            raise ValueError("Official Oracle owns its action scaling and per-env delay")
        if self.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra")
        scales = self.reward_config.scales
        if set(scales) != set(REWARD_NAMES) or not np.isfinite(list(scales.values())).all():
            raise ValueError("Official Oracle requires exactly 19 finite reward scales")

