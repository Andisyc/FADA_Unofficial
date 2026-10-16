"""Offline contract proof. No MjModel, real simulator, learner, or renderer."""
import copy
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import yaml

from unilab.base import registry
from unilab.base.backend.mujoco.backend import MuJoCoBackend
from unilab.envs.locomotion.g1.official_oracle import env as owner
from unilab.envs.locomotion.g1.official_oracle.actuation import ActuationState
from unilab.envs.locomotion.g1.official_oracle.command import CommandState
from unilab.envs.locomotion.g1.official_oracle.config import (
    ARMATURE, DEFAULT_ANGLES, EFFORT, JOINT_NAMES, KD, KP, LOWER, REWARD_NAMES, UPPER,
    G1OfficialOracleFlatCfg, OfficialOracleRewardConfig,
)
from unilab.envs.locomotion.g1.official_oracle.observation import (
    PhysicalFrame, build_student, build_teacher,
)
from unilab.envs.locomotion.g1.official_oracle.reward import reward_terms, total_reward

ROOT = Path(__file__).resolve().parents[4]
TASK = ROOT / "conf/ppo/task/g1_official_oracle_flat/mujoco.yaml"
SCALES = yaml.safe_load(TASK.read_text())["reward"]["scales"]


class FakePool:
    """Identifiable numeric transitions, not a physics model."""
    def __init__(self, backend):
        self.backend = backend
        self.calls = []
        self.resets = []
        self.contact_clock = False

    def step(self, state, *, nstep, control, **kwargs):
        self.calls.append((nstep, control.copy()))
        out = state.copy()
        out[:, 37 + 6:] += control[:, 0, :29] * .005 * nstep
        sensor = self.backend._sensor_data.copy()
        if self.contact_clock:
            sensor[:, :len(self.backend.names[1:]) * 3] = len(self.calls)
        return out, sensor

    def reset(self, *, env_ids, initial_state, randomization):
        self.resets.append((env_ids.copy(), copy.deepcopy(randomization)))
        sensor = np.zeros((len(env_ids), self.backend._sensor_data.shape[1]))
        sensor[:, -1] = 1
        return initial_state.copy(), sensor


class FakeBackend(MuJoCoBackend):
    """Exercise real backend batching/history/reset/snapshot, replacing only native pool."""
    def __init__(self, n=2):
        cfg = G1OfficialOracleFlatCfg()
        robot = ET.parse(Path(cfg.scene.model_file).with_name("robot.xml")).getroot()
        self.names = ("world",) + tuple(b.get("name") for b in robot.findall(".//worldbody//body"))
        self._num_envs, self._np_dtype, self._sim_dt = n, np.float64, .005
        self._model = SimpleNamespace(nbody=len(self.names), ngeom=4, nq=36, nv=35, nu=29)
        self.nq, self.nv = 36, 35
        self._physics_state = np.zeros((n, 72))
        self._physics_state[:, 1:37] = self.get_keyframe_qpos("stand")
        self._idx_qpos, self._idx_qvel = 1, 37
        self._root_qpos_dim, self._root_qvel_dim = 7, 6
        self._base_body_id = self.names.index("pelvis")
        self._push_body_id = self._base_body_id
        self._base_lin_vel_view = self._physics_state[:, 37:40]
        self._pending_xfrc_applied = np.zeros((n, len(self.names) * 6))
        self._pre_step_control_fn = None
        self._isolated_rollout_pool = None
        self._isolated_rollout_active = False
        self._isolated_rollout_failure = None
        self._reset_transaction_count = 0
        self._post_step_forward_sensor = True
        self._sensor_data = np.zeros((n, (len(self.names) - 1) * 3 + 6))
        self._sensor_data[:, -1] = 1
        self._sensor_views = {f"oracle_contact_{name}": self._sensor_data[:, i * 3:i * 3 + 3]
                              for i, name in enumerate(self.names[1:])}
        self._sensor_views["torso_gyro"] = self._sensor_data[:, -6:-3]
        self._sensor_views["torso_upvector"] = self._sensor_data[:, -3:]
        self._pool = FakePool(self)
        self.closed = False

    def get_actuated_joint_names(self): return JOINT_NAMES
    def get_actuator_force_range(self): return np.stack((-EFFORT, EFFORT), axis=1)
    def get_keyframe_qpos(self, name): return np.r_[0, 0, .8, 1, 0, 0, 0, DEFAULT_ANGLES]
    def get_init_qvel(self): return np.zeros(35)
    def get_body_names(self): return self.names
    def get_body_ids(self, names): return np.array([self.names.index(name) for name in names])
    def get_body_mass(self): return np.r_[0, np.full(len(self.names) - 1, 5.)]
    def get_body_ipos(self): return np.zeros((len(self.names), 3))
    def get_body_inertia(self): return np.full((len(self.names), 3), .1)
    def get_geom_friction(self): return np.tile([1., .005, .0001], (4, 1))
    def get_geom_contact_masks(self): return np.array([0, 1, 1, 0]), np.array([1, 0, 0, 0])
    def get_base_pos(self): return self._physics_state[:, 1:4]
    def get_base_quat(self): return self._physics_state[:, 4:8]
    def get_base_lin_vel(self): return self._base_lin_vel_view
    def get_dof_pos(self): return self._physics_state[:, 8:37]
    def get_dof_vel(self): return self._physics_state[:, 43:72]
    def get_body_ang_vel_w(self, ids): return np.zeros((self._num_envs, len(ids), 3))
    def get_body_state_w(self, ids):
        pos = np.zeros((self._num_envs, len(ids), 3))
        for k, body in enumerate(ids):
            pos[:, k] = [0, .15 if "left" in self.names[body] else -.15, .05]
        quat = np.zeros((self._num_envs, len(ids), 4)); quat[..., 0] = 1
        vel = np.zeros_like(pos)
        return pos, quat, vel, vel.copy()
    def materialize(self): pass
    def close(self): self.closed = True


@pytest.fixture
def environment(monkeypatch):
    backend = FakeBackend()
    monkeypatch.setattr(owner, "create_backend", lambda *args, **kwargs: backend)
    cfg = G1OfficialOracleFlatCfg(reward_config=OfficialOracleRewardConfig(SCALES),
                                 randomization_enabled=False, evaluating=True)
    env = owner.G1OfficialOracleFlatEnv(cfg, num_envs=2)
    yield env
    env.close()


def fixed_frame():
    n = 1
    z = lambda d: np.zeros((n, d), np.float32)
    q = DEFAULT_ANGLES[None].copy()
    quat = np.array([[1., 0, 0, 0]], np.float32)
    foot_pos = np.array([[[0, .15, .05], [0, -.15, .05]]], np.float32)
    force = np.array([[[0, 0, 10], [0, 0, 10], [0, 0, 0]]], np.float32)
    velocity = np.array([[1., 0, 0]], np.float32)
    return PhysicalFrame(q, z(29), np.array([[0, 0, .78]], np.float32), quat,
                         velocity, z(3), foot_pos, np.repeat(quat[:, None], 2, axis=1),
                         np.repeat(velocity[:, None], 2, axis=1), foot_pos.copy(),
                         np.repeat(velocity[:, None], 2, axis=1), force,
                         np.repeat(force[:, None], 3, axis=1), z(3), np.array([[0, 0, 1.]], np.float32))


def test_asset_config_and_registration_static_only():
    cfg = G1OfficialOracleFlatCfg(reward_config=OfficialOracleRewardConfig(SCALES))
    cfg.validate()
    assert registry.contains("G1OfficialOracleFlat")
    asset = Path(cfg.scene.model_file)
    robot = ET.parse(asset.with_name("robot.xml")).getroot()
    scene = ET.parse(asset).getroot()
    assert robot.find("keyframe") is None and robot.find("tendon") is None
    assert robot.find("contact") is None
    assert len(scene.findall("sensor/contact")) == 32
    assert float(scene.find("option").get("timestep")) == .005
    qpos = np.fromstring(scene.find("keyframe/key").get("qpos"), sep=" ")
    assert qpos[2] == .8
    np.testing.assert_allclose(qpos[7:], DEFAULT_ANGLES)
    motors = robot.findall("actuator/motor")
    assert tuple(m.get("joint") for m in motors) == JOINT_NAMES
    for i, motor in enumerate(motors):
        np.testing.assert_equal(np.fromstring(motor.get("forcerange"), sep=" "), [-EFFORT[i], EFFORT[i]])
        joint = robot.find(f'.//worldbody//joint[@name="{JOINT_NAMES[i]}"]')
        np.testing.assert_allclose(np.fromstring(joint.get("range"), sep=" "), [LOWER[i], UPPER[i]])
        assert float(joint.get("armature")) == pytest.approx(ARMATURE[i])
        assert float(joint.get("frictionloss")) == 0
    meshdir = asset.parent / robot.find("compiler").get("meshdir")
    assert all((meshdir / mesh.get("file")).is_file() for mesh in robot.findall("asset/mesh"))
    assert set(SCALES) == set(REWARD_NAMES) and len(SCALES) == 19


def test_teacher_layout_and_student_privilege_boundary():
    frame, act, command = fixed_frame(), ActuationState.create(1, False), CommandState.create(1)
    command.velocity[:] = [1, 0, 0]; command.phase[:] = [0, np.pi]
    act.previous[:] = .1; act.executed[:] = .2; act.delay[:] = 1
    act.torque[:] = EFFORT * .3
    dr = np.arange(18, dtype=np.float32)[None]
    obs = build_teacher(frame, act, command, dr, [0, 1])
    assert obs.shape == (1, 322) and obs.dtype == np.float32
    np.testing.assert_allclose(obs[0, :6], [0, 0, 0, 2, 0, 0])
    np.testing.assert_equal(obs[0, 93:96], [0, 0, -1])
    np.testing.assert_allclose(obs[0, 96:125], .1)
    np.testing.assert_allclose(obs[0, 125:154], .2)
    assert obs[0, 154] == 1
    np.testing.assert_allclose(obs[0, 155:184], .3)
    np.testing.assert_allclose(obs[0, 184:271], 1)
    np.testing.assert_allclose(obs[0, 271:281], [1, .1, 0, 0, .1, 0, 0, .1, 1, 1])
    np.testing.assert_allclose(obs[0, 281:294], [.78, .05, .05] + [0] * 10)
    np.testing.assert_allclose(obs[0, 294:304], [1, 0, 0, 0, 0, 1, -1, 1, 2*np.pi*.02, 0], atol=2e-7)
    np.testing.assert_equal(obs[:, 304:], dr)
    student = build_student(frame, act, command)
    frame.root_vel[:] = 42; frame.forces[:] = 123; dr[:] = -10
    np.testing.assert_equal(build_student(frame, act, command), student)
    assert student.shape == (1, 66)
    np.testing.assert_allclose(student[0, :6], [0, 0, 0, 0, 0, -1])


def test_root_and_yaw_frames_not_world_sensor_aliases():
    frame = fixed_frame()
    frame.root_quat[:] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
    frame.root_vel[:] = [0, 1, .3]
    np.testing.assert_allclose(frame.lin_body, [[1, 0, .3]], atol=2e-7)
    np.testing.assert_allclose(frame.lin_yaw, [[1, 0]], atol=2e-7)
    frame.root_quat[:] = [np.cos(.3), np.sin(.3), 0, 0]
    np.testing.assert_allclose(frame.gravity, [[0, -np.sin(.6), -np.cos(.6)]], atol=2e-7)


def test_action_clip_delay_and_explicit_pd():
    act = ActuationState.create(2, False)
    act.delay[:] = [0, 1]
    act.process(np.full((2, 29), 200.))
    np.testing.assert_allclose(act.target[0], DEFAULT_ANGLES + 25)
    np.testing.assert_allclose(act.target[1], DEFAULT_ANGLES)
    act.process(np.full((2, 29), .4))
    np.testing.assert_allclose(act.target[0], DEFAULT_ANGLES + .1)
    np.testing.assert_allclose(act.target[1], DEFAULT_ANGLES + 25)
    dq = np.full((2, 29), 2.)
    tau = act.pd(np.tile(DEFAULT_ANGLES, (2, 1)), dq, False)
    np.testing.assert_allclose(tau[0], np.clip(KP * .1 - KD * 2, -EFFORT, EFFORT), atol=4e-6)
    np.testing.assert_allclose(tau[1], EFFORT)


def test_all_19_reward_terms_and_dt_once():
    frame, act, command = fixed_frame(), ActuationState.create(1, False), CommandState.create(1)
    command.velocity[:] = [1, 0, 0]; command.phase[:] = [0, np.pi]
    terms = reward_terms(frame, act, command, np.zeros((1, 29)), [0, 1], [2])
    expected = {name: 0. for name in REWARD_NAMES}
    expected.update(track_xy=1, track_yaw=1, alive=1, gait=2, clearance=1)
    for name, value in expected.items():
        np.testing.assert_allclose(terms[name], value, atol=1e-10, err_msg=name)
    np.testing.assert_allclose(total_reward(terms, SCALES), [.183], atol=1e-7)
    frame.dq[0, 0] = 2; act.torque[0, 0] = 3; act.action[0, 0] = .2
    frame.q[0, 0] += .1
    frame.forces[0, 2, 0] = 2; frame.history[0, 1, 2, 0] = 2
    terms = reward_terms(frame, act, command, np.zeros((1, 29)), [0, 1], [2])
    for name, value in {"joint_vel": 4, "joint_acc": 10000, "energy": 6,
                        "action_rate": .04, "pose": .0001, "undesired": 1}.items():
        np.testing.assert_allclose(terms[name], value, rtol=2e-6, err_msg=name)


def test_gait_standing_resume_and_resample_keep_offsets():
    command = CommandState.create(2)
    command.reset(np.arange(2), evaluating=True)
    command.advance(np.array([10, 10]), evaluating=True)
    np.testing.assert_allclose(command.phase, np.pi)
    command.velocity[:] = [1, 0, 0]
    command.advance(np.array([11, 11]), evaluating=True)
    expected = (11 * 2 * np.pi * .02 + np.array([0, -np.pi]) + np.pi) % (2*np.pi) - np.pi
    np.testing.assert_allclose(command.phase[0], expected, atol=4e-7)
    offsets = command.offset.copy()
    command.advance(np.array([500, 500]))
    np.testing.assert_equal(command.offset, offsets)


def test_four_substeps_contact_history_partial_reset_snapshot(environment):
    env = environment
    env.init_state()
    backend = env._backend
    backend._pool.contact_clock = True
    backend.set_pre_step_control(None)  # contact opt-in alone must still use physics samples
    backend.step(np.zeros((2, 29)), 4)
    assert [n for n, _ in backend._pool.calls] == [1, 1, 1, 1]
    history = backend.get_body_contact_force_history_w(env.body_ids)
    np.testing.assert_equal(history[0, :, 0, 0], [4, 3, 2])
    snap = backend.capture_rollout_state()
    env.reset(np.array([0]))
    history = backend.get_body_contact_force_history_w(env.body_ids)
    np.testing.assert_equal(history[0], 0)
    np.testing.assert_equal(history[1, :, 0, 0], [4, 3, 2])
    backend.restore_rollout_state(snap)
    np.testing.assert_equal(backend.get_body_contact_force_history_w(env.body_ids), snap["contact_force_history"])


def test_disabled_history_preserves_old_batch_path_and_snapshot_keys():
    backend = FakeBackend()
    backend.step(np.zeros((2, 29)), 4)
    assert [n for n, _ in backend._pool.calls] == [4]
    assert set(backend.capture_rollout_state()) == {"physics_state", "sensor_data", "pending_xfrc_applied"}


def test_refresh_snapshot_reset_and_terminal_student_view(environment):
    env = environment
    state = env.reset_all()
    assert len(env._backend._pool.calls) == 4  # explicit warmup only
    assert env.step_counter == 1
    assert state.obs["obs"].shape == (2, 322)
    assert env.obs_groups_spec == {"obs": 322, "critic": 322}
    assert env.action_space.high[0] == 100
    snap = env.capture_rollout_snapshot()
    random_before = copy.deepcopy(np.random.get_state())
    action_before = env.actuation.action.copy(); phase_before = env.command.phase.copy()
    env.refresh_state(); env.refresh_state()
    assert len(env._backend._pool.calls) == 4
    np.testing.assert_equal(env.actuation.action, action_before)
    np.testing.assert_equal(env.command.phase, phase_before)
    np.testing.assert_equal(np.random.get_state()[1], random_before[1])
    env.step(np.full((2, 29), .1))
    submitted = env._backend._pool.calls[-4:]
    assert all(n == 1 for n, _ in submitted)
    assert not np.array_equal(submitted[0][1], submitted[1][1])  # fresh dq per PD substep
    np.testing.assert_allclose(env.actuation.torque, submitted[-1][1][:, 0, :29])
    env.restore_rollout_snapshot(snap)
    np.testing.assert_equal(env.state.obs["obs"], snap.state.obs["obs"])
    np.testing.assert_equal(env.command.phase, phase_before)
    # A timeout row has an independent pre-reset student view and teacher final obs.
    env.state.info["steps"][0] = env.cfg.max_episode_steps - 1
    env.command.velocity[:] = [1, 0, 0]
    env.command.phase[0] = [.3, -.4]
    out = env.step(np.full((2, 29), .2))
    assert out.truncated.tolist() == [True, False]
    assert out.info["final_student_observation_mask"].tolist() == [True, False]
    np.testing.assert_allclose(out.info["final_student_observation"][0, -2:], [.3, -.4])
    assert not np.allclose(env.get_student_observation()[0, -2:], [.3, -.4])
    np.testing.assert_allclose(out.final_observation["obs"][0, 64:93], .2)


def test_startup_dr_cached_and_fail_closed(environment):
    env = environment
    env.cfg.randomization_enabled = True
    from unilab.envs.locomotion.g1.official_oracle.randomization import OfficialOracleDRProvider
    provider = OfficialOracleDRProvider(env)
    env.actuation = ActuationState.create(2, True)
    env.reset_all()
    one = provider.build_reset_plan(env, np.array([0]))
    two = provider.build_reset_plan(env, np.array([0]))
    for name in ("body_mass", "body_inertia", "body_ipos", "geom_friction"):
        np.testing.assert_equal(getattr(one.randomization, name), getattr(two.randomization, name))
    assert not np.array_equal(one.qpos, two.qpos)
    assert np.max(np.abs(env.actuation.default - DEFAULT_ANGLES)) <= .05
    np.testing.assert_allclose(provider.ipos[:, provider.mass_ids[-1]], provider.com_offset)
    assert np.all((env.actuation.kp_scale >= .8) & (env.actuation.kp_scale <= 1.2))
    assert np.all((provider.friction[:, 1:3, 0] >= .1) & (provider.friction[:, 1:3, 0] <= 2))
    from unilab.dr.types import DomainRandomizationCapabilities
    with pytest.raises(NotImplementedError):
        provider.validate(env, DomainRandomizationCapabilities())


def test_push_is_post_physics_velocity_replacement(environment):
    env = environment
    env.reset_all()
    env.cfg.randomization_enabled = True
    env.cfg.evaluating = False
    env.push_counter[:] = 249; env.push_interval[:] = 5
    env._backend._base_lin_vel_view[:] = [10, 10, .4]
    assert env.randomization.build_interval_randomization_plan(env, 0) is None
    env._post_physics_randomization = True
    plan = env.randomization.build_interval_randomization_plan(env, 0)
    env._backend.apply_interval_randomization(plan)
    velocity = env.get_base_lin_vel()
    assert np.all(np.abs(velocity[:, 0]) <= .1)
    assert np.all(np.abs(velocity[:, 1]) <= 1.5)
    np.testing.assert_allclose(velocity[:, 2], .4)
    np.testing.assert_equal(env.push_counter, 0)
