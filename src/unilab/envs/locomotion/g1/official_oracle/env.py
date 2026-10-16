"""Standalone Oracle task. The PPO learner and rollout transport live elsewhere."""
import copy
from dataclasses import fields

import gymnasium as gym
import numpy as np

from unilab.base.backend import create_backend
from unilab.envs.locomotion.g1.base import G1BaseEnv

from .actuation import ActuationState
from .command import CommandState
from .config import DEFAULT_ANGLES, EFFORT, JOINT_NAMES, STUDENT_SCHEMA, build_environment_contract
from .observation import PhysicalFrame, build_student, build_teacher
from .randomization import OfficialOracleDRProvider
from .reward import reward_terms, total_reward


class G1OfficialOracleFlatEnv(G1BaseEnv):
    def __init__(self, cfg, num_envs=1, backend_type="mujoco"):
        cfg.validate()
        self._official_oracle_contract = build_environment_contract(cfg)
        self._external_command_rows = None
        if backend_type != "mujoco":
            raise ValueError("Official Oracle flat v1 requires MuJoCo")
        backend = create_backend(
            backend_type, cfg.scene, num_envs, cfg.sim_dt, base_name="pelvis",
            push_body_name="pelvis", add_body_sensors=True,
            mujoco_num_threads=cfg.mujoco_num_threads, post_step_forward_sensor=True,
        )
        try:
            super().__init__(cfg, backend, num_envs)
            if backend.get_actuated_joint_names() != JOINT_NAMES:
                raise ValueError("Official Oracle actuator joint order mismatch")
            if not np.allclose(self.default_angles, DEFAULT_ANGLES, rtol=0, atol=1e-7):
                raise ValueError("Official Oracle stand keyframe mismatch")
            if not np.allclose(backend.get_actuator_force_range(), np.stack((-EFFORT, EFFORT), axis=1)):
                raise ValueError("Official Oracle motor effort mismatch")
            names = tuple(name for name in backend.get_body_names() if name != "world")
            self.body_ids = backend.get_body_ids(names)
            backend.configure_contact_history(names, history_length=3, sample_dt=cfg.sim_dt)
            self.root_ids = backend.get_body_ids(("pelvis",))
            self.foot_ids = backend.get_body_ids(("left_ankle_roll_link", "right_ankle_roll_link"))
            self.foot_point_ids = backend.get_body_ids(("left_foot_contact_point", "right_foot_contact_point"))
            self.foot_slots = np.array([names.index(name) for name in ("left_ankle_roll_link", "right_ankle_roll_link")])
            self.undesired_slots = np.array([i for i, name in enumerate(names)
                                            if "ankle" not in name and "foot_contact" not in name], dtype=int)
            self.termination_slots = np.array([i for i, name in enumerate(names)
                                              if any(part in name for part in ("pelvis", "shoulder", "hip"))], dtype=int)
            self.actuation = ActuationState.create(num_envs, cfg.randomization_enabled)
            self.command = CommandState.create(num_envs)
            self.last_dq = np.zeros((num_envs, 29), np.float32)
            self.push_counter = np.zeros(num_envs, np.int64)
            self.push_interval = np.full(num_envs, 10., np.float32)
            self.max_push_xy = np.broadcast_to([.1, 1.5], (num_envs, 2)).astype(np.float32).copy()
            self._transition_pending = False
            self._post_physics_randomization = False
            self.randomization = OfficialOracleDRProvider(self)
            self._init_domain_randomization(self.randomization)
            backend.set_pre_step_control(self._substep_torque)
        except BaseException:
            backend.set_pre_step_control(None)
            backend.close()
            raise

    def _init_action_space(self):
        self._action_space = gym.spaces.Box(-100., 100., (29,), dtype=np.float32)

    @property
    def obs_groups_spec(self):
        return {"obs": 322, "critic": 322}

    def _frame(self):
        backend = self._backend
        foot_pos, foot_quat, foot_vel, _ = backend.get_body_state_w(self.foot_ids)
        point_pos, _, point_vel, _ = backend.get_body_state_w(self.foot_point_ids)
        return PhysicalFrame(
            backend.get_dof_pos(), backend.get_dof_vel(), backend.get_base_pos(),
            backend.get_base_quat(), backend.get_base_lin_vel(), backend.get_body_ang_vel_w(self.root_ids)[:, 0],
            foot_pos, foot_quat, foot_vel, point_pos, point_vel,
            backend.get_body_contact_forces_w(self.body_ids),
            backend.get_body_contact_force_history_w(self.body_ids),
            backend.get_sensor_data("torso_gyro"), backend.get_sensor_data("torso_upvector"),
        )

    def _observe(self):
        obs = build_teacher(self._frame(), self.actuation, self.command,
                            self.randomization.features(self), self.foot_slots,
                            rfi_enabled=self.cfg.randomization_enabled)
        return {"obs": obs, "critic": obs.copy()}

    def get_student_observation(self):
        return build_student(self._frame(), self.actuation, self.command)

    def get_official_oracle_contract(self):
        return copy.deepcopy(self._official_oracle_contract)

    def initialize_training_progress(self):
        """One post-warm-reset clock scatter; refresh must not step physics."""
        if self.cfg.evaluating or self._state is None:
            raise ValueError("Initial training progress requires a prepared training environment")
        self.state.info["steps"][:] = np.random.randint(0, self.cfg.max_episode_steps, self.num_envs)
        self.command.update_phase(self.state.info["steps"])
        return self.refresh_state()

    def _publish_collection_views(self, state):
        state.info["commands"] = self.command.velocity
        state.info["student_observation"] = self.get_student_observation()
        state.info["student_observation_contract"] = STUDENT_SCHEMA

    def set_external_command(self, rows):
        rows = np.asarray(rows, dtype=np.float32)
        if rows.shape == (3,):
            rows = np.broadcast_to(rows, (self.num_envs, 3))
        if rows.shape != (self.num_envs, 3) or not np.isfinite(rows).all() or np.any(np.abs(rows) > 1):
            raise ValueError("Official Oracle external commands require finite (N,3) rows in [-1,1]")
        self._external_command_rows = rows.copy()
        self.command.velocity[:] = rows
        self.command.update_phase(self.state.info["steps"])
        return self.refresh_state()

    def clear_external_command(self):
        self._external_command_rows = None

    def _restore_external_command(self, state):
        if self._external_command_rows is not None:
            self.command.velocity[:] = self._external_command_rows
            self.command.update_phase(state.info["steps"])

    def _substep_torque(self, backend, ctrl):
        return self.actuation.pd(backend.get_dof_pos(), backend.get_dof_vel(), self.cfg.randomization_enabled)

    def apply_action(self, actions, state):
        target = self.actuation.process(actions)
        state.info["final_student_observation_mask"].fill(False)
        self._transition_pending = True
        return target

    def update_state(self, state):
        frame = self._frame()
        if self._transition_pending:
            terms = reward_terms(frame, self.actuation, self.command, self.last_dq,
                                 self.foot_slots, self.undesired_slots, self.cfg.ctrl_dt)
            state.reward[:] = total_reward(terms, self.cfg.reward_config.scales, self.cfg.ctrl_dt)
            state.info["reward_terms"] = terms
            state.terminated[:] = (
                np.any(np.linalg.norm(frame.forces[:, self.termination_slots], axis=-1) > 1, axis=1)
                | np.any(np.abs(frame.gravity[:, :2]) > .75, axis=1) | (frame.root_pos[:, 2] < .32)
            )
            self.last_dq[:] = frame.dq
            self._transition_pending = False
        self._restore_external_command(state)
        state.obs = self._observe()
        self._publish_collection_views(state)
        return state

    def _on_step_completed(self, state):
        done = state.terminated | state.truncated
        state.info["final_student_observation_mask"][:] = done
        state.info["final_student_observation"][done] = self.get_student_observation()[done]

    def init_state(self):
        state = super().init_state()
        state.obs = self._observe()
        state.terminated.fill(False)
        state.info["final_student_observation"] = np.zeros((self.num_envs, 66), np.float32)
        state.info["final_student_observation_mask"] = np.zeros(self.num_envs, bool)
        self._restore_external_command(state)
        state.obs = self._observe()
        self._publish_collection_views(state)
        return state

    def reset_all(self):
        super().reset_all()
        # Upstream reset_all prepares observations with one zero-action control
        # step. Only this explicit path steps physics, never refresh/partial reset.
        state = self.step(np.zeros((self.num_envs, 29), np.float32))
        state.reward.fill(0)
        self._clear_step_final_observation()
        state.info["final_student_observation_mask"].fill(False)
        return state

    def step(self, actions):
        if self._state is None:
            self.reset_all()
        state = super().step(actions)
        self.command.advance(state.info["steps"], self.cfg.evaluating,
                             external=self._external_command_rows is not None)
        self._restore_external_command(state)
        self._post_physics_randomization = True
        try:
            self._dr_manager.apply_interval_randomization_if_due(self.step_counter)
        finally:
            self._post_physics_randomization = False
        state.obs = self._observe()
        self._publish_collection_views(state)
        return state

    def _capture_task_rollout_state(self):
        return copy.deepcopy({
            "actuation": self.actuation, "command": self.command,
            "last_dq": self.last_dq, "push_counter": self.push_counter,
            "push_interval": self.push_interval, "max_push_xy": self.max_push_xy,
            "transition_pending": self._transition_pending,
            "post_physics_randomization": self._post_physics_randomization,
            "external_command_rows": self._external_command_rows,
        })

    def _restore_task_rollout_state(self, snapshot):
        for name in ("actuation", "command"):
            target = getattr(self, name)
            for field in fields(target):
                getattr(target, field.name)[:] = getattr(snapshot[name], field.name)
        for name in ("last_dq", "push_counter", "push_interval", "max_push_xy"):
            getattr(self, name)[:] = snapshot[name]
        self._transition_pending = snapshot["transition_pending"]
        self._post_physics_randomization = snapshot["post_physics_randomization"]
        self._external_command_rows = copy.deepcopy(snapshot["external_command_rows"])
        if self._state is not None:
            self._publish_collection_views(self._state)

    def close(self):
        self._backend.set_pre_step_control(None)
        super().close()
