"""Startup physical draws are cached; episode actuator state is resampled."""
import numpy as np

from unilab.dr import IntervalRandomizationPlan, ResetPlan, ResetRandomizationPayload
from unilab.dr.provider import DomainRandomizationProvider

from .config import MASS_BODIES


class OfficialOracleDRProvider(DomainRandomizationProvider):
    def __init__(self, env):
        backend, n = env._backend, env.num_envs
        self.mass_ids = backend.get_body_ids(MASS_BODIES)
        torso = int(self.mass_ids[-1])
        nominal = backend.get_body_mass()
        self.mass = np.broadcast_to(nominal, (n, len(nominal))).copy()
        self.ipos = np.broadcast_to(backend.get_body_ipos(), (n, len(nominal), 3)).copy()
        nominal_inertia = backend.get_body_inertia()
        self.friction = np.broadcast_to(backend.get_geom_friction(), (n, *backend.get_geom_friction().shape)).copy()
        collision_type, _ = backend.get_geom_contact_masks()
        collision_geoms = np.flatnonzero(collision_type != 0)
        if not len(collision_geoms):
            raise ValueError("Oracle scene must expose robot collision geoms")
        self.com_offset = np.zeros((n, 3), np.float32)
        if env.cfg.randomization_enabled:
            self.mass[:, self.mass_ids[:-1]] *= np.random.uniform(.8, 1.3, (n, 9))
            self.mass[:, torso] += np.random.uniform(-3, 6, n)
            self.com_offset[:] = np.random.uniform(-.15, .15, (n, 3))
            self.ipos[:, torso] += self.com_offset
            # IsaacSim material randomization selects buckets per rigid shape.
            # MuJoCo has one sliding coefficient, not separate static/dynamic friction.
            buckets = np.random.uniform(.1, 2., 64)
            self.friction[:, collision_geoms, 0] = buckets[np.random.randint(0, 64, (n, len(collision_geoms)))]
        self.friction_obs = self.friction[:, collision_geoms, 0].mean(axis=1)
        mass_ratio = np.divide(self.mass, nominal, out=np.ones_like(self.mass), where=nominal > 0)
        self.inertia = nominal_inertia[None] * mass_ratio[:, :, None]

    def validate(self, env, capabilities):
        needed = frozenset({"body_mass", "body_inertia", "body_ipos", "geom_friction"})
        missing = capabilities.get_unsupported_reset_terms(needed)
        if missing or not capabilities.supports_interval_body_velocity_delta:
            raise NotImplementedError(f"Official Oracle DR capabilities missing: {sorted(missing)} / velocity push")

    def build_reset_plan(self, env, env_ids):
        ids = np.asarray(env_ids, dtype=np.int32)
        n, randomize = len(ids), env.cfg.randomization_enabled
        env.actuation.reset(ids, randomize)
        env.command.reset(ids, env.cfg.evaluating)
        env.last_dq[ids] = 0
        env.push_counter[ids] = 0
        env.push_interval[ids] = np.random.uniform(5, 10, n) if randomize else 10
        qpos = np.broadcast_to(env._init_qpos, (n, len(env._init_qpos))).copy()
        qvel = np.zeros((n, len(env._init_qvel)), dtype=np.float64)
        qpos[:, 7:] = env.actuation.default[ids]
        if randomize:
            qpos[:, 7:] *= np.random.uniform(.5, 1.5, (n, 29))
            qvel[:, :6] = np.random.uniform(-.5, .5, (n, 6))
        return ResetPlan(ids, qpos, qvel, {}, ResetRandomizationPayload(
            body_mass=self.mass[ids].copy(), body_inertia=self.inertia[ids].copy(),
            body_ipos=self.ipos[ids].copy(), geom_friction=self.friction[ids].copy(),
        ))

    def build_reset_observation(self, env, env_ids, info_updates):
        return {name: value[env_ids] for name, value in env._observe().items()}

    def build_interval_randomization_plan(self, env, step_counter):
        # NpEnv's pre-physics hook is intentionally inactive: upstream pushes
        # after reward/termination and partial resets, immediately before obs.
        if not env._post_physics_randomization or not env.cfg.randomization_enabled or env.cfg.evaluating:
            return None
        env.push_counter += 1
        due = env.push_counter >= (env.push_interval / env.cfg.ctrl_dt).astype(np.int64)
        ids = np.flatnonzero(due)
        if not len(ids):
            return None
        delta = np.zeros((env.num_envs, 1, 3), np.float64)
        target = np.random.uniform(-1, 1, (len(ids), 2)) * env.max_push_xy[ids]
        delta[ids, 0, :2] = target - env._backend.get_base_lin_vel()[ids, :2]
        env.push_counter[ids] = 0
        env.push_interval[ids] = np.random.uniform(5, 10, len(ids))
        return IntervalRandomizationPlan(body_ids=env.root_ids, body_linear_velocity_delta=delta)

    def features(self, env):
        progress = np.clip(env.push_counter / (env.push_interval / env.cfg.ctrl_dt), 0, 1)
        return np.concatenate((self.mass[:, self.mass_ids], self.friction_obs[:, None], self.com_offset,
                               env.push_interval[:, None], progress[:, None], env.max_push_xy), axis=1).astype(np.float32)
