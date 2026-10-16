from unilab.base import registry

from .config import G1OfficialOracleFlatCfg
from .env import G1OfficialOracleFlatEnv

registry.register_env("G1OfficialOracleFlat", G1OfficialOracleFlatEnv, sim_backend="mujoco")

__all__ = ["G1OfficialOracleFlatCfg", "G1OfficialOracleFlatEnv"]
