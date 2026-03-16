# src/sixnimmt_env/__init__.py
from .env import SixQuiPrendEnv
from .env_fixed_opponents import SixQuiPrendEnvFixedOpponents

__all__ = [
    "SixQuiPrendEnv",
    "SixQuiPrendEnvFixedOpponents",
]