from __future__ import annotations

from project.mdp.config import MDPConfig
from project.mdp.env import MDPEnv


def build_env(use_data: bool = True, seed: int = 42) -> MDPEnv:
    if not use_data:
        return MDPEnv(MDPConfig(), seed=seed, use_data=False)
    try:
        from project.data.integration import calibrate_config_from_data

        cfg, _ = calibrate_config_from_data(MDPConfig())
        return MDPEnv(cfg, seed=seed, use_data=True)
    except Exception:
        # fallback to defaults if data not available
        return MDPEnv(MDPConfig(), seed=seed, use_data=False)
