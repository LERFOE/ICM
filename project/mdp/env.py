from __future__ import annotations
from typing import Dict, Tuple
import numpy as np

from project.mdp.action import ActionVector
from project.mdp.config import MDPConfig
from project.mdp.mask import is_action_valid, roll_K
from project.mdp.phase import PHASE_OFFSEASON, next_phase
from project.mdp.reward import terminal_value
from project.mdp.state import State, initial_state
from project.mdp.transitions_comp import comp_transition
from project.mdp.transitions_env import env_transition
from project.mdp.transitions_fin import fin_transition_and_reward


class MDPEnv:
    def __init__(self, config: MDPConfig | None = None, seed: int = 42):
        self.config = config or MDPConfig()
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> State:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        return initial_state(self.config, self.rng)

    def step(
        self,
        state: State,
        action: ActionVector,
        rng: np.random.Generator | None = None,
    ) -> Tuple[State, float, bool, Dict]:
        rng = rng or self.rng
        if not is_action_valid(state.Theta, state.K, action):
            raise ValueError("Action violates phase-aware mask")

        # 0) roll K
        K_next = roll_K(state.Theta, state.K, action)

        # 1) environment transition
        E_next = env_transition(state.E, state.year, rng, self.config)

        # 2) competitive transition
        R_next, comp_info = comp_transition(state.R, action, state.Theta, E_next, rng, self.config)

        # 3) financial transition and reward
        F_next, reward = fin_transition_and_reward(state.F, R_next, E_next, action, state.Theta, rng, self.config)
        F_next.K = list(K_next)

        # 4) phase advance
        Theta_next, wraps = next_phase(state.Theta)
        year_next = state.year + 1 if wraps else state.year

        next_state = State(
            R=R_next,
            F=F_next,
            E=E_next,
            Theta=Theta_next,
            K=list(K_next),
            year=year_next,
        )

        done = self._terminal_check(next_state)
        if done:
            reward += self.config.terminal_weight * terminal_value(next_state.F)

        info = {"comp": comp_info, "wraps": wraps}
        return next_state, reward, done, info

    def _terminal_check(self, state: State) -> bool:
        if state.F.leverage > self.config.max_leverage:
            return True
        if state.F.Cash < self.config.bankruptcy_cash:
            return True
        if state.year >= self.config.start_year + self.config.horizon_years and state.Theta == PHASE_OFFSEASON:
            return True
        return False
