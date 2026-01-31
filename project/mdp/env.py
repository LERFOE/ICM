from __future__ import annotations
from typing import Dict, Tuple
import numpy as np

from project.mdp.action import ActionVector
from project.mdp.config import MDPConfig
from project.mdp.mask import enumerate_valid_actions, is_action_valid, roll_K, mutable_mask
from project.mdp.phase import PHASE_OFFSEASON, next_phase
from project.mdp.reward import terminal_value
from project.mdp.state import State, initial_state
from project.mdp.transitions_comp import comp_transition
from project.mdp.transitions_env import env_transition
from project.mdp.transitions_fin import fin_transition_and_reward


class MDPEnv:
    def __init__(self, config: MDPConfig | None = None, seed: int = 42, use_data: bool = False):
        self.config = config or MDPConfig()
        self.rng = np.random.default_rng(seed)
        self.use_data = use_data
        self.player_model = None

    def valid_actions(self, state: State):
        actions = list(enumerate_valid_actions(state.Theta, state.K))
        mask = mutable_mask(state.Theta)

        # Apply caps only when the dimension is mutable; if frozen, allow current K
        if self.config.max_debt_action is not None and mask[4] == 1:
            actions = [a for a in actions if a.a_debt <= self.config.max_debt_action]
        if self.config.max_equity_action is not None and mask[5] == 1:
            actions = [a for a in actions if a.a_equity <= self.config.max_equity_action]

        # Guard leverage-up only when debt is mutable
        if mask[4] == 1 and state.F.leverage >= self.config.leverage_soft:
            actions = [a for a in actions if a.a_debt <= 2]
        return actions

    def reset(self, seed: int | None = None, use_data: bool | None = None, year: int | None = None) -> State:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        use_data = self.use_data if use_data is None else use_data
        if not use_data:
            return initial_state(self.config, self.rng)
        # Lazy import to avoid pandas dependency unless needed
        from project.data.integration import build_initial_state_from_data

        try:
            from project.data.player_kmeans import build_player_model

            self.player_model = build_player_model(roster_size=self.config.roster_size)
        except Exception:
            self.player_model = None

        return build_initial_state_from_data(self.config, self.rng, year=year)

    def step(
        self,
        state: State,
        action: ActionVector,
        rng: np.random.Generator | None = None,
    ) -> Tuple[State, float, bool, Dict]:
        rng = rng or self.rng
        if not is_action_valid(state.Theta, state.K, action):
            raise ValueError("Action violates phase-aware mask")
        if action not in self.valid_actions(state):
            raise ValueError("Action violates risk caps")

        # 0) roll K
        K_next = roll_K(state.Theta, state.K, action)

        # 1) competitive transition uses the *current* environment (what the GM observes at decision time).
        R_next, comp_info = comp_transition(
            state.R,
            action,
            state.Theta,
            state.E,
            rng,
            self.config,
            player_model=self.player_model,
            team_code=self.config.team_code,
        )

        # 2) financial transition and reward uses the current environment as well.
        F_next, reward = fin_transition_and_reward(state.F, R_next, state.E, action, state.Theta, rng, self.config)
        F_next.K = list(K_next)

        # 3) phase advance
        Theta_next, wraps = next_phase(state.Theta)
        year_next = state.year + 1 if wraps else state.year

        # 4) environment transition into the next phase/year.
        # This makes expansion shocks visible *before* offseason decisions (state includes I_expansion etc.).
        E_next = env_transition(state.E, year_next, Theta_next, rng, self.config)

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
