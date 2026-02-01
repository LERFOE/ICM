from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from project.data.integration import build_initial_state_from_data
from project.data.player_kmeans import build_player_model
from project.mdp.action import ActionVector
from project.mdp.config import MDPConfig
from project.mdp.env import MDPEnv
from project.mdp.mask import enumerate_valid_actions, mutable_mask, roll_K
from project.mdp.phase import PHASE_OFFSEASON, next_phase
from project.mdp.reward import terminal_value
from project.mdp.state import State, initial_state
from project.mdp.transitions_comp import comp_transition
from project.mdp.transitions_env import env_transition
from project.mdp.transitions_fin import fin_transition_and_reward


@dataclass
class GameState:
    ind: State
    opp: State

    def copy(self) -> "GameState":
        return GameState(ind=self.ind.copy(), opp=self.opp.copy())


def _valid_actions_for_state(state: State, config: MDPConfig) -> List[ActionVector]:
    actions = list(enumerate_valid_actions(state.Theta, state.K))
    mask = mutable_mask(state.Theta)
    if config.max_debt_action is not None and mask[4] == 1:
        actions = [a for a in actions if a.a_debt <= config.max_debt_action]
    if config.max_equity_action is not None and mask[5] == 1:
        actions = [a for a in actions if a.a_equity <= config.max_equity_action]
    if mask[4] == 1 and state.F.leverage >= config.leverage_soft:
        actions = [a for a in actions if a.a_debt <= 2]
    return actions


def _terminal_check(state: State, config: MDPConfig) -> bool:
    if state.F.leverage > config.max_leverage:
        return True
    if state.F.Cash < config.bankruptcy_cash:
        return True
    if state.year >= config.start_year + config.horizon_years and state.Theta == PHASE_OFFSEASON:
        return True
    return False


class StochasticGameEnv:
    """Two-player stochastic game wrapper around the single-team MDP dynamics."""

    def __init__(
        self,
        config: Optional[MDPConfig] = None,
        team_ind: str = "IND",
        team_opp: str = "NYL",
        seed: int = 42,
        use_data: bool = True,
    ):
        self.base_config = config or MDPConfig()
        self.config_ind = replace(self.base_config, team_code=team_ind)
        self.config_opp = replace(self.base_config, team_code=team_opp)
        self.rng = np.random.default_rng(seed)
        self.use_data = use_data
        self.player_model = None
        if use_data:
            try:
                self.player_model = build_player_model(roster_size=self.base_config.roster_size)
            except Exception:
                self.player_model = None

    def reset(self, seed: Optional[int] = None, year: Optional[int] = None) -> GameState:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if not self.use_data:
            s_ind = initial_state(self.config_ind, self.rng)
            s_opp = initial_state(self.config_opp, self.rng)
            return GameState(ind=s_ind, opp=s_opp)
        s_ind = build_initial_state_from_data(self.config_ind, self.rng, year=year)
        s_opp = build_initial_state_from_data(self.config_opp, self.rng, year=year)
        return GameState(ind=s_ind, opp=s_opp)

    def _apply_competition(self, E_base, opp_action: ActionVector) -> object:
        E = E_base.copy()
        # Talent scarcity: aggressive roster/salary behavior reduces available star FA
        if opp_action.a_roster >= 5 or opp_action.a_salary >= 4:
            E.n_star_fa = max(0, int(E.n_star_fa - 1))
        # Bidding intensity rises with opponent aggression
        E.bidding_intensity = max(
            0.0,
            float(E.bidding_intensity)
            + 0.3 * max(0, opp_action.a_roster - 3)
            + 0.5 * max(0, opp_action.a_salary - 2),
        )
        # Local market share pressure from opponent marketing
        market_share = 1.0 - 0.02 * max(0, opp_action.a_marketing - 1)
        E.mu_size = max(0.5, min(2.0, E.mu_size * market_share))
        return E

    def step(
        self,
        state: GameState,
        action_ind: ActionVector,
        action_opp: ActionVector,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[GameState, Tuple[float, float], bool, Dict]:
        rng = rng or self.rng
        # Phase masking and K-roll
        K_ind = roll_K(state.ind.Theta, state.ind.K, action_ind)
        K_opp = roll_K(state.opp.Theta, state.opp.K, action_opp)

        # Effective environments (competition enters here)
        E_eff_ind = self._apply_competition(state.ind.E, action_opp)
        E_eff_opp = self._apply_competition(state.opp.E, action_ind)

        # Competitive transitions
        R_ind, comp_info_ind = comp_transition(
            state.ind.R,
            action_ind,
            state.ind.Theta,
            E_eff_ind,
            rng,
            self.config_ind,
            player_model=self.player_model,
            team_code=self.config_ind.team_code,
        )
        R_opp, comp_info_opp = comp_transition(
            state.opp.R,
            action_opp,
            state.opp.Theta,
            E_eff_opp,
            rng,
            self.config_opp,
            player_model=self.player_model,
            team_code=self.config_opp.team_code,
        )

        # Financial transitions
        F_ind, r_ind = fin_transition_and_reward(
            state.ind.F, R_ind, E_eff_ind, action_ind, state.ind.Theta, rng, self.config_ind
        )
        F_opp, r_opp = fin_transition_and_reward(
            state.opp.F, R_opp, E_eff_opp, action_opp, state.opp.Theta, rng, self.config_opp
        )
        F_ind.K = list(K_ind)
        F_opp.K = list(K_opp)

        # Phase advance (synchronized league calendar)
        Theta_next, wraps = next_phase(state.ind.Theta)
        year_next = state.ind.year + 1 if wraps else state.ind.year

        # Shared league environment update
        E_next = env_transition(state.ind.E, year_next, Theta_next, rng, self.base_config)

        next_ind = State(R=R_ind, F=F_ind, E=E_next, Theta=Theta_next, K=list(K_ind), year=year_next)
        next_opp = State(R=R_opp, F=F_opp, E=E_next, Theta=Theta_next, K=list(K_opp), year=year_next)
        next_state = GameState(ind=next_ind, opp=next_opp)

        done = _terminal_check(next_ind, self.config_ind) or _terminal_check(next_opp, self.config_opp)
        if done:
            r_ind += self.base_config.terminal_weight * terminal_value(next_ind.F)
            r_opp += self.base_config.terminal_weight * terminal_value(next_opp.F)

        info = {"comp_ind": comp_info_ind, "comp_opp": comp_info_opp}
        return next_state, (r_ind, r_opp), done, info


class BestResponseEnv:
    """Single-agent view of the stochastic game with opponent policy fixed."""

    def __init__(
        self,
        game_env: StochasticGameEnv,
        opponent_policy: Callable[[GameState], ActionVector],
        player: str = "ind",
    ):
        self.game_env = game_env
        self.opponent_policy = opponent_policy
        self.player = player
        self.config = game_env.config_ind if player == "ind" else game_env.config_opp

    def reset(self, seed: Optional[int] = None) -> GameState:
        return self.game_env.reset(seed=seed)

    def valid_actions(self, game_state: GameState) -> List[ActionVector]:
        state = game_state.ind if self.player == "ind" else game_state.opp
        return _valid_actions_for_state(state, self.config)

    def step(self, game_state: GameState, action: ActionVector, rng=None):
        if self.player == "ind":
            opp_action = self.opponent_policy(game_state)
            next_state, (r_ind, _), done, info = self.game_env.step(
                game_state, action, opp_action, rng=rng
            )
            return next_state, r_ind, done, info
        opp_action = self.opponent_policy(game_state)
        next_state, (_, r_opp), done, info = self.game_env.step(
            game_state, opp_action, action, rng=rng
        )
        return next_state, r_opp, done, info
