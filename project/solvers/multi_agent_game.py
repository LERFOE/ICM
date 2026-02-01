from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from project.data.geo import TEAM_LOCATIONS, haversine_km
from project.data.integration import build_initial_state_from_data, calibrate_config_from_data
from project.data.player_kmeans import build_player_model
from project.experiments.q3_expansion_site_sensitivity import _market_index, TEAM_CODE_TO_NAME
from project.mdp.action import ActionVector
from project.mdp.config import MDPConfig
from project.mdp.mask import enumerate_valid_actions, mutable_mask, roll_K
from project.mdp.phase import PHASE_OFFSEASON, next_phase
from project.mdp.reward import terminal_value
from project.mdp.state import State, initial_state
from project.mdp.transitions_comp import comp_transition
from project.mdp.transitions_env import env_transition
from project.mdp.transitions_fin import fin_transition_and_reward


@dataclass
class MultiGameState:
    teams: List[State]

    def copy(self) -> "MultiGameState":
        return MultiGameState(teams=[s.copy() for s in self.teams])


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


def _team_expansion_params(site: Dict, team_code: str) -> Dict[str, float | int]:
    loc = TEAM_LOCATIONS.get(team_code)
    if loc is None:
        dist = 1500.0
    else:
        dist = haversine_km(loc.lat, loc.lon, site["lat"], site["lon"])
    overlap = max(0.0, 1.0 - dist / 600.0)
    market_delta = 0.02 * float(site["market_score"]) - 0.03 * overlap - 0.01 * overlap
    compete_delta = 1 if overlap > 0.30 else 0
    travel_fatigue = min(1.0, dist / 3000.0) * (1.0 - 0.2 * float(site["hub_score"]))
    return {
        "dist_km": dist,
        "overlap": overlap,
        "market_delta": market_delta,
        "compete_delta": compete_delta,
        "travel_fatigue": travel_fatigue,
    }


class MultiAgentGameEnv:
    """N-agent simultaneous stochastic game with shared league environment and team-specific market profiles."""

    def __init__(
        self,
        team_codes: List[str],
        expansion_site: Dict,
        seed: int = 42,
        use_data: bool = True,
    ):
        base_cfg = MDPConfig()
        try:
            base_cfg, _ = calibrate_config_from_data(base_cfg)
        except Exception:
            pass

        self.rng = np.random.default_rng(seed)
        self.use_data = use_data
        self.expansion_site = expansion_site
        self.base_cfg = base_cfg
        self.team_codes = team_codes
        self.team_names = {c: TEAM_CODE_TO_NAME.get(c, c) for c in team_codes}

        # Team-specific market sizes from attendance
        market_idx = _market_index()
        self.team_market = {c: float(np.clip(market_idx.get(c, 1.0), 0.5, 2.0)) for c in team_codes}
        self.team_base_compete = {c: 0 for c in team_codes}
        self.team_expansion = {c: _team_expansion_params(expansion_site, c) for c in team_codes}

        # Configs per team
        self.team_cfgs: Dict[str, MDPConfig] = {}
        for code in team_codes:
            cfg = replace(base_cfg, team_code=code)
            scale = self.team_market[code]
            cfg.market_size = scale
            # Scale team financial baselines by market size to avoid identical states.
            cfg.base_franchise_value = base_cfg.base_franchise_value * scale
            cfg.base_gate_revenue = base_cfg.base_gate_revenue * scale
            cfg.base_media_revenue = base_cfg.base_media_revenue * scale
            cfg.base_sponsor_revenue = base_cfg.base_sponsor_revenue * scale
            cfg.base_cash = base_cfg.base_cash * scale
            cfg.base_debt = base_cfg.base_debt * scale
            self.team_cfgs[code] = cfg

        self.player_model = None
        if use_data:
            try:
                self.player_model = build_player_model(roster_size=base_cfg.roster_size)
            except Exception:
                self.player_model = None

    def reset(self, seed: Optional[int] = None, year: Optional[int] = None) -> MultiGameState:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        teams: List[State] = []
        for code in self.team_codes:
            cfg = self.team_cfgs[code]
            if self.use_data:
                state = build_initial_state_from_data(cfg, self.rng, year=year)
            else:
                state = initial_state(cfg, self.rng)
            state.E.mu_size = cfg.market_size
            state.E.compete_local = cfg.compete_local
            state.E.travel_fatigue = 0.0
            teams.append(state)
        return MultiGameState(teams=teams)

    def valid_actions(self, game_state: MultiGameState, team_idx: int) -> List[ActionVector]:
        code = self.team_codes[team_idx]
        return _valid_actions_for_state(game_state.teams[team_idx], self.team_cfgs[code])

    def _apply_competition(self, E_base, actions: List[ActionVector], team_idx: int) -> object:
        # Aggregate other teams' actions
        roster_aggr = 0.0
        salary_aggr = 0.0
        marketing_aggr = 0.0
        for j, a in enumerate(actions):
            if j == team_idx:
                continue
            roster_aggr += max(0, a.a_roster - 3)
            salary_aggr += max(0, a.a_salary - 2)
            marketing_aggr += max(0, a.a_marketing - 1)
        E = E_base.copy()
        E.n_star_fa = max(0, int(E.n_star_fa - 0.5 * roster_aggr - 0.5 * salary_aggr))
        E.bidding_intensity = max(0.0, float(E.bidding_intensity) + 0.3 * roster_aggr + 0.5 * salary_aggr)
        E.mu_size = max(0.5, min(2.0, E.mu_size * (1.0 - 0.02 * marketing_aggr)))
        return E

    def step(
        self,
        state: MultiGameState,
        actions: List[ActionVector],
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[MultiGameState, List[float], bool, Dict]:
        rng = rng or self.rng
        rewards: List[float] = []
        next_states: List[State] = []
        comp_infos: List[Dict] = []

        # Roll K for each team
        K_next_list = [roll_K(s.Theta, s.K, a) for s, a in zip(state.teams, actions)]

        for idx, (team_state, action) in enumerate(zip(state.teams, actions)):
            code = self.team_codes[idx]
            cfg = self.team_cfgs[code]

            E_eff = self._apply_competition(team_state.E, actions, idx)
            if E_eff.i_expansion == 1:
                E_eff.travel_fatigue = self.team_expansion[code]["travel_fatigue"]

            R_next, comp_info = comp_transition(
                team_state.R,
                action,
                team_state.Theta,
                E_eff,
                rng,
                cfg,
                player_model=self.player_model,
                team_code=code,
            )
            F_next, reward = fin_transition_and_reward(team_state.F, R_next, E_eff, action, team_state.Theta, rng, cfg)
            F_next.K = list(K_next_list[idx])

            next_states.append(
                State(
                    R=R_next,
                    F=F_next,
                    E=team_state.E.copy(),
                    Theta=team_state.Theta,
                    K=list(K_next_list[idx]),
                    year=team_state.year,
                )
            )
            rewards.append(reward)
            comp_infos.append(comp_info)

        # Phase advance (shared)
        Theta_next, wraps = next_phase(state.teams[0].Theta)
        year_next = state.teams[0].year + 1 if wraps else state.teams[0].year

        # Shared env transition (macro/cap/FA/bidding), but do NOT apply team-specific market deltas here.
        shared_E = state.teams[0].E
        E_next_shared = env_transition(
            shared_E,
            year_next,
            Theta_next,
            rng,
            self.base_cfg,
            expansion_market_delta=0.0,
            expansion_compete_delta=0,
            expansion_media_bonus=self.base_cfg.expansion_media_bonus,
            expansion_travel_fatigue=0.0,
        )

        # Apply team-specific market/competition/fatigue updates
        for idx, team_state in enumerate(next_states):
            code = self.team_codes[idx]
            cfg = self.team_cfgs[code]
            E_next = E_next_shared.copy()
            base_market = cfg.market_size
            base_compete = cfg.compete_local
            if E_next.i_expansion == 1:
                delta_m = self.team_expansion[code]["market_delta"]
                delta_c = self.team_expansion[code]["compete_delta"]
                E_next.mu_size = max(0.5, min(2.0, base_market + delta_m))
                E_next.compete_local = max(0, min(2, base_compete + int(delta_c)))
                E_next.travel_fatigue = float(self.team_expansion[code]["travel_fatigue"])
            else:
                E_next.mu_size = base_market
                E_next.compete_local = base_compete
                E_next.travel_fatigue = 0.0

            team_state.E = E_next
            team_state.Theta = Theta_next
            team_state.year = year_next

        next_state = MultiGameState(teams=next_states)

        done = False
        for idx, team_state in enumerate(next_states):
            code = self.team_codes[idx]
            if _terminal_check(team_state, self.team_cfgs[code]):
                done = True
                rewards[idx] += self.base_cfg.terminal_weight * terminal_value(team_state.F)
        info = {"comp": comp_infos, "wraps": wraps}
        return next_state, rewards, done, info


class BestResponseEnvMulti:
    """Single-agent view of N-agent game with other policies fixed."""

    def __init__(
        self,
        game_env: MultiAgentGameEnv,
        policies: List[Callable[[MultiGameState], ActionVector]],
        team_idx: int,
    ):
        self.game_env = game_env
        self.policies = policies
        self.team_idx = team_idx
        code = game_env.team_codes[team_idx]
        self.config = game_env.team_cfgs[code]

    def reset(self, seed: Optional[int] = None) -> MultiGameState:
        return self.game_env.reset(seed=seed)

    def valid_actions(self, game_state: MultiGameState) -> List[ActionVector]:
        return self.game_env.valid_actions(game_state, self.team_idx)

    def step(self, game_state: MultiGameState, action: ActionVector, rng=None):
        actions = []
        for idx, pol in enumerate(self.policies):
            if idx == self.team_idx:
                actions.append(action)
            else:
                actions.append(pol(game_state))
        next_state, rewards, done, info = self.game_env.step(game_state, actions, rng=rng)
        return next_state, rewards[self.team_idx], done, info
