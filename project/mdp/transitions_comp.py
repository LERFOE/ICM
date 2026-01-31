from __future__ import annotations
from typing import Dict, Tuple
import numpy as np

from project.mdp.action import ActionVector
from project.mdp.config import MDPConfig
from project.mdp.phase import PHASE_OFFSEASON, PHASE_PLAYOFF, PHASE_REGULAR, PHASE_TRADE_DEADLINE
from project.mdp.state import CompetitiveState, EnvState


def _clamp_array(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def roster_update(
    R: CompetitiveState,
    action: ActionVector,
    E: EnvState,
    rng: np.random.Generator,
    config: MDPConfig,
    player_model=None,
) -> CompetitiveState:
    if player_model is not None:
        return player_model.roster_update(R, action, rng, config)

    R_next = R.copy()

    # Skill delta influenced by roster and salary actions
    base_delta = config.roster_delta[action.a_roster]
    salary_scale = config.salary_factor[action.a_salary]
    fa_bonus = 1.0 + 0.05 * max(0, E.n_star_fa - 2)
    bidding_penalty = 1.0 - 0.02 * max(0.0, E.bidding_intensity - 5.0)
    delta = base_delta * salary_scale * fa_bonus * bidding_penalty

    noise = rng.normal(0.0, 0.05, size=R_next.Q.shape)
    delta_vec = np.array([delta, delta * 0.9, delta * 0.8, delta * 0.85])
    R_next.Q = _clamp_array(R_next.Q + delta_vec + noise, -3.0, 3.0)

    # Archetype mix: shift one slot toward development or stars
    C = R_next.C.copy()
    if action.a_roster <= 2:
        # move one count to developmental archetype (index 0)
        idx_from = int(rng.integers(1, len(C)))
        if C[idx_from] > 0:
            C[idx_from] -= 1
            C[0] += 1
    elif action.a_roster >= 4:
        idx_from = int(rng.integers(0, len(C)))
        if C[idx_from] > 0:
            C[idx_from] -= 1
            C[-1] += 1
    R_next.C = C

    # Position balance: slight nudges but keep totals
    P = R_next.P.copy()
    if action.a_roster >= 6:
        # bias to wings/forwards
        P[2] += 1
        P[3] += 1
        P[0] = max(1, P[0] - 1)
        P[4] = max(1, P[4] - 1)
    elif action.a_roster == 0:
        P[0] += 1
        P[4] += 1
        P[2] = max(1, P[2] - 1)
        P[3] = max(1, P[3] - 1)
    R_next.P = P

    # Contract maturity buckets
    L = R_next.L.copy()
    if action.a_roster <= 2:
        # more expirings
        shift = min(1, L[2])
        L[2] -= shift
        L[0] += shift
    elif action.a_roster >= 4:
        shift = min(1, L[0])
        L[0] -= shift
        L[2] += shift
    R_next.L = L

    # Age profile
    A = R_next.A.copy()
    if action.a_roster <= 2:
        A[0] = max(23.0, A[0] - 0.6)
        A[2] = max(0.0, A[2] - 1.0)
    elif action.a_roster >= 4:
        A[0] = min(32.0, A[0] + 0.6)
        A[2] = min(12.0, A[2] + 1.0)
    A[1] = max(0.5, A[1] + rng.normal(0, 0.2))
    R_next.A = A

    # Synergy penalty on large roster changes
    change_intensity = abs(base_delta)
    R_next.Syn -= config.syn_penalty * change_intensity
    return R_next


def roll_forward_contract_and_age(
    R: CompetitiveState,
    rng: np.random.Generator,
    config: MDPConfig,
) -> CompetitiveState:
    R_next = R.copy()
    # Age progresses smoothly (quarter season)
    R_next.A[0] += 0.25
    if R_next.A[0] > 28.0:
        R_next.A[2] = min(config.roster_size, R_next.A[2] + 0.2)

    # Contracts: small drift toward shorter maturity
    L = R_next.L.copy()
    decay = min(0.2, L[2])
    L[2] -= decay
    L[1] += decay * 0.6
    L[0] += decay * 0.4
    R_next.L = L

    # Synergy recovers slowly toward 0
    R_next.Syn += config.syn_recovery * (0.0 - R_next.Syn)
    return R_next


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def game_sim_update(
    R: CompetitiveState,
    E: EnvState,
    rng: np.random.Generator,
    config: MDPConfig,
    player_model=None,
    team_code: str | None = None,
) -> Tuple[CompetitiveState, Dict[str, float]]:
    R_next = R.copy()
    info: Dict[str, float] = {}

    # Sample opponent from real-player league context if available
    if player_model is not None:
        opp = player_model.sample_opponent(team_code, rng)
        R_next.O[0] = float(opp.get("elo", 1500.0))
        R_next.O[1] = float(opp.get("pace", 80.0))
        R_next.O[2] = float(opp.get("stars", 5.0))
        R_next.SOS = np.clip(1.0 + (R_next.O[0] - 1500.0) / 4000.0, 0.85, 1.15)

    elo_diff = (R_next.ELO - R_next.O[0]) / 400.0
    skill_term = np.tanh(float(np.mean(R_next.Q)))
    base = (
        config.win_eta0
        + config.win_eta1 * elo_diff
        - config.win_eta_sos * (R_next.SOS - 1.0)
        + config.win_eta2 * R_next.Syn
        + config.win_eta3 * skill_term
    )
    win_prob = _sigmoid(base)
    win_pct = float(np.clip(rng.normal(win_prob, config.win_noise), 0.05, 0.95))

    # Injury shock
    if rng.random() < config.injury_prob:
        severity = rng.uniform(*config.injury_severity_range)
        R_next.Q = R_next.Q * (1.0 - severity)
        R_next.Syn -= severity * config.injury_syn_penalty
        win_pct = max(0.05, win_pct - 0.1 * severity)
        info["injury"] = severity

    # Update W
    R_next.W[0] = 0.7 * R_next.W[0] + 0.3 * win_pct
    if win_pct > 0.55:
        R_next.W[1] = min(10.0, R_next.W[1] + 1.0)
    elif win_pct < 0.45:
        R_next.W[1] = max(-10.0, R_next.W[1] - 1.0)
    else:
        R_next.W[1] *= 0.5
    if win_pct > 0.6:
        R_next.W[2] = 0.0
    elif win_pct < 0.4:
        R_next.W[2] = 2.0
    else:
        R_next.W[2] = 1.0

    # Update ELO
    expected = _sigmoid(config.elo_scale * elo_diff)
    R_next.ELO = R_next.ELO + config.elo_k * (win_pct - expected)

    # Synergy recovery
    R_next.Syn += config.syn_recovery * (0.0 - R_next.Syn) + rng.normal(0.0, 0.05)

    # Opponent stats and SOS drift (fallback to synthetic if no player model)
    if player_model is None:
        R_next.O[0] = np.clip(rng.normal(1500.0, 50.0), 1400.0, 1600.0)
        R_next.O[1] = np.clip(rng.normal(80.0, 10.0), 40.0, 120.0)
        R_next.O[2] = max(0.0, rng.normal(5.0, 2.0))
        R_next.SOS = np.clip(1.0 + (R_next.O[0] - 1500.0) / 4000.0, 0.85, 1.15)

    info["win_pct"] = win_pct
    return R_next, info


def comp_transition(
    R: CompetitiveState,
    action: ActionVector,
    Theta: str,
    E_next: EnvState,
    rng: np.random.Generator,
    config: MDPConfig,
    player_model=None,
    team_code: str | None = None,
) -> Tuple[CompetitiveState, Dict[str, float]]:
    info: Dict[str, float] = {}
    if Theta in (PHASE_OFFSEASON, PHASE_TRADE_DEADLINE):
        R_mid = roster_update(R, action, E_next, rng, config, player_model=player_model)
    else:
        R_mid = roll_forward_contract_and_age(R, rng, config)

    if Theta in (PHASE_REGULAR, PHASE_TRADE_DEADLINE, PHASE_PLAYOFF):
        R_next, game_info = game_sim_update(
            R_mid, E_next, rng, config, player_model=player_model, team_code=team_code
        )
        info.update(game_info)
    else:
        R_next = R_mid

    return R_next, info
