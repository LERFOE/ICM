from __future__ import annotations
import numpy as np

from project.mdp.config import MDPConfig
from project.mdp.state import CompetitiveState, FinancialState
from project.mdp.phase import PHASE_PLAYOFF


def risk_penalty(F: FinancialState, R: CompetitiveState, config: MDPConfig) -> float:
    penalty = 0.0
    if F.leverage > config.leverage_soft:
        penalty += config.leverage_soft_penalty * (F.leverage - config.leverage_soft) ** 2
    if F.leverage > config.leverage_hard:
        penalty += config.leverage_hard_penalty * (F.leverage - config.leverage_hard) ** 2
    penalty += config.tax_penalty * float(F.tax_status)
    if F.psi_commit > config.commit_threshold:
        penalty += config.commit_penalty * (F.psi_commit - config.commit_threshold)
    if R.Syn < 0:
        penalty += config.synergy_penalty * abs(R.Syn)
    return penalty


def compute_reward(
    F_next: FinancialState,
    R_next: CompetitiveState,
    Theta: str,
    config: MDPConfig,
) -> float:
    cf_term = config.w_cf * (F_next.CF / max(config.cf_scale, 1e-6))
    val_term = config.w_val * F_next.valuation_growth
    win_term = config.w_win if Theta == PHASE_PLAYOFF else 0.0
    win_pct_term = config.w_win_pct * (float(R_next.W[0]) - config.win_pct_baseline)
    penalty = risk_penalty(F_next, R_next, config)
    return cf_term + val_term + win_term + win_pct_term - penalty


def terminal_value(F: FinancialState) -> float:
    return F.owner_share * (F.FV - F.D)
