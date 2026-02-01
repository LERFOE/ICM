from __future__ import annotations
import numpy as np

from project.mdp.action import ActionVector
from project.mdp.config import MDPConfig
from project.mdp.state import CompetitiveState, EnvState, FinancialState
from project.mdp.reward import compute_reward


def _salary_commit_ratio(R: CompetitiveState, config: MDPConfig) -> float:
    # Use long-term contract bucket share as a proxy for rigidity
    total = max(1.0, np.sum(R.L))
    return float(R.L[2] / total)


def fin_transition_and_reward(
    F: FinancialState,
    R_next: CompetitiveState,
    E_next: EnvState,
    action: ActionVector,
    Theta: str,
    rng: np.random.Generator,
    config: MDPConfig,
) -> tuple[FinancialState, float]:
    F_next = F.copy()

    win_pct = float(R_next.W[0])

    # Revenue components
    ticket_mult = config.ticket_multipliers[action.a_ticket]
    marketing_rate = config.marketing_rates[action.a_marketing]
    equity_rate = config.equity_rates[action.a_equity]

    performance_factor = 1.0 + config.rev_win_beta * (win_pct - 0.5)
    star_factor = 1.0 + config.rev_star_beta * np.tanh(float(np.mean(R_next.Q)))
    macro_factor = config.macro_revenue_factor[E_next.macro]
    # Local competition dilutes demand (e.g., new team in region competes for attention).
    # Keep the channel explicit so Q3 can trace "Δ_compete -> revenue/CF" quantitatively.
    compete_factor = float(np.clip(1.0 - config.compete_revenue_beta * float(E_next.compete_local), 0.70, 1.00))
    market_factor = E_next.mu_size * compete_factor

    # Ticket price elasticity: Revenue scales as P^(1 - e)
    ticket_revenue_mult = ticket_mult ** (1.0 - config.ticket_elasticity)
    # Marketing lift on attendance
    marketing_att_mult = 1.0 + config.marketing_attendance_beta * marketing_rate

    gate_revenue = (
        config.base_gate_revenue
        * performance_factor
        * star_factor
        * macro_factor
        * market_factor
        * ticket_revenue_mult
        * marketing_att_mult
    )

    sponsor_revenue = config.base_sponsor_revenue * (1.0 + config.sponsor_beta * marketing_rate)

    media_revenue = config.base_media_revenue * macro_factor
    # Apply a media-lift either at the renewal event or in expansion years (Q3).
    if E_next.t_media_deal == 0 or E_next.i_expansion == 1:
        media_revenue *= 1.0 + config.expansion_media_bonus

    total_revenue = gate_revenue + sponsor_revenue + media_revenue

    # Payroll
    payroll = config.salary_cap * config.salary_multipliers[action.a_salary]
    if action.a_roster >= 4:
        payroll *= 1.1
    elif action.a_roster <= 2:
        payroll *= 0.95
    # Bidding intensity raises the effective cost of acquiring/retaining talent.
    payroll *= 1.0 + config.bidding_payroll_beta * max(0.0, float(E_next.bidding_intensity - config.base_bidding))

    payroll_cash = payroll * (1.0 - equity_rate)

    # Tax status
    tax_line = config.salary_cap * config.tax_line_multiplier
    apron_line = config.salary_cap * config.apron_multiplier
    if payroll <= config.salary_cap:
        tax_status = 0
    elif payroll <= tax_line:
        tax_status = 1
    elif payroll <= apron_line:
        tax_status = 2
    else:
        tax_status = 2
    tax_bill = max(0.0, payroll - tax_line) * config.tax_rate

    # Interest
    interest_rate = config.interest_rate + config.macro_interest_spread[E_next.macro]
    interest_rate += max(0.0, F_next.leverage - config.leverage_soft) * config.leverage_interest_spread
    interest = F_next.D * max(0.0, interest_rate)

    # Marketing cost
    marketing_cost = marketing_rate * total_revenue

    fatigue = float(E_next.travel_fatigue) if E_next.i_expansion == 1 else 0.0
    travel_cost = config.travel_cost_beta * fatigue

    # Cash flow (CFO)
    CF = total_revenue - (
        payroll_cash + config.fixed_ops_cost + marketing_cost + tax_bill + interest + travel_cost
    )

    # Debt change
    debt_delta = config.debt_delta_ratio[action.a_debt] * F_next.FV
    if F_next.D + debt_delta < 0:
        debt_delta = -F_next.D
    D_next = F_next.D + debt_delta

    # Cash reserve update (include financing)
    Cash_next = F_next.Cash + CF + debt_delta

    # Owner share dilution
    owner_share_next = F_next.owner_share * (1.0 - equity_rate)

    # Valuation growth
    V_t = (
        config.base_growth
        + config.growth_win_beta * (win_pct - 0.5)
        + config.growth_marketing_beta * marketing_rate
        + config.growth_macro[E_next.macro]
    )
    if E_next.t_media_deal == 0:
        V_t += config.media_spike_growth
    if E_next.i_expansion == 1:
        # A smaller lift than a full media-cycle spike: expansion tends to increase attention,
        # but the long-run effect is uncertain and should be treated conservatively.
        V_t += 0.5 * config.expansion_media_bonus
    V_t = float(np.clip(V_t, -config.growth_clip, config.growth_clip))

    FV_next = F_next.FV * (1.0 + V_t)
    leverage_next = D_next / max(FV_next, 1e-6)

    # Update salary structure
    psi_mean = payroll / max(1.0, config.roster_size)
    psi_std = 0.2 * psi_mean
    psi_max_ratio = 0.25 + 0.05 * action.a_salary
    psi_commit = _salary_commit_ratio(R_next, config)
    cap_space = max(0.0, config.salary_cap - payroll)

    F_next.FV = FV_next
    F_next.D = D_next
    F_next.leverage = leverage_next
    F_next.CF = CF
    F_next.Cash = Cash_next
    F_next.psi_mean_salary = psi_mean
    F_next.psi_std_salary = psi_std
    F_next.psi_max_salary_ratio = psi_max_ratio
    F_next.psi_commit = psi_commit
    F_next.cap_space_avail = cap_space
    F_next.tax_status = tax_status
    F_next.valuation_growth = V_t
    F_next.owner_share = owner_share_next

    reward = compute_reward(F_next, R_next, Theta, config)
    if debt_delta > 0:
        reward -= config.debt_issue_penalty * (debt_delta / max(F_next.FV, 1e-6))
    return F_next, reward
