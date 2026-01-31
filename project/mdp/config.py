from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class MDPConfig:
    # Time and horizon
    start_year: int = 2025
    horizon_years: int = 5

    # Roster and league scale (all monetary units in millions USD)
    roster_size: int = 12
    salary_cap: float = 1.5
    base_franchise_value: float = 90.0
    base_debt: float = 18.0
    base_cash: float = 5.0

    # Revenue components
    base_gate_revenue: float = 25.0
    base_media_revenue: float = 15.0
    base_sponsor_revenue: float = 8.0
    rev_win_beta: float = 0.6
    rev_star_beta: float = 0.3
    sponsor_beta: float = 0.5

    # Cost components
    fixed_ops_cost: float = 4.0
    interest_rate: float = 0.06
    leverage_interest_spread: float = 0.04

    # Ticket, marketing, equity mappings
    ticket_multipliers: List[float] = field(default_factory=lambda: [0.9, 1.0, 1.1, 1.2, 1.3])
    marketing_rates: List[float] = field(default_factory=lambda: [0.0, 0.05, 0.10])
    equity_rates: List[float] = field(default_factory=lambda: [0.0, 0.01, 0.02, 0.05])

    # Salary spending multipliers (relative to cap)
    salary_multipliers: List[float] = field(default_factory=lambda: [0.9, 1.05, 1.2, 1.35])
    tax_line_multiplier: float = 1.15
    apron_multiplier: float = 1.30
    tax_rate: float = 0.5

    # Debt actions (ratio to FV)
    debt_delta_ratio: List[float] = field(default_factory=lambda: [-0.05, 0.0, 0.08])

    # Valuation growth
    base_growth: float = 0.03
    growth_win_beta: float = 0.08
    growth_marketing_beta: float = 0.10
    growth_macro: Dict[int, float] = field(default_factory=lambda: {0: -0.02, 1: 0.0, 2: 0.02})
    media_spike_growth: float = 0.20
    growth_clip: float = 0.5

    # Macro environment
    macro_transition: List[List[float]] = field(
        default_factory=lambda: [
            [0.70, 0.25, 0.05],
            [0.15, 0.70, 0.15],
            [0.05, 0.30, 0.65],
        ]
    )
    macro_revenue_factor: Dict[int, float] = field(default_factory=lambda: {0: 0.85, 1: 1.0, 2: 1.15})
    macro_interest_spread: Dict[int, float] = field(default_factory=lambda: {0: 0.02, 1: 0.0, 2: -0.01})

    # Media deal cycle
    media_cycle_years: int = 5

    # Expansion
    expansion_years: List[int] = field(default_factory=lambda: [2026, 2028])
    expansion_market_delta: float = -0.05
    expansion_compete_delta: int = 1
    expansion_star_fa_delta: int = -2
    expansion_bidding_delta: float = 2.0
    expansion_media_bonus: float = 0.03

    # Base market profile (team-specific)
    market_size: float = 0.6
    compete_local: int = 0
    base_star_fa: int = 3
    base_bidding: float = 5.0

    # Competitive dynamics
    elo_k: float = 20.0
    elo_scale: float = 1.0
    win_eta0: float = 0.0
    win_eta1: float = 1.1
    win_eta2: float = 0.6
    win_eta3: float = 0.4
    win_eta_sos: float = 0.8
    win_noise: float = 0.05
    syn_recovery: float = 0.2

    # Roster update
    roster_delta: List[float] = field(default_factory=lambda: [-0.6, -0.25, 0.0, 0.25, 0.6])
    salary_factor: List[float] = field(default_factory=lambda: [0.6, 0.9, 1.1, 1.3])
    syn_penalty: float = 0.6
    injury_prob: float = 0.05
    injury_severity_range: List[float] = field(default_factory=lambda: [0.1, 0.3])
    injury_syn_penalty: float = 0.6

    # Reward weights
    w_cf: float = 1.0
    w_val: float = 0.6
    w_win: float = 0.2
    cf_scale: float = 5.0

    # Risk penalties
    leverage_soft: float = 0.35
    leverage_hard: float = 0.60
    leverage_soft_penalty: float = 1.0
    leverage_hard_penalty: float = 4.0
    tax_penalty: float = 0.5
    commit_threshold: float = 0.55
    commit_penalty: float = 0.8
    synergy_penalty: float = 0.3

    # Termination
    max_leverage: float = 0.85
    bankruptcy_cash: float = -10.0
    terminal_weight: float = 1.0
