import numpy as np
from dataclasses import dataclass, asdict

# Action Constants
ACT_ROSTER_TANK = 0
ACT_ROSTER_REBUILD = 1
ACT_ROSTER_HOLD = 2
ACT_ROSTER_BUY = 3
ACT_ROSTER_ALLIN = 4

ACT_SALARY_FLOOR = 0
ACT_SALARY_CAP = 1
ACT_SALARY_TAX = 2
ACT_SALARY_APRON = 3

ACT_TICKET_LOW = 0
ACT_TICKET_NORMAL = 1
ACT_TICKET_HIGH = 2
ACT_TICKET_VERYHIGH = 3
ACT_TICKET_MAX = 4

ACT_MARKETING_MIN = 0
ACT_MARKETING_NORMAL = 1
ACT_MARKETING_MAX = 2

ACT_DEBT_PAYDOWN = 0
ACT_DEBT_HOLD = 1
ACT_DEBT_BORROW = 2

ACT_EQUITY_0 = 0
ACT_EQUITY_1 = 1
ACT_EQUITY_2 = 2
ACT_EQUITY_5 = 3

# Phase Constants
PHASE_OFFSEASON = 0
PHASE_REGULAR = 1
PHASE_TRADE_DEADLINE = 2
PHASE_PLAYOFF = 3

@dataclass
class ActionVector:
    a_roster: int  # 0-4
    a_salary: int  # 0-3
    a_ticket: int  # 0-4
    a_marketing: int # 0-2
    a_debt: int    # 0-2
    a_equity: int  # 0-3

    def to_list(self):
        return [self.a_roster, self.a_salary, self.a_ticket, self.a_marketing, self.a_debt, self.a_equity]

# Environment Constants
MACRO_RECESSION = 0
MACRO_NORMAL = 1
MACRO_BOOM = 2

CAP_GROWTH_SMOOTH = 0
CAP_GROWTH_SPIKE = 1

@dataclass
class EnvState:
    macro: int # 0,1,2
    cap_growth: int # 0,1
    i_expansion: int # 0,1
    t_media_deal: int # 0-5
    mu_size: float # 0.5 - 2.0
    compete_local: int # 0 (Low), 1 (Mid), 2 (High)
    n_star_fa: int 
    bidding_intensity: float # Millions

@dataclass
class FinancialState:
    leverage: float # lambda
    cash_flow: float
    psi_mean_salary: float
    psi_std_salary: float
    psi_max_salary_ratio: float
    psi_guaranteed: float
    cap_space_avail: float
    tax_status: int # 0, 1, 2
    valuation_growth: float
    owner_share: float
    player_equity_pool: float
    franchise_value: float # FV_t
    debt_stock: float # D_t

