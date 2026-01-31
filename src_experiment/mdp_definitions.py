# Action Constants
ACT_ROSTER_HOLD = 0
ACT_ROSTER_BUY = 1
ACT_ROSTER_SELL = 2
ACT_ROSTER_REBUILD = 3
ACT_ROSTER_ALLIN = 4

ACT_SALARY_FLOOR = 0
ACT_SALARY_OVERCAP = 1
ACT_SALARY_TAX = 2
ACT_SALARY_APRON = 3

ACT_TICKET_DISCOUNT = 0
ACT_TICKET_NORMAL = 1
ACT_TICKET_PREMIUM = 2

ACT_MARKETING_MIN = 0
ACT_MARKETING_NORMAL = 1
ACT_MARKETING_HIGH = 2

ACT_DEBT_PAYDOWN = 0
ACT_DEBT_HOLD = 1
ACT_DEBT_BORROW = 2

ACT_EQUITY_0 = 0
ACT_EQUITY_1 = 1
ACT_EQUITY_2 = 2
ACT_EQUITY_5 = 3

# Phases
PHASE_OFFSEASON = 'Offseason'
PHASE_REGULAR = 'Regular'
PHASE_TRADE_DEADLINE = 'TradeDeadline'
PHASE_PLAYOFF = 'Playoff'

class ActionVector:
    def __init__(self, a_roster, a_salary, a_ticket, a_marketing, a_debt, a_equity, micro_action=None):
        self.a_roster = a_roster
        self.a_salary = a_salary
        self.a_ticket = a_ticket
        self.a_marketing = a_marketing
        self.a_debt = a_debt
        self.a_equity = a_equity
        self.micro_action = micro_action

    def __repr__(self):
        return f"Act(R={self.a_roster}, S={self.a_salary}, Micro={self.micro_action})"
