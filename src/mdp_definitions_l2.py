from dataclasses import dataclass

@dataclass
class MicroRosterAction:
    transaction_type: str  # 'TRADE', 'SIGN', 'CUT', 'DRAFT'
    target_archetype: str  # 'Superstar', 'Star_Guard', 'Rim_Protector', 'Shooter', 'Pick'
    asset_out: str         # 'Pick', 'CapSpace', 'Player_X'

class ActionVector:
    def __init__(self, a_roster, a_salary, a_ticket, a_marketing, a_debt, a_equity, micro_action=None):
        self.a_roster = a_roster
        self.a_salary = a_salary
        self.a_ticket = a_ticket
        self.a_marketing = a_marketing
        self.a_debt = a_debt
        self.a_equity = a_equity
        self.micro_action = micro_action # New Level 2 Component

    def to_list(self):
        return [self.a_roster, self.a_salary, self.a_ticket, self.a_marketing, 
                self.a_debt, self.a_equity]
