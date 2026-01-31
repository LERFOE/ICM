from dataclasses import dataclass
from typing import List, Tuple

# Action ranges aligned with paper
ACTION_RANGES = {
    "roster": list(range(5)),   # 0..4
    "salary": list(range(4)),   # 0..3
    "ticket": list(range(5)),   # 0..4
    "marketing": list(range(3)),# 0..2
    "debt": list(range(3)),     # 0..2
    "equity": list(range(4)),   # 0..3
}

ACTION_LABELS = {
    "roster": [
        "seller_aggressive",
        "seller_conservative",
        "hold",
        "buyer_conservative",
        "buyer_aggressive",
    ],
    "salary": ["floor", "over_cap", "taxpayer", "apron"],
    "ticket": ["0.9x", "1.0x", "1.1x", "1.2x", "1.3x"],
    "marketing": ["low", "medium", "high"],
    "debt": ["deleverage", "maintain", "leverage_up"],
    "equity": ["0%", "1%", "2%", "5%"],
}


@dataclass(frozen=True)
class ActionVector:
    a_roster: int
    a_salary: int
    a_ticket: int
    a_marketing: int
    a_debt: int
    a_equity: int

    def to_list(self) -> List[int]:
        return [
            self.a_roster,
            self.a_salary,
            self.a_ticket,
            self.a_marketing,
            self.a_debt,
            self.a_equity,
        ]

    def to_tuple(self) -> Tuple[int, int, int, int, int, int]:
        return (
            self.a_roster,
            self.a_salary,
            self.a_ticket,
            self.a_marketing,
            self.a_debt,
            self.a_equity,
        )

    @staticmethod
    def from_list(values: List[int]) -> "ActionVector":
        return ActionVector(*values)


DEFAULT_ACTION = ActionVector(2, 1, 1, 1, 1, 0)
