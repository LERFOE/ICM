from dataclasses import dataclass
from typing import List, Tuple

# Action ranges aligned with paper
ACTION_RANGES = {
    "roster": list(range(7)),   # 0..6
    "salary": list(range(6)),   # 0..5
    "ticket": list(range(7)),   # 0..6
    "marketing": list(range(4)),# 0..3
    "debt": list(range(5)),     # 0..4
    "equity": list(range(5)),   # 0..4
}

ACTION_LABELS = {
    "roster": [
        "seller_aggressive",
        "seller",
        "seller_light",
        "hold",
        "buyer_light",
        "buyer",
        "buyer_aggressive",
    ],
    "salary": ["floor", "low", "over_cap", "taxpayer", "apron", "max_apron"],
    "ticket": ["0.85x", "0.95x", "1.0x", "1.05x", "1.10x", "1.20x", "1.30x"],
    "marketing": ["low", "mid_low", "mid_high", "high"],
    "debt": ["deleverage_high", "deleverage", "maintain", "leverage", "leverage_high"],
    "equity": ["0%", "0.5%", "1%", "2%", "3%"],
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


DEFAULT_ACTION = ActionVector(3, 2, 2, 1, 2, 0)
