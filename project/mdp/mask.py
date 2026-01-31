from itertools import product
from typing import Iterable, List

from project.mdp.action import ACTION_RANGES, ActionVector
from project.mdp.phase import (
    PHASE_OFFSEASON,
    PHASE_PLAYOFF,
    PHASE_REGULAR,
    PHASE_TRADE_DEADLINE,
)

# Mutable table aligned with paper's action constraint table
# Order: roster, salary, ticket, marketing, debt, equity
MUTABLE = {
    PHASE_OFFSEASON: [1, 1, 1, 1, 1, 1],
    PHASE_REGULAR: [0, 0, 0, 1, 0, 1],
    PHASE_TRADE_DEADLINE: [1, 1, 0, 1, 0, 0],
    PHASE_PLAYOFF: [0, 0, 0, 1, 0, 0],
}


def mutable_mask(phase: str) -> List[int]:
    return MUTABLE[phase]


def roll_K(phase: str, K: List[int], action: ActionVector) -> List[int]:
    mask = mutable_mask(phase)
    a = action.to_list()
    return [a[i] if mask[i] == 1 else K[i] for i in range(6)]


def is_action_valid(phase: str, K: List[int], action: ActionVector) -> bool:
    mask = mutable_mask(phase)
    a = action.to_list()
    for i in range(6):
        if mask[i] == 0 and a[i] != K[i]:
            return False
    return True


def action_space_per_dim(phase: str, K: List[int]) -> List[List[int]]:
    mask = mutable_mask(phase)
    ranges = [
        ACTION_RANGES["roster"],
        ACTION_RANGES["salary"],
        ACTION_RANGES["ticket"],
        ACTION_RANGES["marketing"],
        ACTION_RANGES["debt"],
        ACTION_RANGES["equity"],
    ]
    allowed = []
    for i in range(6):
        if mask[i] == 1:
            allowed.append(list(ranges[i]))
        else:
            allowed.append([K[i]])
    return allowed


def enumerate_valid_actions(phase: str, K: List[int]) -> Iterable[ActionVector]:
    allowed = action_space_per_dim(phase, K)
    for values in product(*allowed):
        yield ActionVector(*values)
