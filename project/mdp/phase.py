PHASE_OFFSEASON = "offseason"
PHASE_REGULAR = "regular"
PHASE_TRADE_DEADLINE = "trade_deadline"
PHASE_PLAYOFF = "playoff"

PHASE_ORDER = [
    PHASE_OFFSEASON,
    PHASE_REGULAR,
    PHASE_TRADE_DEADLINE,
    PHASE_PLAYOFF,
]

PHASE_TO_INDEX = {name: idx for idx, name in enumerate(PHASE_ORDER)}


def next_phase(current: str) -> tuple[str, bool]:
    """Return next phase and whether the season wraps to next year."""
    idx = PHASE_ORDER.index(current)
    next_idx = (idx + 1) % len(PHASE_ORDER)
    wraps = next_idx == 0
    return PHASE_ORDER[next_idx], wraps
