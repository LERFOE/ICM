from __future__ import annotations

from project.mdp.reward import terminal_value, risk_penalty


def heuristic_value(state, config) -> float:
    # Fast heuristic: combine terminal value, current CF, and win% with risk penalty
    cf_term = state.F.CF
    tv_term = terminal_value(state.F)
    win_term = 10.0 * (float(state.R.W[0]) - 0.5)
    penalty = risk_penalty(state.F, state.R, config)
    return 0.002 * tv_term + 0.5 * cf_term + win_term - penalty
