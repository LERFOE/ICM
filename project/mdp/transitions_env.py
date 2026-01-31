from __future__ import annotations
from typing import Optional
import numpy as np

from project.mdp.config import MDPConfig
from project.mdp.state import EnvState

CAP_GROWTH_SMOOTH = 0
CAP_GROWTH_SPIKE = 1


def env_transition(
    E: EnvState,
    year: int,
    rng: np.random.Generator,
    config: MDPConfig,
    expansion_market_delta: Optional[float] = None,
    expansion_compete_delta: Optional[int] = None,
    expansion_media_bonus: Optional[float] = None,
) -> EnvState:
    E_next = E.copy()

    # Macro Markov chain
    probs = config.macro_transition[E.macro]
    E_next.macro = int(rng.choice([0, 1, 2], p=probs))

    # Media deal countdown
    if E_next.t_media_deal > 0:
        E_next.t_media_deal -= 1
    else:
        E_next.t_media_deal = config.media_cycle_years

    # Cap growth regime
    E_next.cap_growth = CAP_GROWTH_SPIKE if E_next.t_media_deal == 0 else CAP_GROWTH_SMOOTH

    # Expansion trigger
    E_next.i_expansion = 1 if year in config.expansion_years else 0

    # Market and FA adjustments when expansion happens
    if E_next.i_expansion == 1:
        m_delta = expansion_market_delta if expansion_market_delta is not None else config.expansion_market_delta
        c_delta = expansion_compete_delta if expansion_compete_delta is not None else config.expansion_compete_delta
        E_next.mu_size = max(0.5, min(2.0, E_next.mu_size + m_delta))
        E_next.compete_local = max(0, min(2, E_next.compete_local + c_delta))
        E_next.n_star_fa = max(0, E_next.n_star_fa + config.expansion_star_fa_delta)
        E_next.bidding_intensity = max(0.0, E_next.bidding_intensity + config.expansion_bidding_delta)
        if expansion_media_bonus is not None:
            # apply a temporary macro-like lift via media bonus proxy
            E_next.bidding_intensity += expansion_media_bonus * 10.0

    # Exogenous FA market noise
    E_next.n_star_fa = max(0, int(E_next.n_star_fa + rng.normal(0, 1)))
    E_next.bidding_intensity = max(0.0, E_next.bidding_intensity + rng.normal(0, 0.5))

    return E_next
