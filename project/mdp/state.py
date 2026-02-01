from dataclasses import dataclass
from typing import List
import numpy as np

from project.mdp.config import MDPConfig
from project.mdp.phase import PHASE_OFFSEASON, PHASE_TO_INDEX


@dataclass
class CompetitiveState:
    Q: np.ndarray
    C: np.ndarray
    P: np.ndarray
    L: np.ndarray
    A: np.ndarray
    W: np.ndarray
    ELO: float
    Syn: float
    O: np.ndarray
    SOS: float
    roster_ids: List[int]

    def copy(self) -> "CompetitiveState":
        return CompetitiveState(
            Q=self.Q.copy(),
            C=self.C.copy(),
            P=self.P.copy(),
            L=self.L.copy(),
            A=self.A.copy(),
            W=self.W.copy(),
            ELO=float(self.ELO),
            Syn=float(self.Syn),
            O=self.O.copy(),
            SOS=float(self.SOS),
            roster_ids=list(self.roster_ids),
        )


@dataclass
class FinancialState:
    FV: float
    D: float
    leverage: float
    CF: float
    Cash: float
    psi_mean_salary: float
    psi_std_salary: float
    psi_max_salary_ratio: float
    psi_commit: float
    cap_space_avail: float
    tax_status: int
    valuation_growth: float
    owner_share: float
    K: List[int]

    def copy(self) -> "FinancialState":
        return FinancialState(
            FV=float(self.FV),
            D=float(self.D),
            leverage=float(self.leverage),
            CF=float(self.CF),
            Cash=float(self.Cash),
            psi_mean_salary=float(self.psi_mean_salary),
            psi_std_salary=float(self.psi_std_salary),
            psi_max_salary_ratio=float(self.psi_max_salary_ratio),
            psi_commit=float(self.psi_commit),
            cap_space_avail=float(self.cap_space_avail),
            tax_status=int(self.tax_status),
            valuation_growth=float(self.valuation_growth),
            owner_share=float(self.owner_share),
            K=list(self.K),
        )


@dataclass
class EnvState:
    macro: int
    cap_growth: int
    i_expansion: int
    t_media_deal: int
    mu_size: float
    compete_local: int
    n_star_fa: int
    bidding_intensity: float
    travel_fatigue: float

    def copy(self) -> "EnvState":
        return EnvState(
            macro=int(self.macro),
            cap_growth=int(self.cap_growth),
            i_expansion=int(self.i_expansion),
            t_media_deal=int(self.t_media_deal),
            mu_size=float(self.mu_size),
            compete_local=int(self.compete_local),
            n_star_fa=int(self.n_star_fa),
            bidding_intensity=float(self.bidding_intensity),
            travel_fatigue=float(self.travel_fatigue),
        )


@dataclass
class State:
    R: CompetitiveState
    F: FinancialState
    E: EnvState
    Theta: str
    K: List[int]
    year: int

    def copy(self) -> "State":
        return State(
            R=self.R.copy(),
            F=self.F.copy(),
            E=self.E.copy(),
            Theta=str(self.Theta),
            K=list(self.K),
            year=int(self.year),
        )

    def to_vector(self, config: MDPConfig) -> np.ndarray:
        # Scale features to stable ranges for RL
        r_parts = [
            self.R.Q,
            self.R.C / max(1.0, config.roster_size),
            self.R.P / max(1.0, config.roster_size),
            self.R.L / max(1.0, config.roster_size),
            np.array([self.R.A[0] / 40.0, self.R.A[1] / 10.0, self.R.A[2] / config.roster_size]),
            self.R.W,
            np.array([self.R.ELO / 2000.0, self.R.Syn / 5.0]),
            self.R.O / 2000.0,
            np.array([self.R.SOS / 2.0]),
        ]
        r_vec = np.concatenate([np.ravel(x) for x in r_parts])

        f_vec = np.array([
            self.F.FV / 200.0,
            self.F.D / 200.0,
            self.F.leverage,
            self.F.CF / config.cf_scale,
            self.F.Cash / 50.0,
            self.F.psi_mean_salary / 1.0,
            self.F.psi_std_salary / 1.0,
            self.F.psi_max_salary_ratio,
            self.F.psi_commit,
            self.F.cap_space_avail / 2.0,
            float(self.F.tax_status) / 2.0,
            self.F.valuation_growth,
            self.F.owner_share,
        ])

        e_vec = np.array([
            float(self.E.macro) / 2.0,
            float(self.E.cap_growth),
            float(self.E.i_expansion),
            float(self.E.t_media_deal) / max(1.0, config.media_cycle_years),
            self.E.mu_size / 2.0,
            float(self.E.compete_local) / 2.0,
            self.E.n_star_fa / 10.0,
            self.E.bidding_intensity / 20.0,
            float(self.E.travel_fatigue),
        ])

        phase_onehot = np.zeros(len(PHASE_TO_INDEX), dtype=float)
        phase_onehot[PHASE_TO_INDEX[self.Theta]] = 1.0

        k_vec = np.array(self.K, dtype=float) / np.array([6, 5, 6, 3, 4, 4], dtype=float)

        return np.concatenate([r_vec, f_vec, e_vec, phase_onehot, k_vec])


def initial_competitive_state(config: MDPConfig, rng: np.random.Generator) -> CompetitiveState:
    Q = rng.normal(0.0, 0.3, size=4)
    C = np.array([2, 2, 2, 2, 2, 2], dtype=float)
    P = np.array([2, 2, 2, 3, 3], dtype=float)
    L = np.array([2, 4, 6], dtype=float)
    A = np.array([27.0, 3.5, 5.0], dtype=float)
    W = np.array([0.5, 0.0, 1.0], dtype=float)
    ELO = 1500.0
    Syn = 0.0
    O = np.array([1500.0, 80.0, 5.0], dtype=float)
    SOS = 1.0
    return CompetitiveState(Q, C, P, L, A, W, ELO, Syn, O, SOS, roster_ids=[])


def initial_financial_state(config: MDPConfig, K: List[int]) -> FinancialState:
    FV = config.base_franchise_value
    D = config.base_debt
    leverage = D / max(FV, 1e-6)
    return FinancialState(
        FV=FV,
        D=D,
        leverage=leverage,
        CF=0.0,
        Cash=config.base_cash,
        psi_mean_salary=config.salary_cap / max(1, config.roster_size),
        psi_std_salary=0.05,
        psi_max_salary_ratio=0.25,
        psi_commit=0.5,
        cap_space_avail=0.0,
        tax_status=0,
        valuation_growth=0.0,
        owner_share=1.0,
        K=list(K),
    )


def initial_env_state(config: MDPConfig) -> EnvState:
    return EnvState(
        macro=1,
        cap_growth=0,
        i_expansion=0,
        t_media_deal=config.media_cycle_years,
        mu_size=config.market_size,
        compete_local=config.compete_local,
        n_star_fa=config.base_star_fa,
        bidding_intensity=config.base_bidding,
        travel_fatigue=0.0,
    )


def initial_state(config: MDPConfig, rng: np.random.Generator) -> State:
    K = [3, 2, 2, 1, 2, 0]
    R = initial_competitive_state(config, rng)
    F = initial_financial_state(config, K)
    E = initial_env_state(config)
    return State(R=R, F=F, E=E, Theta=PHASE_OFFSEASON, K=K, year=config.start_year)
