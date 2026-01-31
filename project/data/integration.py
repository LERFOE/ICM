from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

from project.data import loaders
from project.mdp.config import MDPConfig
from project.mdp.state import (
    CompetitiveState,
    EnvState,
    FinancialState,
    State,
    initial_competitive_state,
    initial_env_state,
    initial_financial_state,
)
from project.mdp.phase import PHASE_OFFSEASON


@dataclass
class CalibrationSummary:
    attendance_slope: float = 0.0
    attendance_intercept: float = 0.0
    attendance_method: str = "ols"
    rev_win_beta: float = 0.0
    ticket_elasticity: float = 0.0
    marketing_attendance_beta: float = 0.0
    elo_b0: float = 0.0
    elo_b1: float = 0.0
    elo_b2: float = 0.0
    elo_samples: int = 0
    market_size: float = 1.0
    base_gate_revenue: float = 0.0
    base_media_revenue: float = 0.0
    base_sponsor_revenue: float = 0.0
    franchise_value: float = 0.0
    revenue_total: float = 0.0
    gate_ratio: float = 0.55
    media_ratio: float = 0.30
    sponsor_ratio: float = 0.15
    notes: str = ""


def _fit_logit(X: np.ndarray, y: np.ndarray, lr: float = 0.2, steps: int = 2000) -> np.ndarray:
    beta = np.zeros(X.shape[1])
    for _ in range(steps):
        z = X @ beta
        p = 1.0 / (1.0 + np.exp(-z))
        grad = X.T @ (p - y) / len(y)
        beta -= lr * grad
    return beta


def _fit_lasso_line(x: np.ndarray, y: np.ndarray, alpha: float) -> Tuple[float, float]:
    # Standardize x/y for stable L1, then convert back to original scale.
    from sklearn.linear_model import Lasso

    x = x.reshape(-1, 1)
    x_mean = x.mean()
    x_std = x.std() if x.std() > 0 else 1.0
    y_mean = y.mean()
    y_std = y.std() if y.std() > 0 else 1.0

    x_s = (x - x_mean) / x_std
    y_s = (y - y_mean) / y_std

    model = Lasso(alpha=alpha, fit_intercept=True, max_iter=5000)
    model.fit(x_s, y_s)
    b = float(model.coef_[0])
    a = float(model.intercept_)

    # Convert back: y = slope * x + intercept
    slope = b * (y_std / x_std)
    intercept = y_mean + a * y_std - slope * x_mean
    return slope, intercept


def _safe_float(val, default=np.nan):
    if pd.isna(val):
        return default
    try:
        return float(val)
    except Exception:
        return default


def _latest_year(values) -> Optional[int]:
    if values is None:
        return None
    vals = [int(v) for v in values if not pd.isna(v)]
    return max(vals) if vals else None


def calibrate_config_from_data(config: Optional[MDPConfig] = None) -> Tuple[MDPConfig, CalibrationSummary]:
    cfg = config or MDPConfig()
    summary = CalibrationSummary()

    # Attendance vs Win% regression (OLS or Lasso)
    try:
        att = loaders.load_attendance()
        adv = loaders.load_advanced_stats()
        merged = att.merge(adv, left_on=["Season", "Team"], right_on=["Season", "Team"], how="inner")
        merged = merged.dropna(subset=["Avg_Attendance", "Win%"])
        if len(merged) >= 10:
            x = merged["Win%"].astype(float).values
            y = merged["Avg_Attendance"].astype(float).values
            if cfg.use_lasso:
                slope, intercept = _fit_lasso_line(x, y, cfg.lasso_alpha)
                summary.attendance_method = "lasso"
            else:
                slope, intercept = np.polyfit(x, y, 1)
                summary.attendance_method = "ols"
            summary.attendance_slope = float(slope)
            summary.attendance_intercept = float(intercept)
            base_att = intercept + slope * 0.5
            if base_att > 0:
                summary.rev_win_beta = float(slope / base_att)
                cfg.rev_win_beta = summary.rev_win_beta * cfg.rev_win_beta_scale

            residuals = y - (slope * x + intercept)
            resid_ratio = float(np.std(residuals) / max(np.mean(y), 1e-6))
            ticket_elasticity = float(np.clip(0.6 + 0.8 * resid_ratio, 0.4, 1.5))
            marketing_beta = float(np.clip(2.0 * resid_ratio, 0.2, 1.5))
            cfg.ticket_elasticity = ticket_elasticity
            cfg.marketing_attendance_beta = marketing_beta
            summary.ticket_elasticity = ticket_elasticity
            summary.marketing_attendance_beta = marketing_beta
        else:
            summary.notes += "Attendance regression sample too small; used default rev_win_beta. "
    except Exception as exc:
        summary.notes += f"Attendance regression failed: {exc}. "

    # Market size for Indiana from latest season attendance
    try:
        ind_att = att[att["Team"].str.contains("Indiana", na=False)]
        if not ind_att.empty:
            latest_season = _latest_year(ind_att["Season"].values)
            season_att = att[att["Season"] == latest_season]
            ind_row = ind_att[ind_att["Season"] == latest_season]
            if not season_att.empty and not ind_row.empty:
                league_avg = season_att["Avg_Attendance"].mean()
                ind_avg = ind_row["Avg_Attendance"].mean()
                if league_avg > 0:
                    mu = float(ind_avg / league_avg)
                    mu = max(0.5, min(2.0, mu))
                    cfg.market_size = mu
                    summary.market_size = mu
    except Exception as exc:
        summary.notes += f"Market size calc failed: {exc}. "

    # Franchise value and revenue
    try:
        vals = loaders.load_valuations()
        ind_vals = vals[vals["Team"].str.contains("Indiana", na=False)]
        if not ind_vals.empty:
            latest_year = _latest_year(ind_vals["Year"].values)
            row_latest = ind_vals[ind_vals["Year"] == latest_year].iloc[0]
            fv = _safe_float(row_latest.get("Valuation_M"), cfg.base_franchise_value)
            cfg.base_franchise_value = fv
            summary.franchise_value = fv

            # Prefer a year with revenue if available
            ind_with_rev = ind_vals.dropna(subset=["Revenue_M"])
            if not ind_with_rev.empty:
                rev_year = _latest_year(ind_with_rev["Year"].values)
                row_rev = ind_with_rev[ind_with_rev["Year"] == rev_year].iloc[0]
                rev_total = _safe_float(row_rev.get("Revenue_M"), np.nan)
                if not np.isnan(rev_total):
                    summary.revenue_total = rev_total
            else:
                rev_total = _safe_float(row_latest.get("Revenue_M"), np.nan)
                if not np.isnan(rev_total):
                    summary.revenue_total = rev_total
    except Exception as exc:
        summary.notes += f"Valuation load failed: {exc}. "

    # Revenue split assumptions
    gate_ratio = summary.gate_ratio
    media_ratio = summary.media_ratio
    sponsor_ratio = summary.sponsor_ratio

    if summary.revenue_total > 0:
        base_gate = summary.revenue_total * gate_ratio
        base_media = summary.revenue_total * media_ratio
        base_sponsor = summary.revenue_total * sponsor_ratio
        cfg.base_gate_revenue = base_gate
        cfg.base_media_revenue = base_media
        cfg.base_sponsor_revenue = base_sponsor
        summary.base_gate_revenue = base_gate
        summary.base_media_revenue = base_media
        summary.base_sponsor_revenue = base_sponsor
    else:
        # fallback gate revenue from attendance (assumed $30 average ticket)
        try:
            ind_att = att[att["Team"].str.contains("Indiana", na=False)]
            latest_season = _latest_year(ind_att["Season"].values)
            ind_row = ind_att[ind_att["Season"] == latest_season].iloc[0]
            avg_att = _safe_float(ind_row.get("Avg_Attendance"), 0.0)
            gp = _safe_float(ind_row.get("GP"), 20.0)
            ticket_price = 30.0
            base_gate = avg_att * gp * ticket_price / 1e6
            cfg.base_gate_revenue = base_gate
            cfg.base_media_revenue = base_gate * (media_ratio / gate_ratio)
            cfg.base_sponsor_revenue = base_gate * (sponsor_ratio / gate_ratio)
            summary.base_gate_revenue = base_gate
            summary.base_media_revenue = cfg.base_media_revenue
            summary.base_sponsor_revenue = cfg.base_sponsor_revenue
            summary.notes += "Revenue missing; used attendance*price fallback. "
        except Exception as exc:
            summary.notes += f"Gate revenue fallback failed: {exc}. "

    # Debt baseline (default 20% of FV)
    cfg.base_debt = 0.2 * cfg.base_franchise_value

    # ELO -> Win% logistic calibration
    try:
        elo = loaders.load_elo_game()
        elo = elo.dropna(subset=["ELO_t", "opp_pre_elo_t", "home_win"])
        if len(elo) >= 20:
            is_ind_home = elo["home_abbr"].astype(str).str.upper().eq("IND")
            y = np.where(is_ind_home, elo["home_win"].astype(float), 1.0 - elo["home_win"].astype(float))
            elo_diff = (elo["ELO_t"].astype(float) - elo["opp_pre_elo_t"].astype(float)) / 400.0
            sos = elo.get("SOS_t", pd.Series(np.ones(len(elo))))
            sos = sos.fillna(1.0).astype(float)
            X = np.column_stack([np.ones(len(elo_diff)), elo_diff.values, (sos.values - 1.0)])
            beta = _fit_logit(X, y, lr=0.3, steps=1500)
            summary.elo_b0 = float(beta[0])
            summary.elo_b1 = float(beta[1])
            summary.elo_b2 = float(beta[2])
            summary.elo_samples = int(len(elo))
            cfg.win_eta0 = summary.elo_b0
            cfg.win_eta1 = summary.elo_b1
            cfg.win_eta_sos = -summary.elo_b2
    except Exception as exc:
        summary.notes += f"ELO logit failed: {exc}. "

    return cfg, summary


def build_competitive_state_from_data(
    rng: np.random.Generator,
    year: Optional[int] = None,
    config: Optional[MDPConfig] = None,
) -> CompetitiveState:
    cfg = config or MDPConfig()
    # Defaults
    base = initial_competitive_state(MDPConfig(), rng)
    used_player_model = False

    # If player dataset exists, build lineup-based Q/C/P
    try:
        from project.data.player_kmeans import build_player_model, competitive_state_from_lineup

        model = build_player_model(roster_size=cfg.roster_size)
        roster_df = model.build_roster(team=cfg.team_code, roster_size=cfg.roster_size)
        if not roster_df.empty:
            base = model.compute_state_from_roster(roster_df, base)
            used_player_model = True
    except Exception:
        # fall back to team-level aggregates
        pass

    # Use season stats for W/L and ranks
    try:
        ind_stats = loaders.load_ind_team_stats()
        if year is None:
            year = _latest_year(ind_stats["Year"].values)
        row = ind_stats[ind_stats["Year"] == year]
        if not row.empty:
            win_pct = _safe_float(row.iloc[0].get("W/L%"), 0.5)
            finish = _safe_float(row.iloc[0].get("Finish"), 3.0)
            # Rank bin: top/mid/bottom
            rank_bin = 0.0 if finish <= 4 else 1.0 if finish <= 8 else 2.0
            base.W = np.array([win_pct, 0.0, rank_bin], dtype=float)
    except Exception:
        pass

    # ELO / O / SOS from season level
    try:
        elo_season = loaders.load_elo_season()
        if year is None:
            year = _latest_year(elo_season["Season"].values)
        row = elo_season[elo_season["Season"] == year]
        if not row.empty:
            base.ELO = _safe_float(row.iloc[0].get("ELO_last_pregame"), base.ELO)
            base.SOS = _safe_float(row.iloc[0].get("SOS_mean"), base.SOS)
            o_mean = _safe_float(row.iloc[0].get("O_mean_mean"), base.O[0])
            base.O = np.array([o_mean, base.O[1], base.O[2]], dtype=float)
    except Exception:
        pass

    # Q vector from league advanced stats (z-scores) if not using player model
    try:
        if used_player_model:
            raise RuntimeError("skip team-level Q; player model used")
        adv = loaders.load_advanced_stats()
        if year is None:
            year = _latest_year(adv["Season"].values)
        adv_year = adv[adv["Season"] == year].copy()
        ind_row = adv_year[adv_year["Team"].str.contains("Indiana", na=False)]
        if ind_row.empty:
            # fallback to latest Indiana season available in advanced stats
            ind_all = adv[adv["Team"].str.contains("Indiana", na=False)]
            if not ind_all.empty:
                fallback_year = _latest_year(ind_all["Season"].values)
                adv_year = adv[adv["Season"] == fallback_year].copy()
                ind_row = adv_year[adv_year["Team"].str.contains("Indiana", na=False)]
        if not ind_row.empty and not adv_year.empty:
            cols = ["ORtg", "DRtg", "NetRtg", "SRS"]
            means = adv_year[cols].mean()
            stds = adv_year[cols].replace(0, np.nan).std().fillna(1.0)
            vals = ind_row.iloc[0][cols]
            z_ortg = (vals["ORtg"] - means["ORtg"]) / stds["ORtg"]
            z_drtg = (vals["DRtg"] - means["DRtg"]) / stds["DRtg"]
            z_netrtg = (vals["NetRtg"] - means["NetRtg"]) / stds["NetRtg"]
            z_srs = (vals["SRS"] - means["SRS"]) / stds["SRS"]
            base.Q = np.array([z_ortg, -z_drtg, z_netrtg, z_srs], dtype=float)
    except Exception:
        pass

    # Roster structure from Spotrac (positions, ages, contract maturities)
    try:
        spot = loaders.load_spotrac()
        # Position mapping
        pos_counts = {"PG": 0.0, "SG": 0.0, "SF": 0.0, "PF": 0.0, "C": 0.0}
        for pos in spot["Pos"].fillna("").values:
            p = str(pos).upper().strip()
            if "G" in p and "F" in p:
                pos_counts["SG"] += 0.5
                pos_counts["SF"] += 0.5
            elif "G" in p:
                pos_counts["PG"] += 0.5
                pos_counts["SG"] += 0.5
            elif "F" in p:
                pos_counts["SF"] += 0.5
                pos_counts["PF"] += 0.5
            elif "C" in p:
                pos_counts["C"] += 1.0
        base.P = np.array(
            [pos_counts["PG"], pos_counts["SG"], pos_counts["SF"], pos_counts["PF"], pos_counts["C"]],
            dtype=float,
        )

        # Age profile
        ages = spot["Age"].dropna().astype(float).values
        if len(ages):
            base.A = np.array([np.mean(ages), np.var(ages), float(np.sum(ages >= 28))], dtype=float)

        # Contract maturity (based on salary columns)
        year_cols = [c for c in spot.columns if c.endswith("_salary_m")]
        # assume 2025 is current
        maturity = []
        for _, row in spot.iterrows():
            remaining = 0
            for col in year_cols:
                if not pd.isna(row[col]) and row[col] > 0:
                    remaining += 1
            maturity.append(remaining)
        l0 = sum(1 for m in maturity if m <= 1)
        l1 = sum(1 for m in maturity if m == 2)
        l2 = sum(1 for m in maturity if m >= 3)
        base.L = np.array([float(l0), float(l1), float(l2)], dtype=float)
    except Exception:
        pass

    return base


def build_financial_state_from_data(
    config: MDPConfig,
    K: list[int],
) -> FinancialState:
    F = initial_financial_state(config, K)

    # Salary structure from Spotrac (use 2025_salary_m if available)
    try:
        spot = loaders.load_spotrac()
        if "2025_salary_m" in spot.columns:
            salaries = spot["2025_salary_m"].dropna().astype(float)
            if len(salaries):
                total_payroll = float(salaries.sum())
                F.psi_mean_salary = total_payroll / max(1.0, config.roster_size)
                F.psi_std_salary = float(salaries.std()) if len(salaries) > 1 else 0.0
                F.psi_max_salary_ratio = float(salaries.max() / max(total_payroll, 1e-6))
                F.cap_space_avail = max(0.0, config.salary_cap - total_payroll)
                # approximate commitment ratio using multi-year contracts
                year_cols = [c for c in spot.columns if c.endswith("_salary_m")]
                maturity = []
                for _, row in spot.iterrows():
                    remaining = 0
                    for col in year_cols:
                        if not pd.isna(row[col]) and row[col] > 0:
                            remaining += 1
                    maturity.append(remaining)
                l2 = sum(1 for m in maturity if m >= 3)
                F.psi_commit = l2 / max(1.0, len(maturity))
    except Exception:
        pass

    return F


def build_env_state_from_data(config: MDPConfig) -> EnvState:
    return initial_env_state(config)


def build_initial_state_from_data(
    config: Optional[MDPConfig] = None,
    rng: Optional[np.random.Generator] = None,
    year: Optional[int] = None,
) -> State:
    cfg = config or MDPConfig()
    rng = rng or np.random.default_rng(42)
    K = [3, 2, 2, 1, 2, 0]
    R = build_competitive_state_from_data(rng, year=year, config=cfg)
    F = build_financial_state_from_data(cfg, K)
    E = build_env_state_from_data(cfg)
    return State(R=R, F=F, E=E, Theta=PHASE_OFFSEASON, K=K, year=cfg.start_year)
