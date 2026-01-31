from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import MultiTaskLassoCV, RidgeCV, Ridge
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.data.player_kmeans import FEATURES


OUTPUT_JSON = Path("project/data/skill_weights.json")
OUTPUT_MD = Path("project/data/skill_weights_report.md")

TEAM_CODE_TO_NAME = {
    "ATL": "Atlanta Dream",
    "CHI": "Chicago Sky",
    "CON": "Connecticut Sun",
    "DAL": "Dallas Wings",
    "IND": "Indiana Fever",
    "LVA": "Las Vegas Aces",
    "LAS": "Los Angeles Sparks",
    "MIN": "Minnesota Lynx",
    "NYL": "New York Liberty",
    "PHO": "Phoenix Mercury",
    "SEA": "Seattle Storm",
    "WAS": "Washington Mystics",
    "GSV": "Golden State Valkyries",
}
TEAM_NAME_TO_CODE = {v: k for k, v in TEAM_CODE_TO_NAME.items()}


def _zscore(series: pd.Series) -> pd.Series:
    mu = series.mean()
    std = series.std(ddof=0)
    if std == 0:
        std = 1.0
    return (series - mu) / std


def load_player_data() -> pd.DataFrame:
    df = pd.read_csv("allplayers.csv")
    df["DWS_40"] = df["DWS"].astype(float) / df["MP"].replace(0, np.nan).astype(float) * 40.0
    df = df.dropna(subset=FEATURES + ["MP", "Team"]).copy()
    return df


def compute_team_features(df: pd.DataFrame) -> Tuple[List[str], np.ndarray, pd.DataFrame]:
    z_df = df.copy()
    for f in FEATURES:
        z_df[f + "_z"] = _zscore(z_df[f].astype(float))
    z_df["mp_w"] = z_df["MP"].astype(float) / z_df.groupby("Team")["MP"].transform("sum").astype(float)
    teams = sorted(z_df["Team"].unique().tolist())
    X = []
    for team in teams:
        g = z_df[z_df["Team"] == team]
        w = g["mp_w"].values.reshape(-1, 1)
        feats = g[[f + "_z" for f in FEATURES]].values
        X.append((feats * w).sum(axis=0))
    return teams, np.vstack(X), z_df


def compute_elo_proxy(adv: pd.DataFrame, elo: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float]]:
    ind_adv = adv[adv["Team"] == "Indiana Fever"].copy()
    merged = ind_adv.merge(elo, left_on="Season", right_on="Season", how="inner")
    merged = merged.dropna(subset=["NetRtg", "Win%", "ELO_last_pregame"])
    X = merged[["NetRtg", "Win%"]].values.astype(float)
    y = merged["ELO_last_pregame"].values.astype(float)
    if len(merged) < 3:
        # fallback if insufficient data
        coef = np.array([20.0, 50.0], dtype=float)
        intercept = 1500.0
    else:
        ridge = Ridge(alpha=1.0).fit(X, y)
        coef = ridge.coef_
        intercept = ridge.intercept_
    return {"intercept": float(intercept), "coef_net": float(coef[0]), "coef_win": float(coef[1])}


def build_targets(teams: List[str], adv: pd.DataFrame, elo_proxy_params: Dict[str, float]) -> Tuple[np.ndarray, List[str], int]:
    target_season = int(adv["Season"].max())
    adv_season = adv[adv["Season"] == target_season].copy()
    adv_season["TeamCode"] = adv_season["Team"].map(TEAM_NAME_TO_CODE)
    adv_season = adv_season.dropna(subset=["TeamCode"])
    adv_season = adv_season.set_index("TeamCode")

    y_rows = []
    kept = []
    for team in teams:
        if team not in adv_season.index:
            continue
        row = adv_season.loc[team]
        win = float(row["Win%"])
        net = float(row["NetRtg"])
        elo_proxy = (
            elo_proxy_params["intercept"]
            + elo_proxy_params["coef_net"] * net
            + elo_proxy_params["coef_win"] * win
        )
        y_rows.append([win, net, elo_proxy])
        kept.append(team)
    return np.array(y_rows, dtype=float), kept, target_season


def fit_weights(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, dict]:
    # standardize Y
    Y_mean = Y.mean(axis=0)
    Y_std = Y.std(axis=0, ddof=0)
    Y_std = np.where(Y_std == 0, 1.0, Y_std)
    Yz = (Y - Y_mean) / Y_std

    # RidgeCV
    ridge = RidgeCV(alphas=[0.1, 0.5, 1.0, 2.0, 5.0], cv=min(5, len(X)))
    ridge.fit(X, Yz)
    ridge_cv_mse = float(mean_squared_error(Yz, ridge.predict(X)))

    # MultiTaskLassoCV
    lasso = MultiTaskLassoCV(alphas=[0.01, 0.05, 0.1, 0.2, 0.5], cv=min(5, len(X)), max_iter=5000)
    lasso.fit(X, Yz)
    lasso_cv_mse = float(mean_squared_error(Yz, lasso.predict(X)))

    if lasso_cv_mse <= ridge_cv_mse * 1.02:
        model = lasso
        model_name = "multi_task_lasso"
        cv_mse = lasso_cv_mse
    else:
        model = ridge
        model_name = "ridge"
        cv_mse = ridge_cv_mse

    coef = model.coef_
    weights = coef.mean(axis=0)
    # normalize by L1
    if np.sum(np.abs(weights)) > 0:
        weights = weights / np.sum(np.abs(weights))
    info = {
        "model": model_name,
        "ridge_cv_mse": ridge_cv_mse,
        "lasso_cv_mse": lasso_cv_mse,
        "chosen_cv_mse": cv_mse,
    }
    return weights, info


def _build_roster(df: pd.DataFrame, team_code: str, cluster_to_position: Dict[int, str], roster_size: int = 12) -> pd.DataFrame:
    team_df = df[df["Team"] == team_code].copy()
    if team_df.empty:
        return team_df
    starters = []
    used = set()
    for cluster, pos in cluster_to_position.items():
        pool = team_df[team_df["cluster"] == cluster]
        if pool.empty:
            pool = team_df
        pick = pool[~pool["pid"].isin(used)].sort_values("skill_score", ascending=False).head(1)
        if not pick.empty:
            row = pick.iloc[0].to_dict()
            row["Position"] = pos
            starters.append(row)
            used.add(int(row["pid"]))
    starters_df = pd.DataFrame(starters)
    bench = team_df[~team_df["pid"].isin(used)].sort_values("skill_score", ascending=False).head(
        max(0, roster_size - len(starters_df))
    )
    if not bench.empty and "Position" not in bench.columns:
        bench = bench.copy()
        bench["Position"] = bench["cluster"].map(cluster_to_position).fillna("F")
    roster = pd.concat([starters_df, bench], ignore_index=True)
    return roster


def _competitive_Q(lineup: pd.DataFrame) -> np.ndarray:
    z = lineup[FEATURES]
    off = z["WS/40"] + z["TS%"] + 0.5 * z["USG%"]
    ddef = z["DWS_40"] - 0.3 * z["USG%"]
    play = z["AST%"]
    reb = z["TRB%"]
    return np.array([off.mean(), ddef.mean(), play.mean(), reb.mean()], dtype=float)


def _replace_player(roster: pd.DataFrame, cand: pd.Series) -> pd.DataFrame:
    roster = roster.copy()
    pos = cand.get("Position", None)
    if pos in roster["Position"].values:
        idx = roster[roster["Position"] == pos]["skill_score"].idxmin()
    else:
        idx = roster["skill_score"].idxmin()
    roster.loc[idx] = cand
    return roster


def _evaluate_candidate(roster: pd.DataFrame, cand: pd.Series, star_thresh: float, cost_mult: float) -> float:
    Q_old = _competitive_Q(roster)
    roster_new = _replace_player(roster, cand)
    Q_new = _competitive_Q(roster_new)
    delta_q = float(np.mean(Q_new) - np.mean(Q_old))
    star = 1.0 if float(cand["skill_score"]) >= star_thresh else 0.0
    cost = cost_mult * float(cand["skill_score"])
    return 1.2 * delta_q + 0.6 * star - 0.8 * cost


def sensitivity_analysis(weights: np.ndarray, z_df: pd.DataFrame, cluster_to_position: Dict[int, str]) -> Dict[str, float]:
    rng = np.random.default_rng(42)
    df = z_df.copy()
    # baseline skill_score
    df["skill_score"] = np.sum(np.vstack([df[f + "_z"].values for f in FEATURES]).T * weights, axis=1)
    df["pid"] = np.arange(len(df))
    df["Position"] = df["cluster"].map(cluster_to_position).fillna("F")

    base_roster = _build_roster(df, "IND", cluster_to_position)
    star_thresh = float(df["skill_score"].quantile(0.85))
    pools = {
        "Draft": (0.00, 0.45, 0.4),
        "FA": (0.75, 1.00, 1.2),
        "Trade": (0.60, 0.90, 0.9),
    }

    def top5_by_pool(df_local, roster_local) -> Dict[str, List[str]]:
        out = {}
        for name, (q1, q2, cost_mult) in pools.items():
            q_low = df_local["skill_score"].quantile(q1)
            q_high = df_local["skill_score"].quantile(q2)
            pool = df_local[(df_local["skill_score"] >= q_low) & (df_local["skill_score"] <= q_high)]
            pool = pool[pool["Team"] != "IND"]
            scored = []
            for _, cand in pool.head(60).iterrows():
                val = _evaluate_candidate(roster_local, cand, star_thresh, cost_mult)
                scored.append((cand["Player"], val))
            scored.sort(key=lambda x: x[1], reverse=True)
            out[name] = [p for p, _ in scored[:5]]
        return out

    base_top = top5_by_pool(df, base_roster)

    overlaps = {k: [] for k in base_top.keys()}
    spearmans = []
    base_rank = pd.Series(df["skill_score"].rank().values)

    for _ in range(200):
        noise = rng.normal(0, 0.08, size=len(weights))
        w = weights + noise
        if np.sum(np.abs(w)) == 0:
            continue
        w = w / np.sum(np.abs(w))
        df_local = df.copy()
        df_local["skill_score"] = np.sum(
            np.vstack([df_local[f + "_z"].values for f in FEATURES]).T * w, axis=1
        )
        roster_local = _build_roster(df_local, "IND", cluster_to_position)
        top = top5_by_pool(df_local, roster_local)
        for k in overlaps:
            base_set = set(base_top[k])
            top_set = set(top[k])
            overlaps[k].append(len(base_set & top_set) / 5.0)

        rank = pd.Series(df_local["skill_score"].rank().values)
        corr = np.corrcoef(base_rank, rank)[0, 1]
        spearmans.append(float(corr))

    summary = {"spearman_mean": float(np.mean(spearmans)), "spearman_std": float(np.std(spearmans))}
    for k, vals in overlaps.items():
        summary[f"overlap_{k}_mean"] = float(np.mean(vals))
        summary[f"overlap_{k}_std"] = float(np.std(vals))
    return summary


def main():
    players = load_player_data()
    teams, X_raw, z_df = compute_team_features(players)

    adv = pd.read_csv("wnba_advanced_stats.csv")
    elo = pd.read_csv("IND_ELO_O_SOS_season_level.csv")

    elo_proxy = compute_elo_proxy(adv, elo)
    Y, kept, target_season = build_targets(teams, adv, elo_proxy)
    # align X
    X = np.array([X_raw[teams.index(t)] for t in kept], dtype=float)

    weights, info = fit_weights(X, Y)
    weights_dict = {f: float(w) for f, w in zip(FEATURES, weights)}

    # sensitivity analysis
    from project.data.player_kmeans import build_player_model

    model = build_player_model()
    # ensure z columns on model df
    model_df = model.df.copy()
    for f in FEATURES:
        model_df[f + "_z"] = _zscore(model_df[f].astype(float))
    model_df["cluster"] = model.df["cluster"].values
    sens = sensitivity_analysis(weights, model_df, model.cluster_to_position)

    # save json
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "weights": weights_dict,
        "features": FEATURES,
        "method": info["model"],
        "ridge_cv_mse": info["ridge_cv_mse"],
        "lasso_cv_mse": info["lasso_cv_mse"],
        "elo_proxy": elo_proxy,
        "sensitivity": sens,
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2))

    # write report
    with OUTPUT_MD.open("w") as f:
        f.write("# Skill Weight Calibration Report\n\n")
        f.write("## Data sources\n")
        f.write("- allplayers.csv (2024)\n")
        f.write(f"- wnba_advanced_stats.csv (season={target_season})\n")
        f.write("- IND_ELO_O_SOS_season_level.csv (IND historical)\n\n")
        f.write("## Target construction\n")
        f.write("Targets: Win%, NetRtg, ELO_proxy (calibrated from IND historical ELO).\n\n")
        f.write("## Regression\n")
        f.write(f"- Chosen model: {info['model']}\n")
        f.write(f"- Ridge CV MSE: {info['ridge_cv_mse']:.6f}\n")
        f.write(f"- Lasso CV MSE: {info['lasso_cv_mse']:.6f}\n\n")
        f.write("## Weights (L1-normalized)\n")
        for k, v in weights_dict.items():
            f.write(f"- {k}: {v:+.4f}\n")
        f.write("\n## ELO proxy (IND calibrated)\n")
        f.write(
            f"ELO = {elo_proxy['intercept']:.2f} + {elo_proxy['coef_net']:.3f}*NetRtg + {elo_proxy['coef_win']:.3f}*Win%\n"
        )
        f.write("\n## Sensitivity analysis (200 perturbations)\n")
        f.write(f"- Spearman mean: {sens['spearman_mean']:.3f} (std {sens['spearman_std']:.3f})\n")
        for k in ["Draft", "FA", "Trade"]:
            f.write(
                f"- Top5 overlap {k}: mean {sens['overlap_'+k+'_mean']:.3f} (std {sens['overlap_'+k+'_std']:.3f})\n"
            )

    print(f"Saved weights to {OUTPUT_JSON}")
    print(f"Saved report to {OUTPUT_MD}")


if __name__ == "__main__":
    main()
