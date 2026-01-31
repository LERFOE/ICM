from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import os
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, auc, mean_squared_error
from sklearn.linear_model import Ridge, MultiTaskLasso

from project.data.player_kmeans import build_player_model, FEATURES
from project.experiments.utils import build_env
from project.mdp.action import ActionVector
from project.mdp.mask import action_space_per_dim, mutable_mask
from project.solvers.rl_ppo import PPOAgent, PPOConfig


OUTPUT_DIR = Path("newfigures")
POOL_LABELS = {
    "Draft（低成本/长期潜力）": "Draft (Low Cost)",
    "Free Agency（即战力）": "Free Agency",
    "Trade（中高强度补强）": "Trade (Targeted)",
}

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


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save(fig, name: str):
    out = OUTPUT_DIR / name
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def fig_q1_policy_heatmap():
    df = _load_csv("project/experiments/output/q1_leverage_policy_map.csv")
    df["cash_flow"] = df["cash_flow"].astype(float)
    df["leverage"] = df["leverage"].astype(float)

    for macro in sorted(df["macro"].unique()):
        sub = df[df["macro"] == macro].copy()
        pivot = sub.pivot_table(index="leverage", columns="cash_flow", values="a_debt", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(7.5, 5))
        sns.heatmap(
            pivot,
            ax=ax,
            cmap="RdYlGn",
            annot=True,
            fmt=".1f",
            cbar_kws={"label": "Debt Action (avg)"},
        )
        ax.set_title(f"Q1 Policy Map (Macro={int(macro)})")
        ax.set_xlabel("Cash Flow")
        ax.set_ylabel("Leverage")
        _save(fig, f"q1_policy_heatmap_macro{int(macro)}.png")


def fig_q1_violin():
    df = _load_csv("project/experiments/output/q1_leverage_policy_test_detail.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    sns.violinplot(data=df, x="seasons", y="win_pct", hue="policy", ax=axes[0], inner="quart")
    axes[0].set_title("Win% Distribution by Season")
    axes[0].set_xlabel("Seasons")
    axes[0].set_ylabel("Win%")
    sns.violinplot(data=df, x="seasons", y="cash_flow", hue="policy", ax=axes[1], inner="quart")
    axes[1].set_title("Cash Flow Distribution by Season")
    axes[1].set_xlabel("Seasons")
    axes[1].set_ylabel("Cash Flow")
    axes[0].legend_.remove()
    axes[1].legend(loc="upper right")
    _save(fig, "q1_violin_win_cash.png")


def fig_q1_radar():
    df = _load_csv("project/experiments/output/q1_leverage_policy_test_summary.csv")
    sub = df[df["seasons"] == 3].set_index("policy")
    metrics = [
        ("win_pct_mean", 1.0, False, "Win%"),
        ("cash_flow_mean", 1.0, False, "Cash Flow"),
        ("terminal_value_mean", 1.0, False, "Terminal"),
        ("franchise_value_mean", 1.0, False, "Franchise"),
        ("leverage_mean", 1.0, True, "Leverage"),
        ("debt_mean", 1.0, True, "Debt"),
        ("syn_mean", 1.0, True, "Synergy"),
    ]
    labels = [m[3] for m in metrics]
    policies = ["static", "mcts"]
    values = {}
    # Normalize per metric
    for key, _, invert, _ in metrics:
        vals = sub.loc[policies, key].values.astype(float)
        if invert:
            vals = -vals
        vmin, vmax = vals.min(), vals.max()
        if vmax - vmin < 1e-6:
            norm = np.ones_like(vals)
        else:
            norm = (vals - vmin) / (vmax - vmin)
        values[key] = norm

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])

    fig = plt.figure(figsize=(6.5, 6.5))
    ax = fig.add_subplot(111, polar=True)
    for idx, pol in enumerate(policies):
        vals = []
        for key, _, _, _ in metrics:
            vals.append(values[key][idx])
        vals.append(vals[0])
        ax.plot(angles, vals, label=pol.upper())
        ax.fill(angles, vals, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title("Q1 Policy Radar (Season 3)")
    ax.set_yticklabels([])
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    _save(fig, "q1_radar_season3.png")


def fig_q1_roc():
    df = _load_csv("IND_ELO_O_SOS_game_level.csv")
    ind_games = df[(df["home_abbr"] == "IND") | (df["away_abbr"] == "IND")].copy()
    if ind_games.empty:
        return
    # Actual IND win
    ind_games["ind_win"] = np.where(
        ind_games["home_abbr"] == "IND", ind_games["home_win"], 1 - ind_games["home_win"]
    )
    # Predict using ELO diff
    ind_games["elo_diff"] = np.where(
        ind_games["home_abbr"] == "IND",
        ind_games["ELO_t"] - ind_games["opp_pre_elo_t"],
        ind_games["ELO_t"] - ind_games["opp_pre_elo_t"],
    )
    x = ind_games["elo_diff"].fillna(0).values / 400.0
    prob = 1 / (1 + np.exp(-x))
    y = ind_games["ind_win"].fillna(0).values

    fpr, tpr, _ = roc_curve(y, prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2c7fb8", lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_title("IND Win Prediction ROC (ELO-based)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    _save(fig, "q1_roc_win_prediction.png")


def fig_q2_sankey():
    df = _load_csv("project/experiments/output/q2_recruitment_strategy.csv")
    df["pool_en"] = df["pool"].map(POOL_LABELS).fillna(df["pool"])
    # Bin owner_value into low/med/high by global quantiles
    quant = df["owner_value"].quantile([0.3, 0.7]).values
    def bucket(v):
        if v <= quant[0]:
            return "Low"
        if v <= quant[1]:
            return "Mid"
        return "High"
    df["bucket"] = df["owner_value"].apply(bucket)

    pools = df["pool_en"].unique().tolist()
    fig, axes = plt.subplots(1, len(pools), figsize=(12, 4), sharey=True)
    if len(pools) == 1:
        axes = [axes]
    for ax, pool in zip(axes, pools):
        sub = df[df["pool_en"] == pool]
        counts = sub["bucket"].value_counts().reindex(["High", "Mid", "Low"]).fillna(0)
        flows = [counts.sum(), -counts["High"], -counts["Mid"], -counts["Low"]]
        labels = [pool, "High", "Mid", "Low"]
        from matplotlib.sankey import Sankey
        sankey = Sankey(ax=ax, unit=None, format=".0f", scale=0.01)
        sankey.add(flows=flows, labels=labels, orientations=[0, 1, 0, -1])
        sankey.finish()
        ax.set_title(pool)
    fig.suptitle("Q2 Recruitment Flow: Pool → Value Bucket")
    _save(fig, "q2_sankey_recruitment.png")


def fig_q2_bubble():
    df = _load_csv("project/experiments/output/q2_recruitment_strategy.csv")
    df["pool_en"] = df["pool"].map(POOL_LABELS).fillna(df["pool"])
    top = df.sort_values("owner_value", ascending=False).head(30)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sizes = (top["cost"].abs() + 0.1) * 300
    sns.scatterplot(
        data=top,
        x="delta_q",
        y="owner_value",
        hue="pool_en",
        size=sizes,
        sizes=(50, 800),
        alpha=0.8,
        ax=ax,
    )
    for _, r in top.head(6).iterrows():
        ax.text(r["delta_q"], r["owner_value"], r["Player"].split()[0], fontsize=8)
    ax.set_title("Q2 Candidate Bubble Chart (Top 30 by Owner Value)")
    ax.set_xlabel("ΔQ (Skill Gain)")
    ax.set_ylabel("Owner Value Score")
    ax.legend(loc="lower right")
    _save(fig, "q2_bubble_candidates.png")


def fig_player_corr_heatmap():
    df = _load_csv("allplayers.csv")
    cols = ["WS/40", "TS%", "USG%", "AST%", "TRB%", "DWS", "PER", "ORtg", "DRtg", "WS"]
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr()
    fig = plt.figure(figsize=(7.5, 6))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, square=True)
    plt.title("Player Metric Correlation Heatmap")
    _save(fig, "player_corr_heatmap.png")


def fig_parallel_coordinates_clusters():
    model = build_player_model()
    profile = model.cluster_profiles.copy()
    z = (profile - profile.mean()) / profile.std(ddof=0)
    z = z.reset_index().rename(columns={"cluster": "Cluster"})
    fig = plt.figure(figsize=(8, 4.5))
    pd.plotting.parallel_coordinates(z, "Cluster", colormap=plt.get_cmap("tab10"))
    plt.title("Cluster Skill Profiles (Parallel Coordinates, Z-score)")
    plt.xlabel("Metric")
    plt.ylabel("Z-score")
    _save(fig, "player_cluster_parallel_coordinates.png")


def fig_ridgeline_elo():
    df = _load_csv("IND_ELO_O_SOS_game_level.csv")
    df = df[df["Season"] >= df["Season"].max() - 8]
    seasons = sorted(df["Season"].unique())
    fig, ax = plt.subplots(figsize=(7.5, 6))
    y_offsets = np.linspace(0, 1.6, len(seasons))
    for offset, season in zip(y_offsets, seasons):
        s = df[df["Season"] == season]["ELO_t"].dropna()
        if s.empty:
            continue
        sns.kdeplot(s, ax=ax, bw_adjust=1.0, fill=True, alpha=0.6)
        for line in ax.lines[-1:]:
            line.set_ydata(line.get_ydata() + offset)
        for poly in ax.collections[-1:]:
            verts = poly.get_paths()[0].vertices
            verts[:, 1] += offset
    ax.set_title("Ridgeline: IND ELO Distribution by Season")
    ax.set_xlabel("ELO")
    ax.set_yticks(y_offsets)
    ax.set_yticklabels(seasons)
    _save(fig, "q1_ridgeline_elo.png")


def fig_attendance_streamgraph():
    df = _load_csv("wnba_attendance.csv")
    latest_years = sorted(df["Season"].unique())[-8:]
    sub = df[df["Season"].isin(latest_years)].copy()
    # pick top 6 teams by avg attendance across these years
    team_order = (
        sub.groupby("Team")["Avg_Attendance"].mean().sort_values(ascending=False).head(6).index.tolist()
    )
    sub = sub[sub["Team"].isin(team_order)]
    pivot = sub.pivot_table(index="Season", columns="Team", values="Avg_Attendance", aggfunc="mean").fillna(0)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    pivot.plot.area(ax=ax, cmap="tab20", alpha=0.85)
    ax.set_title("WNBA Attendance Streamgraph (Top Teams)")
    ax.set_ylabel("Avg Attendance")
    ax.set_xlabel("Season")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    _save(fig, "attendance_streamgraph.png")


def fig_q3_expansion_bar():
    df = _load_csv("project/experiments/output/q3_expansion_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(df))
    ax.bar(x - 0.15, df["delta_terminal"], width=0.3, label="ΔTerminal")
    ax.bar(x + 0.15, df["delta_cf"], width=0.3, label="ΔCF")
    ax.set_xticks(x)
    ax.set_xticklabels(df["site"])
    ax.set_title("Q3 Expansion Impact (Indiana Owner)")
    ax.axhline(0, color="gray", lw=1)
    ax.legend()
    _save(fig, "q3_expansion_bar.png")


def fig_q3_chord_like():
    df = _load_csv("project/experiments/output/q3_expansion_sensitivity.csv")
    sites = df["site"].tolist()
    teams = sorted(set(",".join(df["winners"].tolist() + df["losers"].tolist()).split(",")))

    # Circular layout
    nodes = sites + teams
    angles = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
    pos = {node: (math.cos(a), math.sin(a)) for node, a in zip(nodes, angles)}

    fig, ax = plt.subplots(figsize=(7, 7))
    for node in nodes:
        x, y = pos[node]
        ax.scatter([x], [y], s=80 if node in sites else 40, color="#4c72b0" if node in sites else "#999999")
        ax.text(x * 1.08, y * 1.08, node, ha="center", va="center", fontsize=8)

    # Draw edges
    for _, row in df.iterrows():
        site = row["site"]
        winners = row["winners"].split(",")
        losers = row["losers"].split(",")
        for team in winners:
            x1, y1 = pos[site]
            x2, y2 = pos[team]
            ax.plot([x1, x2], [y1, y2], color="#2ca25f", alpha=0.6, lw=1.8)
        for team in losers:
            x1, y1 = pos[site]
            x2, y2 = pos[team]
            ax.plot([x1, x2], [y1, y2], color="#de2d26", alpha=0.6, lw=1.2)

    ax.set_title("Q3 Expansion Winners/Losers (Chord-style Network)")
    ax.axis("off")
    _save(fig, "q3_chord_expansion.png")


def fig_q3_policy_comparison_deltas():
    """Q3: quantify how policies differ under expansion scenarios (season 3 end)."""
    path = Path("project/experiments/output/q3_policy_comparison_summary.csv")
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[df["seasons"] == 3].copy()
    if df.empty:
        return

    # Compute deltas relative to baseline per site to make the chart readable.
    base = df[df["policy"] == "baseline"].set_index("site")
    policies = ["ppo", "defensive", "aggressive"]
    rows = []
    for _, r in df.iterrows():
        if r["policy"] == "baseline":
            continue
        site = r["site"]
        if site not in base.index:
            continue
        rows.append(
            {
                "site": site,
                "policy": r["policy"],
                "delta_terminal": float(r["terminal_value_mean"]) - float(base.loc[site]["terminal_value_mean"]),
                "delta_cf_cum": float(r["cash_flow_cum_mean"]) - float(base.loc[site]["cash_flow_cum_mean"]),
                "delta_win": float(r["win_pct_mean"]) - float(base.loc[site]["win_pct_mean"]),
            }
        )
    plot = pd.DataFrame(rows)
    if plot.empty:
        return
    plot["policy"] = pd.Categorical(plot["policy"], categories=policies, ordered=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)
    sns.barplot(data=plot, x="site", y="delta_terminal", hue="policy", ax=axes[0], palette="Set2")
    axes[0].axhline(0, color="gray", lw=1)
    axes[0].set_title("ΔTerminal vs Baseline (Season 3)")
    axes[0].set_xlabel("Expansion Site Scenario")
    axes[0].set_ylabel("Δ Terminal Value")

    sns.barplot(data=plot, x="site", y="delta_cf_cum", hue="policy", ax=axes[1], palette="Set2")
    axes[1].axhline(0, color="gray", lw=1)
    axes[1].set_title("Δ Cumulative CF vs Baseline (Season 3)")
    axes[1].set_xlabel("Expansion Site Scenario")
    axes[1].set_ylabel("Δ Cumulative CF")
    axes[0].legend_.remove()
    axes[1].legend(loc="upper right")
    _save(fig, "q3_policy_comparison_deltas.png")


def fig_q3_expansion_offseason_action_heatmaps():
    """Q3: show how the expansion-year offseason actions change (evidence for narrative)."""
    path = Path("project/experiments/output/q3_policy_comparison_detail.csv")
    if not path.exists():
        return
    df = pd.read_csv(path)
    sub = df[(df["marker"] == "offseason_decision") & (df["year"] == 2026)].copy()
    if sub.empty:
        return
    # Use stable policy ordering for paper plots.
    pol_order = ["baseline", "ppo", "defensive", "aggressive"]
    sub["policy"] = pd.Categorical(sub["policy"], categories=pol_order, ordered=True)

    for col, title, fname in [
        ("a_debt", "Expansion-Year Offseason: Debt Action (mean)", "q3_offseason_action_debt.png"),
        ("a_salary", "Expansion-Year Offseason: Salary Action (mean)", "q3_offseason_action_salary.png"),
    ]:
        pivot = sub.pivot_table(index="policy", columns="site", values=col, aggfunc="mean").reindex(pol_order)
        fig, ax = plt.subplots(figsize=(7.5, 3.2))
        sns.heatmap(pivot, ax=ax, cmap="YlGnBu", annot=True, fmt=".2f", cbar_kws={"label": col})
        ax.set_title(title)
        ax.set_xlabel("Site Scenario")
        ax.set_ylabel("Policy")
        _save(fig, fname)


def _nearest_valid(env, state) -> ActionVector:
    target = ActionVector.from_list(list(state.K))
    if target in env.valid_actions(state):
        return target
    valid = env.valid_actions(state)
    target_list = target.to_list()

    def dist(a: ActionVector) -> int:
        return sum(abs(x - y) for x, y in zip(a.to_list(), target_list))

    return min(valid, key=dist)


def _mode_allowed_factory(env, mode: str):
    target_idx = 2 if mode == "ticket" else 5

    def allowed(state):
        base_action = _nearest_valid(env, state)
        allowed = action_space_per_dim(state.Theta, state.K)
        for i in range(len(allowed)):
            if i != target_idx:
                allowed[i] = [base_action.to_list()[i]]
        if target_idx == 5:
            mask = mutable_mask(state.Theta)
            if mask[5] == 1 and env.config.max_equity_action is not None:
                allowed[5] = [a for a in allowed[5] if a <= env.config.max_equity_action]
        return allowed

    return allowed


def _collect_action_freq(env, policy, episodes=6, max_steps=20):
    counter = {}
    for _ in range(episodes):
        state = env.reset()
        for _ in range(max_steps):
            action = policy(state)
            counter[action.to_tuple()] = counter.get(action.to_tuple(), 0) + 1
            state, _, done, _ = env.step(state, action)
            if done:
                break
    return counter


def fig_q4_policy_hist():
    env = build_env(use_data=True, seed=42)
    ppo_cfg = PPOConfig(steps_per_update=128, epochs=2)

    for mode, fname in [("ticket", "q4_ticket_policy_hist.png"), ("equity", "q4_equity_policy_hist.png")]:
        allowed_fn = _mode_allowed_factory(env, mode)
        agent = PPOAgent(env, cfg=ppo_cfg, allowed_fn=allowed_fn)
        agent.train(episodes=8)
        freq = _collect_action_freq(env, agent.act, episodes=6)
        # summarize target dimension frequencies
        target_idx = 2 if mode == "ticket" else 5
        counts = {}
        for action, c in freq.items():
            counts[action[target_idx]] = counts.get(action[target_idx], 0) + c
        xs = sorted(counts.keys())
        ys = [counts[x] for x in xs]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(xs, ys, color="#4c72b0")
        ax.set_title(f"Q4 {mode.title()} Action Frequency")
        ax.set_xlabel(f"{mode.title()} action index")
        ax.set_ylabel("Count")
        _save(fig, fname)


def fig_market_bubble():
    att = _load_csv("wnba_attendance.csv")
    val = _load_csv("wnba_valuations.csv")
    att_2024 = att[att["Season"] == 2024]
    val_2024 = val[val["Year"] == 2024]
    merged = att_2024.merge(val_2024, left_on="Team", right_on="Team", how="inner")
    if merged.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    sizes = merged["Valuation_M"].fillna(50) * 3
    scatter = ax.scatter(
        merged["Avg_Attendance"],
        merged["Revenue_M"].fillna(0),
        s=sizes,
        c=merged["Valuation_M"],
        cmap="viridis",
        alpha=0.8,
    )
    for _, r in merged.iterrows():
        ax.text(r["Avg_Attendance"], r["Revenue_M"], r["Team"].split()[0], fontsize=8)
    ax.set_xlabel("Avg Attendance (2024)")
    ax.set_ylabel("Revenue (M)")
    ax.set_title("Market Bubble: Attendance vs Revenue (Size=Valuation)")
    plt.colorbar(scatter, ax=ax, label="Valuation (M)")
    _save(fig, "market_bubble_attendance_revenue.png")


def _zscore(series: pd.Series) -> pd.Series:
    mu = series.mean()
    std = series.std(ddof=0)
    if std == 0:
        std = 1.0
    return (series - mu) / std


def _compute_team_features(df: pd.DataFrame) -> Tuple[List[str], np.ndarray, pd.DataFrame]:
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


def _compute_elo_proxy(adv: pd.DataFrame, elo: pd.DataFrame) -> Tuple[float, float, float]:
    ind_adv = adv[adv["Team"] == "Indiana Fever"].copy()
    merged = ind_adv.merge(elo, left_on="Season", right_on="Season", how="inner")
    merged = merged.dropna(subset=["NetRtg", "Win%", "ELO_last_pregame"])
    if merged.empty:
        return 1500.0, 20.0, 50.0
    X = merged[["NetRtg", "Win%"]].values.astype(float)
    y = merged["ELO_last_pregame"].values.astype(float)
    ridge = Ridge(alpha=1.0).fit(X, y)
    return float(ridge.intercept_), float(ridge.coef_[0]), float(ridge.coef_[1])


def _build_targets(teams: List[str], adv: pd.DataFrame, elo_params: Tuple[float, float, float]) -> Tuple[np.ndarray, List[str], int]:
    target_season = int(adv["Season"].max())
    adv_season = adv[adv["Season"] == target_season].copy()
    adv_season["TeamCode"] = adv_season["Team"].map(TEAM_NAME_TO_CODE)
    adv_season = adv_season.dropna(subset=["TeamCode"]).set_index("TeamCode")
    y_rows = []
    kept = []
    intercept, coef_net, coef_win = elo_params
    for team in teams:
        if team not in adv_season.index:
            continue
        row = adv_season.loc[team]
        win = float(row["Win%"])
        net = float(row["NetRtg"])
        elo_proxy = intercept + coef_net * net + coef_win * win
        y_rows.append([win, net, elo_proxy])
        kept.append(team)
    return np.array(y_rows, dtype=float), kept, target_season


def fig_skill_weights_bar():
    weights_path = Path("project/data/skill_weights.json")
    if not weights_path.exists():
        return
    data = json.loads(weights_path.read_text())
    weights = data["weights"]
    labels = list(weights.keys())
    vals = [weights[k] for k in labels]
    colors = ["#2ca25f" if v >= 0 else "#de2d26" for v in vals]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, vals, color=colors)
    ax.axhline(0, color="gray", lw=1)
    ax.set_title("Skill Weight Coefficients (Ridge/Lasso fitted)")
    ax.set_ylabel("Weight (L1-normalized)")
    ax.set_xticklabels(labels, rotation=25, ha="right")
    _save(fig, "skill_weights_bar.png")


def fig_skill_model_compare():
    weights_path = Path("project/data/skill_weights.json")
    if not weights_path.exists():
        return
    data = json.loads(weights_path.read_text())
    ridge_mse = data.get("ridge_cv_mse", np.nan)
    lasso_mse = data.get("lasso_cv_mse", np.nan)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(["Ridge", "Lasso"], [ridge_mse, lasso_mse], color=["#4c72b0", "#dd8452"])
    ax.set_title("Model Fit Comparison (MSE)")
    ax.set_ylabel("MSE (lower is better)")
    _save(fig, "skill_model_mse.png")


def fig_skill_prediction_scatter():
    # Refit models quickly to show predicted vs actual on Win%/NetRtg/ELO_proxy
    players = pd.read_csv("allplayers.csv")
    players["DWS_40"] = players["DWS"].astype(float) / players["MP"].replace(0, np.nan).astype(float) * 40.0
    players = players.dropna(subset=FEATURES + ["MP", "Team"]).copy()
    teams, X, _ = _compute_team_features(players)
    adv = pd.read_csv("wnba_advanced_stats.csv")
    elo = pd.read_csv("IND_ELO_O_SOS_season_level.csv")
    elo_params = _compute_elo_proxy(adv, elo)
    Y, kept, _ = _build_targets(teams, adv, elo_params)
    if len(kept) < 4:
        return
    X = np.array([X[teams.index(t)] for t in kept], dtype=float)
    Y_mean = Y.mean(axis=0)
    Y_std = np.where(Y.std(axis=0, ddof=0) == 0, 1.0, Y.std(axis=0, ddof=0))
    Yz = (Y - Y_mean) / Y_std

    ridge = Ridge(alpha=1.0).fit(X, Yz)
    pred = ridge.predict(X)
    pred = pred * Y_std + Y_mean

    labels = ["Win%", "NetRtg", "ELO_proxy"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for i, ax in enumerate(axes):
        ax.scatter(Y[:, i], pred[:, i], color="#4c72b0", alpha=0.8)
        minv = min(Y[:, i].min(), pred[:, i].min())
        maxv = max(Y[:, i].max(), pred[:, i].max())
        ax.plot([minv, maxv], [minv, maxv], color="gray", linestyle="--")
        ax.set_title(labels[i])
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
    fig.suptitle("Ridge Fit: Actual vs Predicted Targets")
    _save(fig, "skill_fit_scatter.png")


def fig_skill_sensitivity():
    weights_path = Path("project/data/skill_weights.json")
    if not weights_path.exists():
        return
    data = json.loads(weights_path.read_text())
    weights = np.array([data["weights"][f] for f in FEATURES], dtype=float)

    model = build_player_model()
    df = model.df.copy()
    for f in FEATURES:
        df[f + "_z"] = _zscore(df[f].astype(float))
    df["pid"] = np.arange(len(df))
    df["Position"] = df["cluster"].map(model.cluster_to_position).fillna("F")

    def _build_roster(df_local, team_code="IND", roster_size=12):
        team_df = df_local[df_local["Team"] == team_code].copy()
        if team_df.empty:
            return team_df
        starters = []
        used = set()
        for cluster, pos in model.cluster_to_position.items():
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
            bench["Position"] = bench["cluster"].map(model.cluster_to_position).fillna("F")
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

    pools = {
        "Draft": (0.00, 0.45, 0.4),
        "FA": (0.75, 1.00, 1.2),
        "Trade": (0.60, 0.90, 0.9),
    }

    df["skill_score"] = np.sum(np.vstack([df[f + "_z"].values for f in FEATURES]).T * weights, axis=1)
    base_roster = _build_roster(df, "IND")
    star_thresh = float(df["skill_score"].quantile(0.85))
    base_rank = pd.Series(df["skill_score"].rank().values)

    def top5(df_local, roster_local):
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

    base_top = top5(df, base_roster)
    rng = np.random.default_rng(42)
    spearmans = []
    overlap = {k: [] for k in pools.keys()}

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
        roster_local = _build_roster(df_local, "IND")
        top = top5(df_local, roster_local)
        for k in overlap:
            overlap[k].append(len(set(base_top[k]) & set(top[k])) / 5.0)
        rank = pd.Series(df_local["skill_score"].rank().values)
        spearmans.append(float(np.corrcoef(base_rank, rank)[0, 1]))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(spearmans, bins=20, kde=True, ax=axes[0], color="#4c72b0")
    axes[0].set_title("Spearman Rank Stability (200 perturbations)")
    axes[0].set_xlabel("Spearman correlation")
    axes[0].set_ylabel("Count")

    data = [overlap["Draft"], overlap["FA"], overlap["Trade"]]
    sns.violinplot(data=data, ax=axes[1], inner="quart", palette="Set2")
    axes[1].set_xticklabels(["Draft", "FA", "Trade"])
    axes[1].set_title("Top5 Overlap Distribution")
    axes[1].set_ylabel("Overlap ratio")
    _save(fig, "skill_sensitivity.png")


def main():
    _ensure_dir(OUTPUT_DIR)
    sns.set_theme(style="whitegrid")

    fig_q1_policy_heatmap()
    fig_q1_violin()
    fig_q1_radar()
    fig_q1_roc()
    fig_q2_sankey()
    fig_q2_bubble()
    fig_player_corr_heatmap()
    fig_parallel_coordinates_clusters()
    fig_ridgeline_elo()
    fig_attendance_streamgraph()
    fig_q3_expansion_bar()
    fig_q3_chord_like()
    fig_q3_policy_comparison_deltas()
    fig_q3_expansion_offseason_action_heatmaps()
    fig_q4_policy_hist()
    fig_market_bubble()
    fig_skill_weights_bar()
    fig_skill_model_compare()
    fig_skill_prediction_scatter()
    fig_skill_sensitivity()

    print(f"Saved figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
