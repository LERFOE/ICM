import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from project.data.player_kmeans import FEATURES, load_player_data, load_skill_weights
from project.experiments.utils import build_env


@dataclass
class EquityConfig:
    team_code: str = "IND"
    roster_size: int = 12
    equity_cap: float = 0.03
    equity_step: float = 0.005
    equity_levels: Tuple[float, ...] = (0.0, 0.005, 0.01, 0.015, 0.02)
    star_quantile: float = 0.85
    omega_scale: float = 0.01
    value_scale: float = 0.04
    omega_peak: float = 0.01
    omega_sigma: float = 0.0075
    w_se: float = 1.0
    w_ca: float = 1.0
    w_fd: float = 1.0
    ca_alpha: float = 0.35
    se_alpha: float = 0.30
    se_gate_scale: float = 0.6
    fd_alpha: float = 1.0
    distress_lambda0: float = 0.30
    distress_cf_scale: float = 1.6
    distress_lambda_scale: float = 4.0
    relief_factor: float = 0.6
    relief_horizon: float = 6.0
    win_scale: float = 0.04


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std, mean, std


def compute_skill_tables(team_code: str, roster_size: int) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    df = load_player_data(Path("allplayers.csv"))
    weights = load_skill_weights()

    X = df[FEATURES].values
    Xs, mean, std = _standardize(X)
    z = pd.DataFrame(Xs, columns=FEATURES)
    df = df.copy()
    df[FEATURES] = z
    df["skill_score"] = sum(weights[f] * df[f] for f in FEATURES)
    df["skill_z"] = (df["skill_score"] - df["skill_score"].mean()) / (df["skill_score"].std() + 1e-9)

    team_df = df[df["Team"].astype(str).str.contains(team_code, na=False)].copy()
    if team_df.empty:
        raise ValueError(f"No players found for team {team_code} in allplayers.csv")
    team_df = team_df.sort_values("MP", ascending=False).head(roster_size).copy()
    return df, team_df, weights


def build_tri_factor_options(
    league_df: pd.DataFrame,
    team_df: pd.DataFrame,
    cfg: EquityConfig,
    env_state,
) -> Tuple[pd.DataFrame, Dict[str, List[Dict[str, float]]]]:
    base_fv = env_state.F.FV
    base_debt = env_state.F.D
    owner_share = float(getattr(env_state.F, "owner_share", 1.0))
    owner_value = max(base_fv - base_debt, 1e-6)
    base_rev = max(env_state.F.CF, 0.0)
    if base_rev <= 0:
        base_rev = 1.0

    lambda_t = env_state.F.leverage
    cf_norm = env_state.F.CF / max(base_rev, 1e-6)
    distress = sigmoid(
        cfg.distress_lambda_scale * (lambda_t - cfg.distress_lambda0)
        - cfg.distress_cf_scale * cf_norm
    )

    star_threshold = float(league_df["skill_score"].quantile(cfg.star_quantile))
    skill_scale = float(league_df["skill_score"].std() + 1e-9)

    team_df = team_df.copy()
    # Soft gate around q0.85 to avoid degenerate all-zero SE when no one crosses the threshold
    team_df["selection_eff"] = sigmoid((team_df["skill_score"] - star_threshold) / (cfg.se_gate_scale * skill_scale))
    skill_min = float(league_df["skill_score"].min())
    skill_max = float(league_df["skill_score"].max())
    skill_range = max(skill_max - skill_min, 1e-6)
    team_df["skill_scaled"] = (team_df["skill_score"] - skill_min) / skill_range
    base_salary = env_state.F.psi_mean_salary

    options_rows = []
    options_by_player: Dict[str, List[Dict[str, float]]] = {}
    for _, row in team_df.iterrows():
        player = str(row["Player"])
        skill_z = float(row["skill_z"])
        skill_scaled = float(row["skill_scaled"])
        se = float(row["selection_eff"])
        market_value = base_salary * (0.7 + 1.1 * skill_scaled)
        options = []
        for omega in cfg.equity_levels:
            omega_max = max(cfg.equity_levels)
            omega_ratio = np.sqrt(max(omega / omega_max, 0.0))
            ca_raw = np.clip(skill_z, -2.5, 2.5) * np.log1p(omega / cfg.omega_scale)
            sweet = np.exp(-((omega - cfg.omega_peak) / max(cfg.omega_sigma, 1e-6)) ** 2)
            ca = cfg.ca_alpha * ca_raw * sweet * se
            retention = cfg.se_alpha * se * omega_ratio
            cash_relief = market_value * (omega / omega_max) * cfg.relief_factor
            fd = cfg.fd_alpha * distress * cash_relief * cfg.relief_horizon
            se_term = cfg.w_se * se * omega_ratio
            ca_term = cfg.w_ca * ca
            delta_fv = base_fv * cfg.value_scale * (se_term + ca_term + retention)
            owner_share_after = max(owner_share - omega, 0.0)
            owner_terminal_delta = owner_share_after * (base_fv + delta_fv - base_debt) - owner_share * (
                base_fv - base_debt
            )
            owner_cash_delta = cfg.w_fd * fd
            net = owner_terminal_delta + owner_cash_delta
            win_delta = cfg.win_scale * ca_raw * sweet
            options.append(
                {
                    "Player": player,
                    "omega": omega,
                    "se": se,
                    "ca": ca,
                    "retention": retention,
                    "fd": fd,
                    "delta_fv": delta_fv,
                    "owner_share_after": owner_share_after,
                    "owner_terminal_delta": owner_terminal_delta,
                    "owner_cash_delta": owner_cash_delta,
                    "net": net,
                    "win_delta": win_delta,
                    "cf_delta": fd,
                    "market_value": market_value,
                    "skill_z": skill_z,
                }
            )
        options_by_player[player] = options
        options_rows.extend(options)

    options_df = pd.DataFrame(options_rows)
    meta = {
        "owner_value": owner_value,
        "owner_share": owner_share,
        "base_fv": base_fv,
        "base_debt": base_debt,
        "distress": float(distress),
        "star_threshold": star_threshold,
        "skill_scale": skill_scale,
    }
    return options_df, options_by_player, meta


def solve_mckp(options_by_player: Dict[str, List[Dict[str, float]]], cap: float, step: float):
    players = list(options_by_player.keys())
    cap_units = int(round(cap / step))
    dp = np.full((len(players) + 1, cap_units + 1), -1e18)
    choice = np.full((len(players) + 1, cap_units + 1), -1, dtype=int)
    dp[0, 0] = 0.0

    for i, player in enumerate(players, start=1):
        opts = options_by_player[player]
        for b in range(cap_units + 1):
            best_val = -1e18
            best_idx = -1
            for j, opt in enumerate(opts):
                cost_units = int(round(opt["omega"] / step))
                if cost_units <= b and dp[i - 1, b - cost_units] > -1e17:
                    val = dp[i - 1, b - cost_units] + opt["net"]
                    if val > best_val:
                        best_val = val
                        best_idx = j
            dp[i, b] = best_val
            choice[i, b] = best_idx

    best_b = int(np.nanargmax(dp[len(players)]))
    best_val = dp[len(players), best_b]

    selected = []
    b = best_b
    for i in range(len(players), 0, -1):
        j = choice[i, b]
        if j < 0:
            j = 0
        opt = options_by_player[players[i - 1]][j]
        selected.append(opt)
        b -= int(round(opt["omega"] / step))
    selected = list(reversed(selected))
    return best_val, best_b, pd.DataFrame(selected), dp


def summarize_solution(solution_df: pd.DataFrame, meta: Dict[str, float]):
    total_equity = solution_df["omega"].sum()
    total_net = solution_df["net"].sum()
    total_win = solution_df["win_delta"].sum()
    total_cf = solution_df["cf_delta"].sum()
    total_owner_terminal = solution_df["owner_terminal_delta"].sum()
    total_owner_cash = solution_df["owner_cash_delta"].sum()
    return {
        "total_equity": float(total_equity),
        "total_net_value": float(total_net),
        "total_owner_terminal_delta": float(total_owner_terminal),
        "total_owner_cash_delta": float(total_owner_cash),
        "total_win_delta": float(total_win),
        "total_cf_delta": float(total_cf),
        "owner_value": float(meta["owner_value"]),
        "owner_share": float(meta["owner_share"]),
    }


def plot_figures(
    league_df: pd.DataFrame,
    team_df: pd.DataFrame,
    options_df: pd.DataFrame,
    solution_df: pd.DataFrame,
    frontier_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    output_dir: Path,
    star_threshold: float,
    chosen_cap: float,
):
    sns.set_theme(style="whitegrid")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Skill violin: league vs IND
    fig, ax = plt.subplots(figsize=(8, 4.8))
    plot_df = league_df.copy()
    plot_df["Group"] = "League"
    ind_df = team_df.copy()
    ind_df["Group"] = "IND"
    plot_df = pd.concat([plot_df, ind_df], ignore_index=True)
    sns.violinplot(
        data=plot_df,
        x="Group",
        y="skill_score",
        hue="Group",
        palette=["#6baed6", "#fd8d3c"],
        ax=ax,
        legend=False,
        inner=None,
        cut=0,
        linewidth=1.2,
        bw_adjust=0.5,
    )
    sns.boxplot(
        data=plot_df,
        x="Group",
        y="skill_score",
        width=0.18,
        showfliers=False,
        boxprops=dict(facecolor="white", alpha=0.7, linewidth=1.4),
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        ax=ax,
    )
    sns.stripplot(
        data=ind_df,
        x="Group",
        y="skill_score",
        color="#444444",
        size=3.5,
        alpha=0.75,
        jitter=0.12,
        ax=ax,
    )
    ymin = float(plot_df["skill_score"].min()) - 0.1
    ymax = float(plot_df["skill_score"].max()) + 0.1
    ax.set_ylim(ymin, ymax)
    ax.axhline(star_threshold, color="#d62728", linestyle="--", linewidth=2, label="Star Threshold (q0.85)")
    ax.text(
        -0.25,
        star_threshold + 0.03,
        f"q0.85 = {star_threshold:.2f}",
        color="#d62728",
        fontsize=10,
        ha="left",
        va="bottom",
    )
    ind_stars = team_df[team_df["skill_score"] >= star_threshold].sort_values("skill_score", ascending=False)
    label_suffix = ""
    if ind_stars.empty:
        # If no IND player reaches league q0.85, label the two closest to the threshold.
        ind_stars = team_df.sort_values("skill_score", ascending=False).head(2)
        label_suffix = " (closest)"
    for idx, (_, row) in enumerate(ind_stars.iterrows()):
        y = float(row["skill_score"])
        x = 1.0
        ax.scatter([x], [y], color="#d62728", s=28, zorder=5)
        offset = 0.14 if idx % 2 == 0 else -0.14
        ax.annotate(
            f"{row['Player']}{label_suffix}",
            xy=(x, y),
            xytext=(x + 0.22, y + offset),
            textcoords="data",
            ha="left",
            va="center",
            color="#d62728",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
            arrowprops=dict(arrowstyle="-", color="#d62728", lw=1.2),
        )
    ax.set_title("Skill Score Distribution")
    ax.set_xlabel("")
    ax.set_ylabel("skill_score")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "q4_skill_violin.png", dpi=300)
    plt.close(fig)

    # 2) Tri-factor bars for 1% equity option
    ref = options_df[options_df["omega"] == 0.01].copy()
    ref = ref.sort_values("net", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(ref["Player"], ref["se"], label="Selection", color="#6baed6")
    ax.bar(ref["Player"], ref["ca"], bottom=ref["se"], label="Competitive", color="#9ecae1")
    ax.bar(ref["Player"], ref["fd"], bottom=ref["se"] + ref["ca"], label="Financial", color="#fd8d3c")
    ax.set_title("Tri-Factor Components (1% equity)")
    ax.set_ylabel("Contribution (scaled)")
    ax.tick_params(axis="x", labelrotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "q4_trifactor_bars.png", dpi=300)
    plt.close(fig)

    # 3) Owner terminal value heatmap (players x equity levels)
    heat = options_df.pivot(index="Player", columns="omega", values="owner_terminal_delta")
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    sns.heatmap(heat, cmap="RdYlGn", center=0, annot=False, ax=ax)
    ax.set_title("Equity Option Owner Terminal Δ Heatmap")
    ax.set_xlabel("Equity Share")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_dir / "q4_equity_option_heatmap.png", dpi=300)
    plt.close(fig)

    # 3b) Win delta heatmap (sweet-spot effect)
    heat_win = options_df.pivot(index="Player", columns="omega", values="win_delta")
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    sns.heatmap(heat_win, cmap="YlGnBu", center=0, annot=False, ax=ax)
    ax.set_title("Equity Option Win Δ Heatmap (Sweet Spot)")
    ax.set_xlabel("Equity Share")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_dir / "q4_equity_option_win_heatmap.png", dpi=300)
    plt.close(fig)

    # 4) Equity allocation lollipop
    sol = solution_df.sort_values("omega")
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.hlines(sol["Player"], 0, sol["omega"], color="#3182bd")
    ax.plot(sol["omega"], sol["Player"], "o", color="#08519c")
    ax.set_title("Optimal Equity Allocation (MCKP)")
    ax.set_xlabel("Equity Share")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_dir / "q4_equity_allocation_lollipop.png", dpi=300)
    plt.close(fig)

    # 5) Frontier: equity cap vs metrics
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    sns.lineplot(data=frontier_df, x="equity_cap", y="total_net_value", marker="o", ax=axes[0])
    sns.lineplot(data=frontier_df, x="equity_cap", y="total_win_delta", marker="o", ax=axes[1])
    sns.lineplot(data=frontier_df, x="equity_cap", y="total_cf_delta", marker="o", ax=axes[2])
    axes[0].set_title("Owner Value Gain")
    axes[1].set_title("Win% Delta")
    axes[2].set_title("Cashflow Delta")
    for ax in axes:
        ax.set_xlabel("Equity Cap")
    fig.tight_layout()
    fig.savefig(output_dir / "q4_equity_frontier.png", dpi=300)
    plt.close(fig)

    # 6) Sensitivity heatmap (w_se vs w_fd) for w_ca=1.0
    sens = sensitivity_df[sensitivity_df["w_ca"] == 1.0].pivot(index="w_se", columns="w_fd", values="total_net_value")
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    sns.heatmap(sens, cmap="YlGnBu", annot=True, fmt=".1f", ax=ax)
    ax.set_title("Sensitivity: Owner Value (w_ca=1.0)")
    ax.set_xlabel("w_fd")
    ax.set_ylabel("w_se")
    fig.tight_layout()
    fig.savefig(output_dir / "q4_sensitivity_heatmap.png", dpi=300)
    plt.close(fig)

    # 7) Radar: baseline vs optimized (normalized to avoid scale distortion)
    radar_cols = ["total_win_delta", "total_cf_delta", "total_net_value", "total_equity"]
    baseline = {c: 0.0 for c in radar_cols}
    # Use the selected cap (e.g., 3%) rather than the max cap to avoid a degenerate all-1 radar.
    optimal = frontier_df.iloc[(frontier_df["equity_cap"] - chosen_cap).abs().argsort()].iloc[0]
    scales = {c: max(frontier_df[c].max(), 1e-6) for c in radar_cols}
    values = [baseline[c] / scales[c] for c in radar_cols] + [baseline[radar_cols[0]] / scales[radar_cols[0]]]
    # Plot baseline at a tiny radius so it is visible (true value is 0 on all axes).
    eps = 0.05
    values_plot = [eps for _ in values]
    values2 = [optimal[c] / scales[c] for c in radar_cols] + [optimal[radar_cols[0]] / scales[radar_cols[0]]]

    angles = np.linspace(0, 2 * np.pi, len(radar_cols), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(6.2, 5.4))
    ax = fig.add_subplot(111, polar=True)
    ax.set_ylim(0, 1.05)
    ax.plot(angles, values_plot, color="#9ecae1", linewidth=2, label="Baseline (0 shown as 0.05)")
    ax.fill(angles, values_plot, color="#9ecae1", alpha=0.15)
    ax.plot(angles, values2, color="#fd8d3c", linewidth=2, label="Equity-MCKP")
    ax.fill(angles, values2, color="#fd8d3c", alpha=0.2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["Win Δ (norm)", "CF Δ (norm)", "Owner Δ (norm)", "Equity (norm)"], fontsize=9)
    ax.set_title(f"Baseline vs Equity-MCKP (Normalized, cap={chosen_cap:.2%})")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    fig.tight_layout()
    fig.savefig(output_dir / "q4_radar_comparison.png", dpi=300)
    plt.close(fig)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--equity-cap", type=float, default=0.03)
    parser.add_argument("--equity-step", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="project/experiments/output/q4_equity_mckp")
    parser.add_argument("--team", type=str, default="IND")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(use_data=True, seed=args.seed)
    env_state = env.reset()

    cfg = EquityConfig(team_code=args.team, equity_cap=args.equity_cap, equity_step=args.equity_step)

    league_df, team_df, weights = compute_skill_tables(cfg.team_code, cfg.roster_size)
    options_df, options_by_player, meta = build_tri_factor_options(league_df, team_df, cfg, env_state)

    best_val, best_b, solution_df, dp = solve_mckp(options_by_player, cfg.equity_cap, cfg.equity_step)
    summary = summarize_solution(solution_df, meta)

    # Frontier over equity caps
    frontier_rows = []
    for cap in [0.01, 0.02, 0.03, 0.04, 0.05]:
        best_val_c, best_b_c, sol_c, _ = solve_mckp(options_by_player, cap, cfg.equity_step)
        summary_c = summarize_solution(sol_c, meta)
        summary_c["equity_cap"] = cap
        frontier_rows.append(summary_c)
    frontier_df = pd.DataFrame(frontier_rows)

    # Sensitivity on weights
    sensitivity_rows = []
    for w_se in [0.7, 1.0, 1.3]:
        for w_ca in [0.7, 1.0, 1.3]:
            for w_fd in [0.7, 1.0, 1.3]:
                cfg_s = EquityConfig(
                    team_code=cfg.team_code,
                    roster_size=cfg.roster_size,
                    equity_cap=cfg.equity_cap,
                    equity_step=cfg.equity_step,
                    equity_levels=cfg.equity_levels,
                    star_quantile=cfg.star_quantile,
                    omega_scale=cfg.omega_scale,
                    value_scale=cfg.value_scale,
                    omega_peak=cfg.omega_peak,
                    omega_sigma=cfg.omega_sigma,
                    w_se=w_se,
                    w_ca=w_ca,
                    w_fd=w_fd,
                    ca_alpha=cfg.ca_alpha,
                    se_alpha=cfg.se_alpha,
                    se_gate_scale=cfg.se_gate_scale,
                    fd_alpha=cfg.fd_alpha,
                    distress_lambda0=cfg.distress_lambda0,
                    distress_cf_scale=cfg.distress_cf_scale,
                    distress_lambda_scale=cfg.distress_lambda_scale,
                    relief_factor=cfg.relief_factor,
                    relief_horizon=cfg.relief_horizon,
                    win_scale=cfg.win_scale,
                )
                opts_df_s, opts_by_player_s, meta_s = build_tri_factor_options(league_df, team_df, cfg_s, env_state)
                best_val_s, best_b_s, sol_s, _ = solve_mckp(opts_by_player_s, cfg_s.equity_cap, cfg_s.equity_step)
                summary_s = summarize_solution(sol_s, meta_s)
                sensitivity_rows.append({"w_se": w_se, "w_ca": w_ca, "w_fd": w_fd, **summary_s})
    sensitivity_df = pd.DataFrame(sensitivity_rows)

    # Save outputs
    (output_dir / "q4_equity_config.json").write_text(json.dumps(cfg.__dict__, indent=2))
    league_df.to_csv(output_dir / "q4_league_skill_table.csv", index=False)
    team_df.to_csv(output_dir / "q4_ind_skill_table.csv", index=False)
    options_df.to_csv(output_dir / "q4_equity_options.csv", index=False)
    solution_df.to_csv(output_dir / "q4_mckp_solution.csv", index=False)
    pd.DataFrame(dp).to_csv(output_dir / "q4_mckp_dp_table.csv", index=False)
    frontier_df.to_csv(output_dir / "q4_equity_frontier.csv", index=False)
    sensitivity_df.to_csv(output_dir / "q4_sensitivity_weights.csv", index=False)
    (output_dir / "q4_summary.json").write_text(json.dumps(summary, indent=2))

    # Figures
    plot_figures(
        league_df=league_df,
        team_df=team_df,
        options_df=options_df,
        solution_df=solution_df,
        frontier_df=frontier_df,
        sensitivity_df=sensitivity_df,
        output_dir=Path("newfigures"),
        star_threshold=meta["star_threshold"],
        chosen_cap=cfg.equity_cap,
    )

    print(f"Saved Q4 outputs to {output_dir}")
