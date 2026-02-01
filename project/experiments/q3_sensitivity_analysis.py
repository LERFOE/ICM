import argparse
import ast
import csv
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.data.geo import build_expansion_sites
from project.data.player_kmeans import build_player_model
from project.mdp.action import ActionVector, DEFAULT_ACTION
from project.mdp.mask import mutable_mask
from project.solvers.multi_agent_game import MultiAgentGameEnv


OUT_DIR = Path("project/experiments/output/q3_sensitivity")
FIG_DIR = Path("newfigures")


def _teams_from_players() -> List[str]:
    try:
        model = build_player_model()
        teams = list(model.team_strengths.keys()) if model.team_strengths else None
    except Exception:
        teams = None
    if not teams:
        teams = ["ATL", "CHI", "CON", "DAL", "GSV", "IND", "LAS", "LVA", "MIN", "NYL", "PHO", "SEA", "WAS"]
    return teams


def _relax_terminal(env: MultiAgentGameEnv) -> None:
    for cfg in env.team_cfgs.values():
        cfg.max_leverage = 10.0
        cfg.bankruptcy_cash = -1.0e9


def _apply_no_expansion(env: MultiAgentGameEnv) -> None:
    env.base_cfg.expansion_years = []
    for cfg in env.team_cfgs.values():
        cfg.expansion_years = []


def _apply_overrides(env: MultiAgentGameEnv, overrides: Dict) -> None:
    # Apply to shared base config and per-team configs.
    for k, v in overrides.items():
        setattr(env.base_cfg, k, v)
    for code, cfg in env.team_cfgs.items():
        for k, v in overrides.items():
            setattr(cfg, k, v)


def _phase_policy(off_action: ActionVector, reg_action: ActionVector):
    def policy(game_state, idx: int = 0):
        state = game_state.teams[idx]
        action = off_action if state.Theta == "offseason" else reg_action
        mask = mutable_mask(state.Theta)
        values = action.to_list()
        for i, m in enumerate(mask):
            if m == 0:
                values[i] = state.K[i]
        return ActionVector.from_list(values)

    return policy


def _eval_policy(
    env: MultiAgentGameEnv,
    ind_pol,
    other_pol,
    seeds: List[int],
    max_steps: int,
) -> Dict[str, float]:
    teams = env.team_codes
    ind_idx = teams.index("IND")

    win_vals: List[float] = []
    cf_vals: List[float] = []
    debt_vals: List[float] = []
    owner_vals: List[float] = []

    for seed in seeds:
        state = env.reset(seed=int(seed))
        rng_ep = np.random.default_rng(int(seed) + 77)
        for _ in range(max_steps):
            actions = []
            for i in range(len(teams)):
                if i == ind_idx:
                    actions.append(ind_pol(state, idx=i))
                else:
                    actions.append(other_pol(state, idx=i))
            state, _, done, _ = env.step(state, actions, rng=rng_ep)
            s = state.teams[ind_idx]
            win_vals.append(float(s.R.W[0]))
            cf_vals.append(float(s.F.CF))
            debt_vals.append(float(s.F.D))
            owner_vals.append(float(s.F.owner_share * (s.F.FV - s.F.D)))
            if done:
                break

    return {
        "win_mean": float(np.mean(win_vals)),
        "cf_mean": float(np.mean(cf_vals)),
        "debt_mean": float(np.mean(debt_vals)),
        "owner_mean": float(np.mean(owner_vals)),
    }


def _eval_all_teams(
    env: MultiAgentGameEnv,
    policies: Dict[str, callable],
    seeds: List[int],
    max_steps: int,
) -> pd.DataFrame:
    teams = env.team_codes
    records: Dict[str, Dict[str, List[float]]] = {
        t: {"owner": [], "win": [], "cf": [], "debt": []} for t in teams
    }
    for seed in seeds:
        state = env.reset(seed=int(seed))
        rng_ep = np.random.default_rng(int(seed) + 97)
        for _ in range(max_steps):
            actions = []
            for i, t in enumerate(teams):
                actions.append(policies[t](state, idx=i))
            state, _, done, _ = env.step(state, actions, rng=rng_ep)
            for i, t in enumerate(teams):
                s = state.teams[i]
                records[t]["owner"].append(float(s.F.owner_share * (s.F.FV - s.F.D)))
                records[t]["win"].append(float(s.R.W[0]))
                records[t]["cf"].append(float(s.F.CF))
                records[t]["debt"].append(float(s.F.D))
            if done:
                break
    rows = []
    for t in teams:
        rows.append(
            {
                "team": t,
                "owner_mean": float(np.mean(records[t]["owner"])),
                "win_mean": float(np.mean(records[t]["win"])),
                "cf_mean": float(np.mean(records[t]["cf"])),
                "debt_mean": float(np.mean(records[t]["debt"])),
            }
        )
    return pd.DataFrame(rows)


def _load_best_policy_by_site() -> pd.DataFrame:
    path = Path("project/experiments/output/q3_run/q3_policy_best_by_site.csv")
    if not path.exists():
        raise SystemExit(f"Missing file: {path}. Run q3_run_all.py first.")
    df = pd.read_csv(path)
    # Normalize tuple columns
    df["off_action"] = df["off_action"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else tuple(x))
    df["reg_action"] = df["reg_action"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else tuple(x))
    return df


def _make_figures(df: pd.DataFrame) -> None:
    import os

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
    except Exception:
        sns = None

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Summary per parameter/value (average across sites)
    g = df.groupby(["param", "value"]).agg(
        owner_delta_mean=("owner_delta", "mean"),
        win_delta_mean=("win_delta", "mean"),
        cf_delta_mean=("cf_delta", "mean"),
        debt_delta_mean=("debt_delta", "mean"),
        improved_rate=("improved_all", "mean"),
    ).reset_index()

    # Lines: owner_delta vs value for each parameter
    params = g["param"].unique().tolist()
    n = len(params)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows), sharey=False)
    axes = np.array(axes).reshape(rows, cols)
    for i, p in enumerate(params):
        ax = axes[i // cols, i % cols]
        sub = g[g["param"] == p].sort_values("value")
        ax.plot(sub["value"], sub["owner_delta_mean"], marker="o")
        ax.axhline(0, color="gray", lw=1)
        ax.set_title(p)
        ax.set_xlabel("value")
        ax.set_ylabel("mean ΔOwnerValue (IND policy - baseline)")
    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis("off")
    fig.suptitle("Q3 Sensitivity (OFAT): Indiana OwnerValue Delta vs Parameters", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "q3_sensitivity_owner_lines.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Tornado: range between min and max for each parameter
    ranges = g.groupby("param")["owner_delta_mean"].agg(["min", "max"]).reset_index()
    ranges["mid"] = 0.5 * (ranges["min"] + ranges["max"])
    ranges = ranges.sort_values("max", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hlines(y=ranges["param"], xmin=ranges["min"], xmax=ranges["max"], color="#4c72b0", lw=6, alpha=0.85)
    ax.plot(ranges["mid"], ranges["param"], "o", color="#222222", ms=4)
    ax.axvline(0, color="gray", lw=1)
    ax.set_xlabel("mean ΔOwnerValue range (min..max)")
    ax.set_title("Q3 Sensitivity Tornado (IND OwnerValue Delta)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "q3_sensitivity_owner_tornado.png", dpi=200)
    plt.close(fig)

    # Heatmap: improved_rate by parameter/value
    pivot = g.pivot_table(index="param", columns="value", values="improved_rate", aggfunc="mean").fillna(0.0)
    fig, ax = plt.subplots(figsize=(9.5, 3.2))
    if sns is not None:
        sns.heatmap(pivot, ax=ax, cmap="YlGnBu", annot=True, fmt=".2f", cbar_kws={"label": "P(improved_all)"})
    else:
        ax.imshow(pivot.values, aspect="auto")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(x) for x in pivot.columns])
    ax.set_title("Q3 Sensitivity: Success Rate of Triple-Improve Constraint")
    ax.set_xlabel("value")
    ax.set_ylabel("parameter")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "q3_sensitivity_success_heatmap.png", dpi=200)
    plt.close(fig)


def _make_expansion_figures(df: pd.DataFrame) -> None:
    import os

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
    except Exception:
        sns = None

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    def _heatmap(metric: str, title: str, fname: str) -> None:
        ind = df[df["team"] == "IND"].copy()
        if ind.empty:
            return
        params = ind["param"].unique().tolist()
        cols = 3
        rows = int(np.ceil(len(params) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(12.5, 3.6 * rows), sharey=False)
        axes = np.array(axes).reshape(rows, cols)
        for i, p in enumerate(params):
            ax = axes[i // cols, i % cols]
            sub = ind[ind["param"] == p].copy()
            pivot = sub.pivot_table(index="site", columns="value", values=metric, aggfunc="mean")
            pivot = pivot.reindex(sorted(pivot.index))
            pivot = pivot.reindex(sorted(pivot.columns), axis=1)
            center = float(np.nanmean(pivot.values))
            if sns is not None:
                sns.heatmap(pivot, ax=ax, cmap="RdYlGn", center=center, annot=True, fmt=".2f")
            else:
                ax.imshow(pivot.values, aspect="auto")
                ax.set_yticks(range(len(pivot.index)))
                ax.set_yticklabels(pivot.index)
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_xticklabels([str(x) for x in pivot.columns])
            ax.set_title(p)
            ax.set_xlabel("value")
            ax.set_ylabel("site")
        for j in range(len(params), rows * cols):
            axes[j // cols, j % cols].axis("off")
        fig.suptitle(title, y=1.02)
        fig.tight_layout()
        fig.savefig(FIG_DIR / fname, dpi=200, bbox_inches="tight")
        plt.close(fig)

    _heatmap(
        metric="owner_delta",
        title="Q3 Sensitivity (OFAT): Indiana ΔOwnerValue(expansion vs no-expansion)",
        fname="q3_sensitivity_ind_owner_heatmaps.png",
    )
    _heatmap(
        metric="cf_delta",
        title="Q3 Sensitivity (OFAT): Indiana ΔCF(expansion vs no-expansion)",
        fname="q3_sensitivity_ind_cf_heatmaps.png",
    )
    _heatmap(
        metric="win_delta",
        title="Q3 Sensitivity (OFAT): Indiana ΔWin%(expansion vs no-expansion)",
        fname="q3_sensitivity_ind_win_heatmaps.png",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", type=str, default="Columbus,StLouis,Nashville,Denver,Portland,Toronto")
    parser.add_argument("--episodes", type=int, default=1, help="number of seeds used per scenario")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    best = _load_best_policy_by_site()

    sites_dict = {s["name"].lower(): s for s in build_expansion_sites()}
    chosen_sites = []
    for name in [x.strip() for x in args.sites.split(",") if x.strip()]:
        key = name.lower()
        if key not in sites_dict:
            raise SystemExit(f"Unknown site: {name}")
        chosen_sites.append(sites_dict[key])

    teams = _teams_from_players()
    rng = np.random.default_rng(args.seed)
    seeds = [int(x) for x in rng.integers(0, 1_000_000, size=max(args.episodes, 1))]

    # OFAT parameter grid
    grid = [
        ("win_eta_fatigue", [0.3, 0.6, 0.9]),
        ("travel_cost_beta", [0.6, 1.2, 1.8]),
        ("compete_revenue_beta", [0.02, 0.04, 0.06]),
        ("expansion_bidding_delta", [1.0, 2.0, 3.0]),
        ("expansion_star_fa_delta", [-1, -2, -3]),
        ("expansion_media_bonus", [0.00, 0.03, 0.06]),
    ]

    rows: List[Dict] = []
    impact_rows: List[Dict] = []

    for site in chosen_sites:
        # Best policy (from baseline) for this site
        row_pol = best[best["site"] == site["name"]]
        if row_pol.empty:
            continue
        off = ActionVector(*tuple(row_pol.iloc[0]["off_action"]))
        reg = ActionVector(*tuple(row_pol.iloc[0]["reg_action"]))
        pol_ind = _phase_policy(off, reg)
        pol_base = _phase_policy(DEFAULT_ACTION, DEFAULT_ACTION)

        # Baseline env for this site; we reuse it but override config per sweep.
        env = MultiAgentGameEnv(teams, expansion_site=site, seed=42, use_data=True)
        _relax_terminal(env)

        # Snapshot baseline config values so we can restore after each sweep.
        base_vals = {k: getattr(env.base_cfg, k) for k, _ in grid}

        for param, values in grid:
            for v in values:
                # Restore baseline for all parameters
                _apply_overrides(env, base_vals)
                _apply_overrides(env, {param: v})

                # Evaluate
                ind_metrics = _eval_policy(env, pol_ind, pol_base, seeds, args.max_steps)
                base_metrics = _eval_policy(env, pol_base, pol_base, seeds, args.max_steps)

                row = {
                    "site": site["name"],
                    "param": param,
                    "value": float(v),
                    "off_action": off.to_tuple(),
                    "reg_action": reg.to_tuple(),
                    "win_mean": ind_metrics["win_mean"],
                    "cf_mean": ind_metrics["cf_mean"],
                    "debt_mean": ind_metrics["debt_mean"],
                    "owner_mean": ind_metrics["owner_mean"],
                    "base_win": base_metrics["win_mean"],
                    "base_cf": base_metrics["cf_mean"],
                    "base_debt": base_metrics["debt_mean"],
                    "base_owner": base_metrics["owner_mean"],
                    "win_delta": ind_metrics["win_mean"] - base_metrics["win_mean"],
                    "cf_delta": ind_metrics["cf_mean"] - base_metrics["cf_mean"],
                    "debt_delta": base_metrics["debt_mean"] - ind_metrics["debt_mean"],  # positive is better
                    "owner_delta": ind_metrics["owner_mean"] - base_metrics["owner_mean"],
                }
                row["improved_all"] = int(row["win_delta"] > 0 and row["cf_delta"] > 0 and row["debt_delta"] > 0)
                rows.append(row)

        # Expansion vs no-expansion sensitivity (league-wide, baseline policy)
        pol_base = _phase_policy(DEFAULT_ACTION, DEFAULT_ACTION)
        policies = {t: pol_base for t in teams}
        base_vals = {k: getattr(env.base_cfg, k) for k, _ in grid}
        for param, values in grid:
            for v in values:
                _apply_overrides(env, base_vals)
                _apply_overrides(env, {param: v})
                df_exp = _eval_all_teams(env, policies, seeds, args.max_steps)

                env_no = MultiAgentGameEnv(teams, expansion_site=site, seed=24, use_data=True)
                _relax_terminal(env_no)
                _apply_no_expansion(env_no)
                _apply_overrides(env_no, base_vals)
                _apply_overrides(env_no, {param: v})
                df_no = _eval_all_teams(env_no, policies, seeds, args.max_steps)

                merged = df_exp.merge(df_no, on="team", suffixes=("_exp", "_no"))
                for _, r in merged.iterrows():
                    impact_rows.append(
                        {
                            "site": site["name"],
                            "param": param,
                            "value": float(v),
                            "team": r["team"],
                            "owner_delta": float(r["owner_mean_exp"]) - float(r["owner_mean_no"]),
                            "win_delta": float(r["win_mean_exp"]) - float(r["win_mean_no"]),
                            "cf_delta": float(r["cf_mean_exp"]) - float(r["cf_mean_no"]),
                        }
                    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "q3_sensitivity_metrics.csv", index=False)

    # Also write a compact summary table for paper tables.
    summary = (
        df.groupby(["param", "value"])
        .agg(
            owner_delta_mean=("owner_delta", "mean"),
            owner_delta_std=("owner_delta", "std"),
            win_delta_mean=("win_delta", "mean"),
            cf_delta_mean=("cf_delta", "mean"),
            debt_delta_mean=("debt_delta", "mean"),
            improved_rate=("improved_all", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "q3_sensitivity_summary.csv", index=False)

    _make_figures(df)

    # Expansion impact sensitivity
    if impact_rows:
        df_imp = pd.DataFrame(impact_rows)
        df_imp.to_csv(OUT_DIR / "q3_sensitivity_expansion_impact.csv", index=False)
        _make_expansion_figures(df_imp)

    print("Saved:")
    print(f"- {OUT_DIR / 'q3_sensitivity_metrics.csv'}")
    print(f"- {OUT_DIR / 'q3_sensitivity_summary.csv'}")
    if impact_rows:
        print(f"- {OUT_DIR / 'q3_sensitivity_expansion_impact.csv'}")
    print("Figures:")
    print(f"- {FIG_DIR / 'q3_sensitivity_owner_lines.png'}")
    print(f"- {FIG_DIR / 'q3_sensitivity_owner_tornado.png'}")
    print(f"- {FIG_DIR / 'q3_sensitivity_success_heatmap.png'}")
    if impact_rows:
        print(f"- {FIG_DIR / 'q3_sensitivity_ind_owner_heatmaps.png'}")


if __name__ == "__main__":
    main()
