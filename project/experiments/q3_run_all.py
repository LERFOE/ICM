import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import ast

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.data.geo import build_expansion_sites, TEAM_LOCATIONS, haversine_km
from project.data.player_kmeans import build_player_model
from project.mdp.action import ActionVector, DEFAULT_ACTION
from project.mdp.mask import mutable_mask
from project.solvers.multi_agent_game import MultiAgentGameEnv


OUTPUT_DIR = Path("project/experiments/output/q3_run")
FIG_DIR = Path("newfigures")


def phase_policy(off_action: ActionVector, reg_action: ActionVector):
    def policy(game_state, idx: int = 0):
        state = game_state.teams[idx]
        if state.Theta == "offseason":
            action = off_action
        else:
            action = reg_action
        mask = mutable_mask(state.Theta)
        values = action.to_list()
        for i, m in enumerate(mask):
            if m == 0:
                values[i] = state.K[i]
        return ActionVector.from_list(values)

    return policy


def _teams_from_players() -> List[str]:
    try:
        model = build_player_model()
        teams = list(model.team_strengths.keys()) if model.team_strengths else None
    except Exception:
        teams = None
    if not teams:
        teams = ["ATL", "CHI", "CON", "DAL", "GSV", "IND", "LAS", "LVA", "MIN", "NYL", "PHO", "SEA", "WAS"]
    return teams


def _apply_relax_terminal(env: MultiAgentGameEnv) -> None:
    for cfg in env.team_cfgs.values():
        cfg.max_leverage = 10.0
        cfg.bankruptcy_cash = -1.0e9


def _apply_no_expansion(env: MultiAgentGameEnv) -> None:
    env.base_cfg.expansion_years = []
    for cfg in env.team_cfgs.values():
        cfg.expansion_years = []


def eval_policy(
    env: MultiAgentGameEnv,
    ind_policy,
    other_policy,
    episodes: int,
    max_steps: int,
    seeds: List[int],
) -> Tuple[float, float, float]:
    win_vals: List[float] = []
    cf_vals: List[float] = []
    debt_vals: List[float] = []

    teams = env.team_codes
    ind_idx = teams.index("IND")

    for seed in seeds[:episodes]:
        state = env.reset(seed=int(seed))
        rng_ep = np.random.default_rng(int(seed) + 17)
        for _ in range(max_steps):
            actions = []
            for i in range(len(teams)):
                if i == ind_idx:
                    actions.append(ind_policy(state, idx=i))
                else:
                    actions.append(other_policy(state, idx=i))
            state, _, done, _ = env.step(state, actions, rng=rng_ep)
            s = state.teams[ind_idx]
            win_vals.append(float(s.R.W[0]))
            cf_vals.append(float(s.F.CF))
            debt_vals.append(float(s.F.D))
            if done:
                break

    return float(np.mean(win_vals)), float(np.mean(cf_vals)), float(np.mean(debt_vals))


def eval_all_teams(
    env: MultiAgentGameEnv,
    policies: Dict[str, callable],
    episodes: int,
    max_steps: int,
    seeds: List[int],
) -> pd.DataFrame:
    teams = env.team_codes
    records: Dict[str, Dict[str, List[float]]] = {
        t: {"win": [], "cf": [], "debt": [], "owner": [], "fv": []} for t in teams
    }
    for seed in seeds[:episodes]:
        state = env.reset(seed=int(seed))
        rng_ep = np.random.default_rng(int(seed) + 37)
        for _ in range(max_steps):
            actions = []
            for i, t in enumerate(teams):
                actions.append(policies[t](state, idx=i))
            state, _, done, _ = env.step(state, actions, rng=rng_ep)
            for i, t in enumerate(teams):
                s = state.teams[i]
                owner_val = float(s.F.owner_share * (s.F.FV - s.F.D))
                records[t]["win"].append(float(s.R.W[0]))
                records[t]["cf"].append(float(s.F.CF))
                records[t]["debt"].append(float(s.F.D))
                records[t]["owner"].append(owner_val)
                records[t]["fv"].append(float(s.F.FV))
            if done:
                break
    rows = []
    for t in teams:
        rows.append(
            {
                "team": t,
                "win_mean": float(np.mean(records[t]["win"])),
                "cf_mean": float(np.mean(records[t]["cf"])),
                "debt_mean": float(np.mean(records[t]["debt"])),
                "owner_mean": float(np.mean(records[t]["owner"])),
                "fv_mean": float(np.mean(records[t]["fv"])),
            }
        )
    return pd.DataFrame(rows)


def candidate_actions(rng: np.random.Generator, max_candidates: int, two_phase: bool) -> List[Tuple[ActionVector, ActionVector]]:
    roster_set = [3, 4, 5, 6]
    salary_set = [1, 2, 3]
    ticket_set = [2, 3, 4]
    marketing_set = [1, 2]
    debt_set = [0, 1]
    equity_set = [0, 1]
    candidates: List[Tuple[ActionVector, ActionVector]] = []
    if two_phase:
        for r in roster_set:
            for s in salary_set:
                for d in debt_set:
                    for e in equity_set:
                        off = ActionVector(r, s, 2, 1, d, e)
                        for t in ticket_set:
                            for m in marketing_set:
                                reg = ActionVector(3, s, t, m, d, e)
                                candidates.append((off, reg))
    else:
        for r in roster_set:
            for s in salary_set:
                for t in ticket_set:
                    for m in marketing_set:
                        for d in debt_set:
                            for e in equity_set:
                                a = ActionVector(r, s, t, m, d, e)
                                candidates.append((a, a))
    rng.shuffle(candidates)
    return candidates[:max_candidates]


def select_best_policy(
    env: MultiAgentGameEnv,
    episodes: int,
    max_steps: int,
    seeds: List[int],
    max_candidates: int,
    two_phase: bool,
    rounds: int,
) -> Tuple[Dict, List[Dict]]:
    base_policy = phase_policy(DEFAULT_ACTION, DEFAULT_ACTION)
    base_win, base_cf, base_debt = eval_policy(env, base_policy, base_policy, episodes, max_steps, seeds)

    rows: List[Dict] = []
    best_row = None
    best_score = -1e9

    for round_idx in range(max(1, rounds)):
        rng_local = np.random.default_rng(seeds[0] + 991 + round_idx * 1000)
        for off_action, reg_action in candidate_actions(rng_local, max_candidates, two_phase):
            ind_policy = phase_policy(off_action, reg_action)
            win, cf, debt = eval_policy(env, ind_policy, base_policy, episodes, max_steps, seeds)
            improved = (win > base_win) and (cf > base_cf) and (debt < base_debt)
            score = (win - base_win) * 100.0 + (cf - base_cf) + (base_debt - debt) * 0.1
            row = {
                "off_action": off_action.to_tuple(),
                "reg_action": reg_action.to_tuple(),
                "win_mean": win,
                "cf_mean": cf,
                "debt_mean": debt,
                "base_win": base_win,
                "base_cf": base_cf,
                "base_debt": base_debt,
                "improved_all": int(improved),
                "score": score,
            }
            rows.append(row)
            if improved and score > best_score:
                best_score = score
                best_row = row

    if best_row is None:
        best_row = sorted(rows, key=lambda r: r["score"], reverse=True)[0]
    return best_row, rows


def make_figures(policy_df: pd.DataFrame, ind_df: pd.DataFrame, impact_df: pd.DataFrame, site_df: pd.DataFrame):
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

    # Heatmap of action vectors by site
    action_cols = ["a_roster", "a_salary", "a_ticket", "a_marketing", "a_debt", "a_equity"]
    heat = policy_df[["site"] + action_cols].copy()
    heat = heat.set_index("site")
    plt.figure(figsize=(8, 4))
    if sns is not None:
        sns.heatmap(heat, annot=True, fmt=".0f", cmap="viridis")
    else:
        plt.imshow(heat.values, aspect="auto")
        plt.yticks(range(len(heat.index)), heat.index)
        plt.xticks(range(len(action_cols)), action_cols, rotation=45, ha="right")
    plt.title("IND Policy by Expansion Site (Offseason Action Vector)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "q3_policy_heatmap.png", dpi=200)
    plt.close()

    # IND deltas by site
    plt.figure(figsize=(8, 4))
    x = np.arange(len(ind_df["site"]))
    width = 0.25
    plt.bar(x - width, ind_df["win_delta"], width, label="Win% Δ")
    plt.bar(x, ind_df["cf_delta"], width, label="CF Δ")
    plt.bar(x + width, ind_df["debt_delta"], width, label="Debt Δ (positive=lower)")
    plt.xticks(x, ind_df["site"], rotation=30, ha="right")
    plt.legend()
    plt.title("IND Expansion-Year Delta vs Baseline")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "q3_ind_deltas.png", dpi=200)
    plt.close()

    # League impact distribution per site (owner value)
    plt.figure(figsize=(8, 4))
    if sns is not None:
        sns.boxplot(data=impact_df, x="site", y="owner_delta")
    else:
        # fallback: simple scatter
        for idx, site in enumerate(sorted(impact_df["site"].unique())):
            vals = impact_df[impact_df["site"] == site]["owner_delta"].values
            plt.scatter(np.full_like(vals, idx), vals, alpha=0.6)
        plt.xticks(range(len(sorted(impact_df["site"].unique()))), sorted(impact_df["site"].unique()), rotation=30)
    plt.title("League Owner Value Impact by Expansion Site")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "q3_league_owner_impact.png", dpi=200)
    plt.close()

    # Site summary scatter: distance vs IND owner impact
    plt.figure(figsize=(6, 4))
    plt.scatter(site_df["dist_km"], site_df["ind_owner_delta"], s=60)
    for _, row in site_df.iterrows():
        plt.text(row["dist_km"], row["ind_owner_delta"], row["site"], fontsize=8, ha="left")
    plt.xlabel("Distance from IND (km)")
    plt.ylabel("IND Owner Value Δ")
    plt.title("Site Distance vs IND Owner Impact")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "q3_site_distance_vs_ind.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", type=str, default="Columbus,StLouis,Nashville,Denver,Portland,Toronto")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-candidates", type=int, default=40)
    parser.add_argument("--two-phase", action="store_true")
    parser.add_argument("--search-rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    site_names = [s.strip() for s in args.sites.split(",") if s.strip()]
    sites = {s["name"].lower(): s for s in build_expansion_sites()}
    chosen_sites = []
    for name in site_names:
        key = name.lower()
        if key not in sites:
            raise SystemExit(f"Unknown site: {name}")
        chosen_sites.append(sites[key])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    seeds = [int(s) for s in rng.integers(0, 1_000_000, size=max(args.episodes, 3))]

    policy_rows: List[Dict] = []
    policy_search_rows: List[Dict] = []
    ind_eval_rows: List[Dict] = []
    impact_rows: List[Dict] = []
    site_summary_rows: List[Dict] = []

    teams = _teams_from_players()

    for site in chosen_sites:
        env = MultiAgentGameEnv(teams, expansion_site=site, seed=42, use_data=True)
        _apply_relax_terminal(env)
        best_row, search_rows = select_best_policy(
            env,
            episodes=args.episodes,
            max_steps=args.max_steps,
            seeds=seeds,
            max_candidates=args.max_candidates,
            two_phase=args.two_phase,
            rounds=args.search_rounds,
        )
        best_row["site"] = site["name"]
        policy_rows.append(best_row)
        for r in search_rows:
            r["site"] = site["name"]
            policy_search_rows.append(r)

        # Evaluate IND policy vs baseline
        off_tuple = best_row["off_action"]
        reg_tuple = best_row["reg_action"]
        if isinstance(off_tuple, str):
            off_tuple = ast.literal_eval(off_tuple)
        if isinstance(reg_tuple, str):
            reg_tuple = ast.literal_eval(reg_tuple)
        off_action = ActionVector(*off_tuple)
        reg_action = ActionVector(*reg_tuple)
        ind_policy = phase_policy(off_action, reg_action)
        base_policy = phase_policy(DEFAULT_ACTION, DEFAULT_ACTION)
        win, cf, debt = eval_policy(env, ind_policy, base_policy, args.episodes, args.max_steps, seeds)
        base_win, base_cf, base_debt = best_row["base_win"], best_row["base_cf"], best_row["base_debt"]
        ind_eval_rows.append(
            {
                "site": site["name"],
                "off_action": off_action.to_tuple(),
                "reg_action": reg_action.to_tuple(),
                "win_mean": win,
                "cf_mean": cf,
                "debt_mean": debt,
                "base_win": base_win,
                "base_cf": base_cf,
                "base_debt": base_debt,
                "win_delta": win - base_win,
                "cf_delta": cf - base_cf,
                "debt_delta": base_debt - debt,
            }
        )

        # League impact: expansion vs no expansion
        policies = {t: base_policy for t in teams}
        df_exp = eval_all_teams(env, policies, args.episodes, args.max_steps, seeds)
        env_no = MultiAgentGameEnv(teams, expansion_site=site, seed=24, use_data=True)
        _apply_relax_terminal(env_no)
        _apply_no_expansion(env_no)
        df_no = eval_all_teams(env_no, policies, args.episodes, args.max_steps, seeds)
        df = df_exp.merge(df_no, on="team", suffixes=("_exp", "_no"))
        for _, row in df.iterrows():
            impact_rows.append(
                {
                    "site": site["name"],
                    "team": row["team"],
                    "win_delta": row["win_mean_exp"] - row["win_mean_no"],
                    "cf_delta": row["cf_mean_exp"] - row["cf_mean_no"],
                    "owner_delta": row["owner_mean_exp"] - row["owner_mean_no"],
                }
            )

        # Site summary for IND
        ind_row = df[df["team"] == "IND"].iloc[0]
        ind_owner_delta = float(ind_row["owner_mean_exp"] - ind_row["owner_mean_no"])
        ind_loc = TEAM_LOCATIONS.get("IND")
        dist = np.nan
        if ind_loc is not None:
            dist = haversine_km(ind_loc.lat, ind_loc.lon, site["lat"], site["lon"])
        site_summary_rows.append(
            {
                "site": site["name"],
                "dist_km": dist,
                "ind_owner_delta": ind_owner_delta,
            }
        )

    # Save outputs
    policy_df = pd.DataFrame(policy_rows)
    def _as_tuple(val):
        if isinstance(val, str):
            return ast.literal_eval(val)
        return tuple(val)
    policy_df["a_roster"] = policy_df["off_action"].apply(lambda x: _as_tuple(x)[0])
    policy_df["a_salary"] = policy_df["off_action"].apply(lambda x: _as_tuple(x)[1])
    policy_df["a_ticket"] = policy_df["off_action"].apply(lambda x: _as_tuple(x)[2])
    policy_df["a_marketing"] = policy_df["off_action"].apply(lambda x: _as_tuple(x)[3])
    policy_df["a_debt"] = policy_df["off_action"].apply(lambda x: _as_tuple(x)[4])
    policy_df["a_equity"] = policy_df["off_action"].apply(lambda x: _as_tuple(x)[5])

    policy_df.to_csv(OUTPUT_DIR / "q3_policy_best_by_site.csv", index=False)
    pd.DataFrame(policy_search_rows).to_csv(OUTPUT_DIR / "q3_policy_search_all.csv", index=False)
    pd.DataFrame(ind_eval_rows).to_csv(OUTPUT_DIR / "q3_ind_policy_eval.csv", index=False)
    pd.DataFrame(impact_rows).to_csv(OUTPUT_DIR / "q3_league_impact_by_site.csv", index=False)
    site_df = pd.DataFrame(site_summary_rows)
    site_df.to_csv(OUTPUT_DIR / "q3_site_summary.csv", index=False)

    make_figures(
        policy_df=policy_df,
        ind_df=pd.DataFrame(ind_eval_rows),
        impact_df=pd.DataFrame(impact_rows),
        site_df=site_df,
    )

    print("Q3 run complete.")
    print(f"Outputs in: {OUTPUT_DIR}")
    print(f"Figures in: {FIG_DIR}")


if __name__ == "__main__":
    main()
