import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.data.geo import build_expansion_sites
from project.data.player_kmeans import build_player_model
from project.mdp.action import ActionVector, DEFAULT_ACTION
from project.mdp.mask import mutable_mask
from project.solvers.heuristic_value import heuristic_value
from project.solvers.mcts import MCTS
from project.solvers.multi_agent_game import BestResponseEnvMulti, MultiAgentGameEnv


OUTPUT_ACTIONS = Path("project/experiments/output/q3_multiagent_actions.csv")
OUTPUT_TEAM = Path("project/experiments/output/q3_multiagent_team_summary.csv")
OUTPUT_ENV = Path("project/experiments/output/q3_multiagent_env_trace.csv")
OUTPUT_SHIFT = Path("project/experiments/output/q3_multiagent_shift.csv")
OUTPUT_MD = Path("project/experiments/output/q3_multiagent_summary.md")


def static_policy_factory(target: ActionVector):
    def policy(game_state, idx: int = 0):
        state = game_state.teams[idx]
        mask = mutable_mask(state.Theta)
        values = target.to_list()
        for i, m in enumerate(mask):
            if m == 0:
                values[i] = state.K[i]
        return ActionVector.from_list(values)

    return policy


def phase_policy_factory(profile: Dict[str, ActionVector], fallback: ActionVector):
    def policy(game_state, idx: int = 0):
        state = game_state.teams[idx]
        action = profile.get(state.Theta, fallback)
        mask = mutable_mask(state.Theta)
        values = action.to_list()
        for i, m in enumerate(mask):
            if m == 0:
                values[i] = state.K[i]
        return ActionVector.from_list(values)

    return policy


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p / max(1e-12, np.sum(p))
    q = q / max(1e-12, np.sum(q))
    m = 0.5 * (p + q)
    def _kl(a, b):
        a = np.clip(a, 1e-12, 1.0)
        b = np.clip(b, 1e-12, 1.0)
        return float(np.sum(a * np.log(a / b)))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=str, default="Columbus")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--mcts-iter", type=int, default=60)
    parser.add_argument("--mcts-horizon", type=int, default=4)
    parser.add_argument(
        "--relax-terminal",
        action="store_true",
        help="Disable early terminal stops (useful for multi-agent rollouts).",
    )
    parser.add_argument("--team-limit", type=int, default=None, help="Optional limit on number of teams.")
    args = parser.parse_args()

    sites = {s["name"].lower(): s for s in build_expansion_sites()}
    if args.site.lower() not in sites:
        raise SystemExit(f"Unknown site: {args.site}")
    site = sites[args.site.lower()]

    model = build_player_model()
    teams = list(model.team_strengths.keys()) if model.team_strengths else ["IND"]
    if args.team_limit:
        teams = teams[: args.team_limit]

    env = MultiAgentGameEnv(teams, expansion_site=site, seed=42, use_data=True)
    n_teams = len(teams)
    if args.relax_terminal:
        for cfg in env.team_cfgs.values():
            cfg.max_leverage = 10.0
            cfg.bankruptcy_cash = -1.0e9

    # Initialize policies (baseline)
    base_pol = static_policy_factory(DEFAULT_ACTION)
    policies = [lambda gs, i=i: base_pol(gs, idx=i) for i in range(n_teams)]

    action_rows: List[Dict] = []
    env_rows: List[Dict] = []
    shift_rows: List[Dict] = []

    def rollout(policies, iter_idx: int):
        rng = np.random.default_rng(123 + iter_idx)
        for ep in range(args.episodes):
            state = env.reset(seed=rng.integers(0, 1_000_000))
            for step in range(1, args.max_steps + 1):
                actions = [policies[i](state) for i in range(n_teams)]
                # record environment trace (one row per step)
                env_rows.append(
                    {
                        "iter": iter_idx,
                        "episode": ep,
                        "step": step,
                        "year": state.teams[0].year,
                        "phase": state.teams[0].Theta,
                        "i_expansion": state.teams[0].E.i_expansion,
                        "macro": state.teams[0].E.macro,
                        "cap_growth": state.teams[0].E.cap_growth,
                        "n_star_fa": state.teams[0].E.n_star_fa,
                        "bidding_intensity": state.teams[0].E.bidding_intensity,
                    }
                )
                # record team actions
                if state.teams[0].year == 2026 and state.teams[0].Theta == "offseason":
                    for i, a in enumerate(actions):
                        s = state.teams[i]
                        action_rows.append(
                            {
                                "iter": iter_idx,
                                "episode": ep,
                                "step": step,
                                "team": teams[i],
                                "year": s.year,
                                "phase": s.Theta,
                                "mu_size": s.E.mu_size,
                                "compete_local": s.E.compete_local,
                                "n_star_fa": s.E.n_star_fa,
                                "bidding_intensity": s.E.bidding_intensity,
                                "travel_fatigue": getattr(s.E, "travel_fatigue", 0.0),
                                "leverage": s.F.leverage,
                                "cash_flow": s.F.CF,
                                "cash": s.F.Cash,
                                "franchise_value": s.F.FV,
                                "debt": s.F.D,
                                "owner_share": s.F.owner_share,
                                "win_pct": s.R.W[0],
                                "elo": s.R.ELO,
                                "syn": s.R.Syn,
                                "a_roster": a.a_roster,
                                "a_salary": a.a_salary,
                                "a_ticket": a.a_ticket,
                                "a_marketing": a.a_marketing,
                                "a_debt": a.a_debt,
                                "a_equity": a.a_equity,
                            }
                        )
                state, _, done, _ = env.step(state, actions, rng=rng)
                if done:
                    break

    # Iteration 0 baseline rollout
    rollout(policies, iter_idx=0)

    def build_profiles(iter_idx: int) -> Dict[str, Dict[str, ActionVector]]:
        profiles: Dict[str, Dict[str, ActionVector]] = {t: {} for t in teams}
        df = pd.DataFrame([r for r in action_rows if r["iter"] == iter_idx])
        if df.empty:
            return profiles
        for team in teams:
            sub = df[df["team"] == team]
            if sub.empty:
                continue
            for phase, grp in sub.groupby("phase"):
                # mode per action dimension
                values = []
                for col in ["a_roster", "a_salary", "a_ticket", "a_marketing", "a_debt", "a_equity"]:
                    counts = grp[col].value_counts()
                    top = counts[counts == counts.max()].index.tolist()
                    values.append(int(sorted(top)[0]))
                profiles[team][phase] = ActionVector(*values)
        return profiles

    profiles = build_profiles(iter_idx=0)

    # IBR iterations
    for it in range(1, args.iterations + 1):
        # build static opponent policies from previous iteration profiles
        static_policies = []
        for i, code in enumerate(teams):
            prof = profiles.get(code, {})
            static_policies.append(phase_policy_factory(prof, DEFAULT_ACTION))

        policies_eval = []
        for i in range(n_teams):
            br_env = BestResponseEnvMulti(env, static_policies, team_idx=i)
            mcts = MCTS(
                br_env,
                iterations=args.mcts_iter,
                horizon=args.mcts_horizon,
                gamma=0.95,
                value_fn=lambda gs, cfg, idx=i: heuristic_value(gs.teams[idx], cfg),
            )
            policies_eval.append(lambda gs, mcts=mcts: mcts.search(gs))
        rollout(policies_eval, iter_idx=it)
        profiles = build_profiles(iter_idx=it)

    # JS divergence vs baseline
    if action_rows:
        df = pd.DataFrame(action_rows)
        base = df[df["iter"] == 0]
        for it in range(1, args.iterations + 1):
            cur = df[df["iter"] == it]
            for team in teams:
                b = base[base["team"] == team]
                c = cur[cur["team"] == team]
                if b.empty or c.empty:
                    continue
                for col, size in [("a_debt", 5), ("a_salary", 6), ("a_roster", 7)]:
                    p = b[col].value_counts().reindex(range(size), fill_value=0).values.astype(float)
                    q = c[col].value_counts().reindex(range(size), fill_value=0).values.astype(float)
                    js = _js_divergence(p, q)
                    shift_rows.append(
                        {
                            "site": site["name"],
                            "iter": it,
                            "team": team,
                            "action_dim": col,
                            "js_divergence": js,
                            "baseline_mean": float((p * np.arange(size)).sum() / max(1.0, p.sum())),
                            "current_mean": float((q * np.arange(size)).sum() / max(1.0, q.sum())),
                        }
                    )

    OUTPUT_ACTIONS.parent.mkdir(parents=True, exist_ok=True)
    if action_rows:
        with OUTPUT_ACTIONS.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(action_rows[0].keys()))
            writer.writeheader()
            writer.writerows(action_rows)
    if env_rows:
        with OUTPUT_ENV.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(env_rows[0].keys()))
            writer.writeheader()
            writer.writerows(env_rows)
    if shift_rows:
        with OUTPUT_SHIFT.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(shift_rows[0].keys()))
            writer.writeheader()
            writer.writerows(shift_rows)

    # Team summary
    if action_rows:
        df = pd.DataFrame(action_rows)
        summary = (
            df.groupby(["iter", "team"])
            .agg(
                a_roster_mean=("a_roster", "mean"),
                a_salary_mean=("a_salary", "mean"),
                a_ticket_mean=("a_ticket", "mean"),
                a_marketing_mean=("a_marketing", "mean"),
                a_debt_mean=("a_debt", "mean"),
                a_equity_mean=("a_equity", "mean"),
                leverage_mean=("leverage", "mean"),
                cash_flow_mean=("cash_flow", "mean"),
                win_pct_mean=("win_pct", "mean"),
                debt_mean=("debt", "mean"),
            )
            .reset_index()
        )
        with OUTPUT_TEAM.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary.columns))
            writer.writeheader()
            writer.writerows(summary.to_dict(orient="records"))

    with OUTPUT_MD.open("w") as f:
        f.write("# Q3 Multi-Agent Game Summary\n\n")
        f.write(f"Site={site['name']}, Teams={len(teams)}, Iters={args.iterations}, Episodes={args.episodes}\\n")
        f.write(f"MCTS iters={args.mcts_iter}, horizon={args.mcts_horizon}.\\n")
        f.write("Outputs:\\n")
        f.write(f"- {OUTPUT_ACTIONS}\\n")
        f.write(f"- {OUTPUT_TEAM}\\n")
        f.write(f"- {OUTPUT_ENV}\\n")
        f.write(f"- {OUTPUT_SHIFT}\\n")

    print(f"Saved: {OUTPUT_ACTIONS}, {OUTPUT_TEAM}, {OUTPUT_ENV}, {OUTPUT_SHIFT}, {OUTPUT_MD}")


if __name__ == "__main__":
    main()
