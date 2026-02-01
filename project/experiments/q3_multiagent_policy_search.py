import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.data.geo import build_expansion_sites
from project.mdp.action import ActionVector, DEFAULT_ACTION
from project.mdp.mask import mutable_mask
from project.data.player_kmeans import build_player_model
from project.solvers.multi_agent_game import MultiAgentGameEnv


OUTPUT_CSV = Path("project/experiments/output/q3_multiagent_policy_search.csv")
OUTPUT_TOP = Path("project/experiments/output/q3_multiagent_policy_top.csv")


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


def eval_policy(
    env: MultiAgentGameEnv,
    ind_policy,
    other_policy,
    episodes: int,
    max_steps: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    win_vals: List[float] = []
    cf_vals: List[float] = []
    debt_vals: List[float] = []

    teams = env.team_codes
    ind_idx = teams.index("IND")

    for ep in range(episodes):
        state = env.reset(seed=int(rng.integers(0, 1_000_000)))
        for _ in range(max_steps):
            actions = []
            for i in range(len(teams)):
                if i == ind_idx:
                    actions.append(ind_policy(state, idx=i))
                else:
                    actions.append(other_policy(state, idx=i))
            state, _, done, _ = env.step(state, actions, rng=rng)
            s = state.teams[ind_idx]
            win_vals.append(float(s.R.W[0]))
            cf_vals.append(float(s.F.CF))
            debt_vals.append(float(s.F.D))
            if done:
                break

    return float(np.mean(win_vals)), float(np.mean(cf_vals)), float(np.mean(debt_vals))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=str, default="Columbus")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--two-phase", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=300)
    args = parser.parse_args()

    sites = {s["name"].lower(): s for s in build_expansion_sites()}
    if args.site.lower() not in sites:
        raise SystemExit(f"Unknown site: {args.site}")
    site = sites[args.site.lower()]

    try:
        model = build_player_model()
        teams = list(model.team_strengths.keys()) if model.team_strengths else None
    except Exception:
        teams = None
    if not teams:
        teams = ["ATL", "CHI", "CON", "DAL", "GSV", "IND", "LAS", "LVA", "MIN", "NYL", "PHO", "SEA", "WAS"]
    env = MultiAgentGameEnv(teams, expansion_site=site, seed=42, use_data=True)
    for cfg in env.team_cfgs.values():
        cfg.max_leverage = 10.0
        cfg.bankruptcy_cash = -1.0e9

    rng = np.random.default_rng(args.seed)

    # Baseline: default action for everyone.
    base_policy = phase_policy(DEFAULT_ACTION, DEFAULT_ACTION)
    base_win, base_cf, base_debt = eval_policy(env, base_policy, base_policy, args.episodes, args.max_steps, rng)

    # Candidate action sets (moderate size).
    roster_set = [2, 3, 4, 5, 6]
    salary_set = [1, 2, 3, 4, 5]
    ticket_set = [2, 3, 4]
    marketing_set = [1, 2, 3]
    debt_set = [0, 1, 2]
    equity_set = [0, 1]

    candidates: List[Tuple[ActionVector, ActionVector]] = []
    if args.two_phase:
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
    if args.max_candidates and len(candidates) > args.max_candidates:
        candidates = candidates[: args.max_candidates]

    rows: List[Dict] = []
    top_rows: List[Dict] = []

    for off_action, reg_action in candidates:
        ind_policy = phase_policy(off_action, reg_action)
        win, cf, debt = eval_policy(env, ind_policy, base_policy, args.episodes, args.max_steps, rng)
        improved = (win > base_win) and (cf > base_cf) and (debt < base_debt)
        row = {
            "site": site["name"],
            "two_phase": int(args.two_phase),
            "off_action": off_action.to_tuple(),
            "reg_action": reg_action.to_tuple(),
            "win_mean": win,
            "cf_mean": cf,
            "debt_mean": debt,
            "base_win": base_win,
            "base_cf": base_cf,
            "base_debt": base_debt,
            "improved_all": int(improved),
        }
        rows.append(row)
        if improved:
            top_rows.append(row)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if top_rows:
        with OUTPUT_TOP.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(top_rows[0].keys()))
            writer.writeheader()
            writer.writerows(top_rows)

    print(f"Baseline win={base_win:.3f}, cf={base_cf:.3f}, debt={base_debt:.3f}")
    print(f"Candidates={len(rows)}, Improved(all)={len(top_rows)}")
    print(f"Saved: {OUTPUT_CSV}, {OUTPUT_TOP if top_rows else 'no top rows'}")


if __name__ == "__main__":
    main()
