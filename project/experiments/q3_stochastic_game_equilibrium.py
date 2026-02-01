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

from project.data.integration import calibrate_config_from_data
from project.data.player_kmeans import build_player_model
from project.data.geo import build_expansion_sites
from project.mdp.action import ActionVector, DEFAULT_ACTION
from project.mdp.config import MDPConfig
from project.mdp.mask import mutable_mask
from project.solvers.heuristic_value import heuristic_value
from project.solvers.mcts import MCTS
from project.solvers.stochastic_game import BestResponseEnv, StochasticGameEnv


OUTPUT_ACTIONS = Path("project/experiments/output/q3_stochastic_game_actions.csv")
OUTPUT_SHIFT = Path("project/experiments/output/q3_stochastic_game_shift.csv")
OUTPUT_MD = Path("project/experiments/output/q3_stochastic_game_summary.md")

SITES = build_expansion_sites()


def static_policy_factory(target: ActionVector):
    def policy(state):
        mask = mutable_mask(state.Theta)
        values = target.to_list()
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


def _default_opponents() -> List[str]:
    try:
        model = build_player_model()
        strengths = model.team_strengths or {}
        # Sort by ELO strength desc; exclude IND
        ranked = sorted(strengths.items(), key=lambda kv: float(kv[1].get("elo", 1500.0)), reverse=True)
        opps = [t for t, _ in ranked if t != "IND"]
        return opps[:3] if opps else ["NYL"]
    except Exception:
        return ["NYL"]


def build_game_env(site: Dict, opp_team: str, seed: int = 42) -> StochasticGameEnv:
    cfg = MDPConfig()
    cfg.expansion_market_delta = float(site["market_delta"])
    cfg.expansion_compete_delta = int(site["compete_delta"])
    cfg.expansion_media_bonus = float(site["media_bonus"])
    cfg.expansion_travel_fatigue = float(site["travel_fatigue"])
    cfg.expansion_years = [2026]
    cfg.max_equity_action = 0
    cfg, _ = calibrate_config_from_data(cfg)
    # re-apply scenario overrides
    cfg.expansion_market_delta = float(site["market_delta"])
    cfg.expansion_compete_delta = int(site["compete_delta"])
    cfg.expansion_media_bonus = float(site["media_bonus"])
    cfg.expansion_travel_fatigue = float(site["travel_fatigue"])
    cfg.expansion_years = [2026]
    cfg.max_equity_action = 0
    return StochasticGameEnv(cfg, team_ind="IND", team_opp=opp_team, seed=seed, use_data=True)


def rollout_game(env: StochasticGameEnv, policy_ind, policy_opp, episodes: int, max_steps: int, seed: int):
    rng = np.random.default_rng(seed)
    records = []
    for ep in range(episodes):
        state = env.reset(seed=rng.integers(0, 1_000_000))
        for step in range(1, max_steps + 1):
            a_ind = policy_ind(state)
            a_opp = policy_opp(state)
            # record expansion-year offseason decisions
            if state.ind.year == 2026 and state.ind.Theta == "offseason":
                records.append(
                    {
                        "episode": ep,
                        "step": step,
                        "year": state.ind.year,
                        "phase": state.ind.Theta,
                        "player": "IND",
                        "a_roster": a_ind.a_roster,
                        "a_salary": a_ind.a_salary,
                        "a_ticket": a_ind.a_ticket,
                        "a_marketing": a_ind.a_marketing,
                        "a_debt": a_ind.a_debt,
                        "a_equity": a_ind.a_equity,
                    }
                )
                records.append(
                    {
                        "episode": ep,
                        "step": step,
                        "year": state.opp.year,
                        "phase": state.opp.Theta,
                        "player": "OPP",
                        "a_roster": a_opp.a_roster,
                        "a_salary": a_opp.a_salary,
                        "a_ticket": a_opp.a_ticket,
                        "a_marketing": a_opp.a_marketing,
                        "a_debt": a_opp.a_debt,
                        "a_equity": a_opp.a_equity,
                    }
                )
            state, _, done, _ = env.step(state, a_ind, a_opp, rng=rng)
            if done:
                break
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--mcts-iter", type=int, default=120)
    parser.add_argument("--mcts-horizon", type=int, default=6)
    parser.add_argument("--site", type=str, default=None, help="Optional: run a single site scenario by name.")
    parser.add_argument(
        "--opp-list",
        type=str,
        default=None,
        help="Comma-separated opponent team codes. Default: top-3 by player-based strength.",
    )
    args = parser.parse_args()

    OUTPUT_ACTIONS.parent.mkdir(parents=True, exist_ok=True)

    action_rows: List[Dict] = []
    shift_rows: List[Dict] = []

    sites = SITES
    if args.site:
        sites = [s for s in SITES if s["name"].lower() == args.site.lower()]
        if not sites:
            raise SystemExit(f"Unknown site: {args.site}")

    opponents = _default_opponents()
    if args.opp_list:
        opponents = [x.strip().upper() for x in args.opp_list.split(",") if x.strip()]
        if not opponents:
            opponents = _default_opponents()

    for site in sites:
        for opp in opponents:
            env = build_game_env(site, opp_team=opp)

            # Iteration 0 baseline policies
            base_pol = static_policy_factory(DEFAULT_ACTION)
            pol_ind = lambda gs: base_pol(gs.ind)
            pol_opp = lambda gs: base_pol(gs.opp)
            baseline_records = rollout_game(env, pol_ind, pol_opp, args.episodes, args.max_steps, seed=0)
            for r in baseline_records:
                r.update({"site": site["name"], "opp": opp, "iter": 0})
            action_rows.extend(baseline_records)

            # Iterative Best Response to approximate MPE
            for it in range(1, args.iterations + 1):
                # IND best response to current OPP
                br_env_ind = BestResponseEnv(env, opponent_policy=pol_opp, player="ind")
                mcts_ind = MCTS(
                    br_env_ind,
                    iterations=args.mcts_iter,
                    horizon=args.mcts_horizon,
                    gamma=0.95,
                    value_fn=lambda gs, cfg: heuristic_value(gs.ind, cfg),
                )
                pol_ind = lambda gs: mcts_ind.search(gs)

                # OPP best response to current IND
                br_env_opp = BestResponseEnv(env, opponent_policy=pol_ind, player="opp")
                mcts_opp = MCTS(
                    br_env_opp,
                    iterations=args.mcts_iter,
                    horizon=args.mcts_horizon,
                    gamma=0.95,
                    value_fn=lambda gs, cfg: heuristic_value(gs.opp, cfg),
                )
                pol_opp = lambda gs: mcts_opp.search(gs)

                records = rollout_game(env, pol_ind, pol_opp, args.episodes, args.max_steps, seed=it)
                for r in records:
                    r.update({"site": site["name"], "opp": opp, "iter": it})
                action_rows.extend(records)

            # Structural shift vs baseline (JS divergence on action distributions)
            df = pd.DataFrame(action_rows)
            sub = df[(df["site"] == site["name"]) & (df["opp"] == opp)].copy()
            base = sub[sub["iter"] == 0]
            for it in range(1, args.iterations + 1):
                cur = sub[sub["iter"] == it]
                for player in ["IND", "OPP"]:
                    b = base[base["player"] == player]
                    c = cur[cur["player"] == player]
                    if b.empty or c.empty:
                        continue
                    for col, size in [("a_debt", 5), ("a_salary", 6), ("a_roster", 7)]:
                        p = b[col].value_counts().reindex(range(size), fill_value=0).values.astype(float)
                        q = c[col].value_counts().reindex(range(size), fill_value=0).values.astype(float)
                        js = _js_divergence(p, q)
                        shift_rows.append(
                            {
                                "site": site["name"],
                                "opp": opp,
                                "iter": it,
                                "player": player,
                                "action_dim": col,
                                "js_divergence": js,
                                "baseline_mean": float((p * np.arange(size)).sum() / max(1.0, p.sum())),
                                "current_mean": float((q * np.arange(size)).sum() / max(1.0, q.sum())),
                            }
                        )

    # Save action records
    if action_rows:
        with OUTPUT_ACTIONS.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(action_rows[0].keys()))
            writer.writeheader()
            writer.writerows(action_rows)

    # Save shift summary
    if shift_rows:
        with OUTPUT_SHIFT.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(shift_rows[0].keys()))
            writer.writeheader()
            writer.writerows(shift_rows)

    with OUTPUT_MD.open("w") as f:
        f.write("# Q3 Stochastic Game (Iterative Best Response) Summary\\n\\n")
        f.write("We approximate Markov Perfect Equilibrium via iterative best response (IBR).\\n")
        f.write("Structural shift is measured by JS divergence between action distributions (expansion-year offseason).\\n\\n")
        f.write(f"Iterations={args.iterations}, Episodes={args.episodes}, MCTS iters={args.mcts_iter}.\\n")
        f.write(f"Opponents={','.join(opponents)}\\n\\n")
        f.write("Outputs:\\n")
        f.write(f"- {OUTPUT_ACTIONS}\\n")
        f.write(f"- {OUTPUT_SHIFT}\\n")

    print(f"Saved: {OUTPUT_ACTIONS}, {OUTPUT_SHIFT}, {OUTPUT_MD}")


if __name__ == "__main__":
    main()
