import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.experiments.utils import build_env
from project.mdp.action import ActionVector
from project.mdp.mask import mutable_mask
from project.mdp.reward import terminal_value
from project.solvers.mcts import MCTS
from project.solvers.heuristic_value import heuristic_value

OUTPUT_DETAIL = Path("project/experiments/output/q1_leverage_policy_test_detail.csv")
OUTPUT_SUMMARY = Path("project/experiments/output/q1_leverage_policy_test_summary.csv")


CONSERVATIVE_TARGET = ActionVector(3, 2, 2, 1, 1, 0)  # hold roster, moderate salary, deleverage, no equity


def conservative_action(state, env, target: ActionVector) -> ActionVector:
    mask = mutable_mask(state.Theta)
    values = target.to_list()
    for i, m in enumerate(mask):
        if m == 0:
            values[i] = state.K[i]
    action = ActionVector.from_list(values)
    if action in env.valid_actions(state):
        return action
    valid = env.valid_actions(state)
    target_list = target.to_list()

    def dist(a: ActionVector) -> int:
        return sum(abs(x - y) for x, y in zip(a.to_list(), target_list))

    return min(valid, key=dist)


def static_policy_factory(env, target: ActionVector):
    def policy(_state):
        return conservative_action(_state, env, target)

    return policy


def mcts_policy_factory(env, iterations=220, horizon=12, gamma=0.95, rollout_candidates=7):
    mcts = MCTS(
        env,
        iterations=iterations,
        horizon=horizon,
        gamma=gamma,
        value_fn=heuristic_value,
        rollout_candidates=rollout_candidates,
    )

    def policy(state):
        return mcts.search(state)

    return policy


def collect_metrics(state) -> Dict[str, float]:
    return {
        "leverage": state.F.leverage,
        "cash_flow": state.F.CF,
        "cash": state.F.Cash,
        "franchise_value": state.F.FV,
        "debt": state.F.D,
        "owner_share": state.F.owner_share,
        "terminal_value": terminal_value(state.F),
        "win_pct": float(state.R.W[0]),
        "elo": float(state.R.ELO),
        "syn": float(state.R.Syn),
    }


def run_episode(env, policy, seasons: int, seed: int) -> Dict[int, Dict[str, float]]:
    state = env.reset(seed=seed, use_data=True)
    rng = np.random.default_rng(seed)

    if policy is None:
        fixed_action = ActionVector(*state.K)
        if fixed_action not in env.valid_actions(state):
            fixed_action = env.valid_actions(state)[0]
        policy = static_policy_factory(fixed_action)

    record_steps = {4, 8, 12}
    max_steps = seasons * 4
    results = {}

    for step in range(1, max_steps + 1):
        action = policy(state)
        state, _, done, _ = env.step(state, action, rng)
        if step in record_steps:
            results[step] = collect_metrics(state)
        if done:
            # Fill remaining horizons with last state snapshot
            for s in record_steps:
                if s >= step and s not in results:
                    results[s] = collect_metrics(state)
            break

    return results


def summarize(rows: List[Dict]) -> List[Dict]:
    summary = []
    for policy in sorted(set(r["policy"] for r in rows)):
        for seasons in [1, 2, 3]:
            subset = [r for r in rows if r["policy"] == policy and r["seasons"] == seasons]
            if not subset:
                continue
            metrics = [
                "leverage",
                "cash_flow",
                "cash",
                "franchise_value",
                "debt",
                "owner_share",
                "terminal_value",
                "win_pct",
                "elo",
                "syn",
            ]
            row = {"policy": policy, "seasons": seasons, "n": len(subset)}
            for m in metrics:
                vals = np.array([s[m] for s in subset], dtype=float)
                row[f"{m}_mean"] = float(np.mean(vals))
                row[f"{m}_std"] = float(np.std(vals))
            summary.append(row)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=140)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--rollout-candidates", type=int, default=7)
    parser.add_argument("--w-win", type=float, default=None)
    parser.add_argument("--w-win-pct", type=float, default=None)
    parser.add_argument("--w-val", type=float, default=None)
    parser.add_argument("--w-cf", type=float, default=None)
    parser.add_argument("--leverage-soft-penalty", type=float, default=None)
    parser.add_argument("--leverage-hard-penalty", type=float, default=None)
    parser.add_argument("--win-eta3", type=float, default=None)
    parser.add_argument("--syn-penalty", type=float, default=None)
    parser.add_argument("--syn-recovery", type=float, default=None)
    args = parser.parse_args()

    env = build_env(use_data=True, seed=42)
    cfg = env.config
    if args.w_win is not None:
        cfg.w_win = args.w_win
    if args.w_win_pct is not None:
        cfg.w_win_pct = args.w_win_pct
    if args.w_val is not None:
        cfg.w_val = args.w_val
    if args.w_cf is not None:
        cfg.w_cf = args.w_cf
    if args.leverage_soft_penalty is not None:
        cfg.leverage_soft_penalty = args.leverage_soft_penalty
    if args.leverage_hard_penalty is not None:
        cfg.leverage_hard_penalty = args.leverage_hard_penalty
    if args.win_eta3 is not None:
        cfg.win_eta3 = args.win_eta3
    if args.syn_penalty is not None:
        cfg.syn_penalty = args.syn_penalty
    if args.syn_recovery is not None:
        cfg.syn_recovery = args.syn_recovery

    seeds = list(range(args.seeds))
    policies = {
        "static": static_policy_factory(env, CONSERVATIVE_TARGET),
        "mcts": mcts_policy_factory(
            env,
            iterations=args.iterations,
            horizon=args.horizon,
            gamma=args.gamma,
            rollout_candidates=args.rollout_candidates,
        ),
    }

    rows = []
    for seed in seeds:
        for policy_name, policy in policies.items():
            results = run_episode(env, policy, seasons=3, seed=seed)
            for step, metrics in results.items():
                seasons = step // 4
                row = {"seed": seed, "policy": policy_name, "seasons": seasons}
                row.update(metrics)
                rows.append(row)

    OUTPUT_DETAIL.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_DETAIL.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    with OUTPUT_SUMMARY.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    print(f"Saved detail to {OUTPUT_DETAIL}")
    print(f"Saved summary to {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
