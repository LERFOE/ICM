import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.data.integration import calibrate_config_from_data
from project.data.player_kmeans import build_player_model
from project.data.geo import build_expansion_sites
from project.mdp.action import ActionVector, DEFAULT_ACTION
from project.mdp.config import MDPConfig
from project.mdp.env import MDPEnv
from project.mdp.mask import action_space_per_dim, mutable_mask
from project.mdp.reward import terminal_value
from project.mdp.phase import PHASE_OFFSEASON
from project.solvers.rl_ppo import PPOAgent, PPOConfig


OUTPUT_DETAIL = Path("project/experiments/output/q3_policy_comparison_detail.csv")
OUTPUT_SUMMARY = Path("project/experiments/output/q3_policy_comparison_summary.csv")
OUTPUT_MD = Path("project/experiments/output/q3_policy_comparison_summary.md")


SITES = build_expansion_sites()


def masked_nearest_valid_action(state, env: MDPEnv, target: ActionVector) -> ActionVector:
    """Respect phase mask (frozen dims -> K) then snap to nearest valid action."""
    mask = mutable_mask(state.Theta)
    values = target.to_list()
    for i, m in enumerate(mask):
        if m == 0:
            values[i] = state.K[i]
    candidate = ActionVector.from_list(values)
    if candidate in env.valid_actions(state):
        return candidate
    valid = env.valid_actions(state)
    target_list = candidate.to_list()

    def dist(a: ActionVector) -> int:
        return sum(abs(x - y) for x, y in zip(a.to_list(), target_list))

    return min(valid, key=dist)


def static_policy_factory(env: MDPEnv, target: ActionVector):
    def policy(state):
        return masked_nearest_valid_action(state, env, target)

    return policy


def ppo_allowed_factory(env: MDPEnv):
    def allowed(state):
        allowed = action_space_per_dim(state.Theta, state.K)
        mask = mutable_mask(state.Theta)
        # keep the same debt safety brake as env.valid_actions, but do not hard-cap leverage-up.
        if mask[4] == 1 and state.F.leverage >= env.config.leverage_soft:
            allowed[4] = [a for a in allowed[4] if a <= 2]
        # equity cap (keep consistent with env.valid_actions)
        if mask[5] == 1 and env.config.max_equity_action is not None:
            allowed[5] = [a for a in allowed[5] if a <= env.config.max_equity_action]
        return allowed

    return allowed


def collect_metrics(state, cum_cf: float, action: ActionVector | None) -> Dict[str, float | int | str]:
    a = action.to_list() if action is not None else ["", "", "", "", "", ""]
    return {
        "year": int(state.year),
        "phase": str(state.Theta),
        "i_expansion": int(state.E.i_expansion),
        "mu_size": float(state.E.mu_size),
        "compete_local": int(state.E.compete_local),
        "n_star_fa": int(state.E.n_star_fa),
        "bidding_intensity": float(state.E.bidding_intensity),
        "travel_fatigue": float(getattr(state.E, "travel_fatigue", 0.0)),
        "leverage": float(state.F.leverage),
        "cash_flow": float(state.F.CF),
        "cash_flow_cum": float(cum_cf),
        "cash": float(state.F.Cash),
        "franchise_value": float(state.F.FV),
        "debt": float(state.F.D),
        "owner_share": float(state.F.owner_share),
        "terminal_value": float(terminal_value(state.F)),
        "win_pct": float(state.R.W[0]),
        "elo": float(state.R.ELO),
        "syn": float(state.R.Syn),
        "a_roster": a[0],
        "a_salary": a[1],
        "a_ticket": a[2],
        "a_marketing": a[3],
        "a_debt": a[4],
        "a_equity": a[5],
    }


def run_episode(env: MDPEnv, policy, seasons: int, seed: int) -> List[Dict[str, float | int | str]]:
    state = env.reset(seed=seed, use_data=True)
    rng = np.random.default_rng(seed)
    max_steps = seasons * 4
    records: List[Dict[str, float | int | str]] = []
    season_end_steps = {4, 8, 12}
    seen_season_end = set()
    cum_cf = 0.0

    for step in range(1, max_steps + 1):
        action = policy(state)
        if state.Theta == PHASE_OFFSEASON:
            rec = {"step": step, "marker": "offseason_decision"}
            rec.update(collect_metrics(state, cum_cf=cum_cf, action=action))
            records.append(rec)
        state, _, done, _ = env.step(state, action, rng)
        cum_cf += float(state.F.CF)
        if step in season_end_steps:
            rec = {"step": step, "marker": "season_end"}
            rec.update(collect_metrics(state, cum_cf=cum_cf, action=action))
            records.append(rec)
            seen_season_end.add(step)
        if done:
            # Fill missing season-end snapshots with the terminal (last) state for fair comparisons.
            for s in sorted(season_end_steps):
                if s >= step and s not in seen_season_end:
                    rec = {"step": s, "marker": "season_end"}
                    rec.update(collect_metrics(state, cum_cf=cum_cf, action=action))
                    records.append(rec)
            break
    return records


def summarize(rows: List[Dict]) -> List[Dict]:
    summary = []
    for site in sorted(set(r["site"] for r in rows)):
        for policy in sorted(set(r["policy"] for r in rows if r["site"] == site)):
            for seasons in [1, 2, 3]:
                subset = [
                    r
                    for r in rows
                    if r["marker"] == "season_end"
                    and r["site"] == site
                    and r["policy"] == policy
                    and r["seasons"] == seasons
                ]
                if not subset:
                    continue
                metrics = [
                    "win_pct",
                    "cash_flow",
                    "cash_flow_cum",
                    "terminal_value",
                    "leverage",
                    "franchise_value",
                    "debt",
                ]
                row = {"site": site, "policy": policy, "seasons": seasons, "n": len(subset)}
                for m in metrics:
                    vals = np.array([float(s[m]) for s in subset], dtype=float)
                    row[f"{m}_mean"] = float(np.mean(vals))
                    row[f"{m}_std"] = float(np.std(vals))
                summary.append(row)
    return summary


def summarize_expansion_actions(rows: List[Dict]) -> List[Dict]:
    """Summarize the *offseason decision* actions taken in the expansion year."""
    summary = []
    for site in sorted(set(r["site"] for r in rows)):
        for policy in sorted(set(r["policy"] for r in rows if r["site"] == site)):
            subset = [
                r
                for r in rows
                if r["marker"] == "offseason_decision"
                and r["site"] == site
                and r["policy"] == policy
                and int(r["year"]) == 2026
            ]
            if not subset:
                continue
            row = {"site": site, "policy": policy, "n": len(subset)}
            for k in ["a_roster", "a_salary", "a_ticket", "a_marketing", "a_debt", "a_equity"]:
                vals = np.array([float(s[k]) for s in subset], dtype=float)
                row[f"{k}_mean"] = float(np.mean(vals))
                row[f"{k}_std"] = float(np.std(vals))
            summary.append(row)
    return summary


def build_site_env(site: Dict, seed: int) -> MDPEnv:
    cfg = MDPConfig()
    cfg.expansion_market_delta = float(site["market_delta"])
    cfg.expansion_compete_delta = int(site["compete_delta"])
    cfg.expansion_media_bonus = float(site["media_bonus"])
    cfg.expansion_travel_fatigue = float(site["travel_fatigue"])
    # Only analyze one expansion event to keep interpretation clean.
    cfg.expansion_years = [2026]
    # Allow leverage decisions; keep soft-brake inside env.valid_actions().
    cfg.max_debt_action = None
    # Disallow equity issuance here to keep Q3 policy comparison focused on debt/salary/roster trade-offs.
    cfg.max_equity_action = 0

    cfg, _ = calibrate_config_from_data(cfg)
    # Re-apply scenario overrides in case calibration changes defaults in future.
    cfg.expansion_market_delta = float(site["market_delta"])
    cfg.expansion_compete_delta = int(site["compete_delta"])
    cfg.expansion_media_bonus = float(site["media_bonus"])
    cfg.expansion_travel_fatigue = float(site["travel_fatigue"])
    cfg.expansion_years = [2026]
    cfg.max_debt_action = None
    cfg.max_equity_action = 0

    env = MDPEnv(cfg, seed=seed, use_data=True)
    try:
        env.player_model = build_player_model(roster_size=cfg.roster_size)
    except Exception:
        env.player_model = None
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--train-episodes", type=int, default=10)
    parser.add_argument("--eval-seasons", type=int, default=3)
    args = parser.parse_args()

    OUTPUT_DETAIL.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for site in SITES:
        env = build_site_env(site, seed=42)

        # Policies (targets are interpreted through mask + nearest-valid snapping)
        baseline_pol = static_policy_factory(env, DEFAULT_ACTION)
        aggressive_target = ActionVector(6, 5, 6, 3, 4, 0)  # "high salary + chase stars + leverage"
        # Defensive policy should avoid liquidity crunches; use moderate deleveraging (not max).
        defensive_target = ActionVector(3, 1, 2, 3, 1, 0)   # "roster stability + marketing + deleverage"
        aggressive_pol = static_policy_factory(env, aggressive_target)
        defensive_pol = static_policy_factory(env, defensive_target)

        # Train PPO on this site scenario
        ppo_cfg = PPOConfig(steps_per_update=256, epochs=4, policy_type="mlp", hidden_size=64)
        agent = PPOAgent(env, cfg=ppo_cfg, allowed_fn=ppo_allowed_factory(env))
        agent.train(episodes=args.train_episodes)

        policies = {
            "baseline": baseline_pol,
            "aggressive": aggressive_pol,
            "defensive": defensive_pol,
            "ppo": agent.act,
        }

        for seed in range(args.seeds):
            for policy_name, policy in policies.items():
                # Rebuild env per rollout to keep the same config but independent RNG stream.
                env_run = build_site_env(site, seed=42)
                if policy_name == "ppo":
                    # Reuse trained agent weights with the new env instance.
                    agent.env = env_run
                    policy_fn = agent.act
                else:
                    policy_fn = policy
                records = run_episode(env_run, policy_fn, seasons=args.eval_seasons, seed=seed)
                for rec in records:
                    step = int(rec["step"])
                    seasons = step // 4
                    row = {"site": site["name"], "seed": seed, "policy": policy_name, "seasons": seasons}
                    row.update(rec)
                    rows.append(row)

    # Write detail CSV
    with OUTPUT_DETAIL.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Summary CSV
    summary = summarize(rows)
    with OUTPUT_SUMMARY.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    # Minimal markdown narrative for paper
    with OUTPUT_MD.open("w") as f:
        f.write("# Q3 Expansion Policy Comparison (Baseline vs PPO vs Heuristics)\n\n")
        f.write("Metrics are reported at the end of each season (1–3).\n\n")
        f.write("Policies:\n")
        f.write("- baseline: keep DEFAULT_ACTION (mask-aware + nearest-valid)\n")
        f.write("- aggressive: high salary + buyer_aggressive + leverage_high\n")
        f.write("- defensive: roster stability + marketing + deleverage_high\n")
        f.write("- ppo: learned adaptive policy under the same constraints\n\n")
        f.write(f"Seeds={args.seeds}, PPO train episodes={args.train_episodes}.\n\n")
        f.write("See `project/experiments/output/q3_policy_comparison_summary.csv` for season-end metrics.\n")
        action_summary = summarize_expansion_actions(rows)
        if action_summary:
            f.write("\n## Expansion-Year Offseason Action Summary (Year=2026)\n\n")
            for r in action_summary:
                f.write(
                    f"- {r['site']}/{r['policy']}: "
                    f"a_debt={r['a_debt_mean']:.2f}±{r['a_debt_std']:.2f}, "
                    f"a_salary={r['a_salary_mean']:.2f}±{r['a_salary_std']:.2f}, "
                    f"a_roster={r['a_roster_mean']:.2f}±{r['a_roster_std']:.2f}\n"
                )

    print(f"Saved detail: {OUTPUT_DETAIL}")
    print(f"Saved summary: {OUTPUT_SUMMARY}")
    print(f"Saved md: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
