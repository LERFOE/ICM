import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.data.loaders import load_attendance
from project.data.player_kmeans import build_player_model
from project.data.geo import TEAM_LOCATIONS, build_expansion_sites, haversine_km
from project.experiments.utils import build_env
from project.mdp.action import DEFAULT_ACTION
from project.mdp.config import MDPConfig
from project.solvers.eval import rollout_policy
from project.solvers.rl_ppo import PPOAgent, PPOConfig
from project.mdp.mask import action_space_per_dim, mutable_mask


OUTPUT_CSV = Path("project/experiments/output/q3_expansion_sensitivity.csv")
OUTPUT_MD = Path("project/experiments/output/q3_expansion_summary.md")


TRAIN_EPISODES = 8
EVAL_EPISODES = 6
MAX_STEPS = 20

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



def static_policy_factory(env):
    target = DEFAULT_ACTION

    def policy(state):
        if target in env.valid_actions(state):
            return target
        valid = env.valid_actions(state)
        target_list = target.to_list()

        def dist(a):
            return sum(abs(x - y) for x, y in zip(a.to_list(), target_list))

        return min(valid, key=dist)

    return policy


def ppo_allowed_factory(env):
    def allowed(state):
        allowed = action_space_per_dim(state.Theta, state.K)
        mask = mutable_mask(state.Theta)
        # debt caps
        if mask[4] == 1:
            if env.config.max_debt_action is not None:
                allowed[4] = [a for a in allowed[4] if a <= env.config.max_debt_action]
            if state.F.leverage >= env.config.leverage_soft:
                allowed[4] = [a for a in allowed[4] if a <= 2]
        # equity caps
        if mask[5] == 1 and env.config.max_equity_action is not None:
            allowed[5] = [a for a in allowed[5] if a <= env.config.max_equity_action]
        return allowed

    return allowed


def _market_index() -> Dict[str, float]:
    att = load_attendance()
    latest = att.sort_values("Season").groupby("Team").tail(1)
    league_avg = float(latest["Avg_Attendance"].mean())
    market_index = {}
    for code, name in TEAM_CODE_TO_NAME.items():
        row = latest[latest["Team"] == name]
        if row.empty:
            market_index[code] = 1.0
        else:
            market_index[code] = float(row.iloc[0]["Avg_Attendance"]) / league_avg
    return market_index


def _strength_index(model) -> Dict[str, float]:
    strengths = {}
    for team, stats in model.team_strengths.items():
        elo = float(stats.get("elo", 1500.0))
        strengths[team] = (elo - 1500.0) / 100.0
    return strengths


def _league_impact(site, market_idx, strength_idx, cfg: MDPConfig) -> Dict[str, float]:
    impact: Dict[str, float] = {}
    scarcity = abs(cfg.expansion_star_fa_delta)
    for team in market_idx:
        m = market_idx[team]
        s = strength_idx.get(team, 0.0)
        loc_team = TEAM_LOCATIONS.get(team)
        if loc_team is None:
            dist_km = 1500.0
        else:
            dist_km = haversine_km(loc_team.lat, loc_team.lon, site["lat"], site["lon"])
        overlap = max(0.0, 1.0 - dist_km / 600.0)
        travel_term = -0.02 * (dist_km / 3000.0) + 0.01 * float(site["hub_score"])

        impact_score = (
            0.6 * site["media_bonus"] * m
            + (site["market_delta"] - 0.03 * overlap) * m
            - 0.05 * site["compete_delta"] * (1.0 - m)
            + travel_term
            - 0.02 * scarcity * (1.0 - s)
        )
        impact[team] = impact_score
    return impact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-episodes", type=int, default=TRAIN_EPISODES)
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = parser.parse_args()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    model = build_player_model()
    market_idx = _market_index()
    strength_idx = _strength_index(model)

    rows = []
    sites = build_expansion_sites()
    for site in sites:
        cfg = MDPConfig()
        cfg.expansion_market_delta = site["market_delta"]
        cfg.expansion_compete_delta = site["compete_delta"]
        cfg.expansion_media_bonus = site["media_bonus"]
        cfg.expansion_travel_fatigue = site["travel_fatigue"]
        # For Q3 we focus on one upcoming expansion season for interpretability.
        cfg.expansion_years = [2026]
        # Allow debt decisions in planning; keep equity disabled so results focus on leverage vs roster/salary.
        cfg.max_debt_action = None
        cfg.max_equity_action = 0

        # build env with calibrated config but override expansion deltas
        try:
            from project.data.integration import calibrate_config_from_data

            cfg, _ = calibrate_config_from_data(cfg)
        except Exception:
            pass
        # Re-apply scenario overrides after calibration (future-proof).
        cfg.expansion_market_delta = site["market_delta"]
        cfg.expansion_compete_delta = site["compete_delta"]
        cfg.expansion_media_bonus = site["media_bonus"]
        cfg.expansion_travel_fatigue = site["travel_fatigue"]
        cfg.expansion_years = [2026]
        cfg.max_debt_action = None
        cfg.max_equity_action = 0
        env = build_env(use_data=True, seed=42)
        env.config = cfg
        env.use_data = True

        baseline = rollout_policy(env, static_policy_factory(env), episodes=args.eval_episodes, max_steps=args.max_steps)
        ppo_cfg = PPOConfig(steps_per_update=256, epochs=4, policy_type="mlp", hidden_size=64)
        agent = PPOAgent(env, cfg=ppo_cfg, allowed_fn=ppo_allowed_factory(env))
        agent.train(episodes=args.train_episodes)
        learned = rollout_policy(env, agent.act, episodes=args.eval_episodes, max_steps=args.max_steps)

        league_impact = _league_impact(site, market_idx, strength_idx, cfg)
        winners = sorted(league_impact.items(), key=lambda x: x[1], reverse=True)[:3]
        losers = sorted(league_impact.items(), key=lambda x: x[1])[:3]

        rows.append(
            {
                "site": site["name"],
                "baseline_terminal": baseline["avg_terminal"],
                "learned_terminal": learned["avg_terminal"],
                "delta_terminal": learned["avg_terminal"] - baseline["avg_terminal"],
                "baseline_cf": baseline["avg_cf"],
                "learned_cf": learned["avg_cf"],
                "delta_cf": learned["avg_cf"] - baseline["avg_cf"],
                "dist_to_ind_km": site["dist_to_ind_km"],
                "travel_fatigue": site["travel_fatigue"],
                "market_score": site["market_score"],
                "hub_score": site["hub_score"],
                "nearby_teams": site["nearby_teams"],
                "note": site["note"],
                "winners": ",".join([w[0] for w in winners]),
                "losers": ",".join([l[0] for l in losers]),
            }
        )

    rows.sort(key=lambda r: r["delta_terminal"], reverse=True)

    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "site",
                "baseline_terminal",
                "learned_terminal",
                "delta_terminal",
                "baseline_cf",
                "learned_cf",
                "delta_cf",
                "winners",
                "losers",
                "dist_to_ind_km",
                "travel_fatigue",
                "market_score",
                "hub_score",
                "nearby_teams",
                "note",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r["site"],
                    round(r["baseline_terminal"], 3),
                    round(r["learned_terminal"], 3),
                    round(r["delta_terminal"], 3),
                    round(r["baseline_cf"], 3),
                    round(r["learned_cf"], 3),
                    round(r["delta_cf"], 3),
                    r["winners"],
                    r["losers"],
                    r["dist_to_ind_km"],
                    r["travel_fatigue"],
                    r["market_score"],
                    r["hub_score"],
                    r["nearby_teams"],
                    r["note"],
                ]
            )

    with OUTPUT_MD.open("w") as f:
        f.write("# Q3 Expansion Site Sensitivity\n\n")
        f.write("Sites ranked by change in Indiana owner terminal value (learned policy vs baseline):\n\n")
        for r in rows:
            f.write(
                f"- {r['site']}: ΔTerminal={r['delta_terminal']:.2f}, ΔCF={r['delta_cf']:.2f}, "
                f"Winners={r['winners']}, Losers={r['losers']}\n"
            )
        f.write("\nInterpretation:\n")
        f.write("- Winners/Losers are based on market size (attendance proxy), team strength, and local competition.\n")
        f.write("- Positive ΔTerminal indicates expansion site is net beneficial to the Indiana owner.\n")
        f.write("- Negative ΔTerminal suggests market dilution or higher competition hurts value.\n")

    print(f"Saved expansion sensitivity to {OUTPUT_CSV} and {OUTPUT_MD}")


if __name__ == "__main__":
    main()
