import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.data.geo import build_expansion_sites
from project.data.integration import calibrate_config_from_data
from project.data.player_kmeans import build_player_model
from project.mdp.action import ActionVector, DEFAULT_ACTION
from project.mdp.config import MDPConfig
from project.mdp.env import MDPEnv
from project.mdp.reward import terminal_value
from project.mdp.phase import PHASE_OFFSEASON
from project.solvers.rl_ppo import PPOAgent, PPOConfig
from project.mdp.mask import action_space_per_dim, mutable_mask


OUTPUT = Path("project/experiments/output/q3_policy_diagnostics.csv")


def masked_nearest_valid_action(state, env: MDPEnv, target: ActionVector) -> ActionVector:
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
        if mask[4] == 1 and state.F.leverage >= env.config.leverage_soft:
            allowed[4] = [a for a in allowed[4] if a <= 2]
        if mask[5] == 1 and env.config.max_equity_action is not None:
            allowed[5] = [a for a in allowed[5] if a <= env.config.max_equity_action]
        return allowed

    return allowed


def build_site_env(site: Dict, seed: int, debt_penalty: float, fatigue_on: bool) -> MDPEnv:
    cfg = MDPConfig()
    cfg.expansion_market_delta = float(site["market_delta"])
    cfg.expansion_compete_delta = int(site["compete_delta"])
    cfg.expansion_media_bonus = float(site["media_bonus"])
    cfg.expansion_travel_fatigue = float(site["travel_fatigue"]) if fatigue_on else 0.0
    cfg.expansion_years = [2026]
    cfg.max_debt_action = None
    cfg.max_equity_action = 0
    cfg.debt_issue_penalty = debt_penalty

    cfg, _ = calibrate_config_from_data(cfg)
    cfg.expansion_market_delta = float(site["market_delta"])
    cfg.expansion_compete_delta = int(site["compete_delta"])
    cfg.expansion_media_bonus = float(site["media_bonus"])
    cfg.expansion_travel_fatigue = float(site["travel_fatigue"]) if fatigue_on else 0.0
    cfg.expansion_years = [2026]
    cfg.max_debt_action = None
    cfg.max_equity_action = 0
    cfg.debt_issue_penalty = debt_penalty

    env = MDPEnv(cfg, seed=seed, use_data=True)
    try:
        env.player_model = build_player_model(roster_size=cfg.roster_size)
    except Exception:
        env.player_model = None
    return env


def eval_policy(env: MDPEnv, policy, episodes: int, seasons: int, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    max_steps = seasons * 4
    terminals = []
    cum_cfs = []
    win_pcts = []
    debts = []
    actions = []

    for _ in range(episodes):
        state = env.reset(seed=rng.integers(0, 1_000_000))
        cum_cf = 0.0
        for _ in range(max_steps):
            action = policy(state)
            if state.year == 2026 and state.Theta == PHASE_OFFSEASON:
                actions.append(action.to_list())
            state, _, done, _ = env.step(state, action, rng)
            cum_cf += state.F.CF
            if done:
                break
        terminals.append(terminal_value(state.F))
        cum_cfs.append(cum_cf)
        win_pcts.append(state.R.W[0])
        debts.append(state.F.D)

    actions = np.array(actions) if actions else np.zeros((1, 6))
    return {
        "terminal_mean": float(np.mean(terminals)),
        "cf_cum_mean": float(np.mean(cum_cfs)),
        "win_pct_mean": float(np.mean(win_pcts)),
        "debt_mean": float(np.mean(debts)),
        "a_roster_mean": float(np.mean(actions[:, 0])),
        "a_salary_mean": float(np.mean(actions[:, 1])),
        "a_ticket_mean": float(np.mean(actions[:, 2])),
        "a_marketing_mean": float(np.mean(actions[:, 3])),
        "a_debt_mean": float(np.mean(actions[:, 4])),
        "a_equity_mean": float(np.mean(actions[:, 5])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", type=str, default=None, help="Comma-separated site names.")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--train-episodes", type=int, default=6)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--eval-seasons", type=int, default=3)
    args = parser.parse_args()

    sites = build_expansion_sites()
    if args.sites:
        allow = {s.strip().lower() for s in args.sites.split(",") if s.strip()}
        sites = [s for s in sites if s["name"].lower() in allow]
        if not sites:
            raise SystemExit("No matching sites.")

    variants = [
        {"policy_type": "linear", "debt_penalty": 0.0, "fatigue_on": True},
        {"policy_type": "linear", "debt_penalty": 4.0, "fatigue_on": True},
        {"policy_type": "mlp", "debt_penalty": 0.0, "fatigue_on": True},
        {"policy_type": "mlp", "debt_penalty": 4.0, "fatigue_on": True},
    ]

    rows: List[Dict] = []
    for site in sites:
        for v in variants:
            env = build_site_env(site, seed=42, debt_penalty=v["debt_penalty"], fatigue_on=v["fatigue_on"])
            baseline_pol = static_policy_factory(env, DEFAULT_ACTION)

            ppo_cfg = PPOConfig(steps_per_update=256, epochs=4, policy_type=v["policy_type"], hidden_size=64)
            agent = PPOAgent(env, cfg=ppo_cfg, allowed_fn=ppo_allowed_factory(env))
            agent.train(episodes=args.train_episodes)

            for seed in range(args.seeds):
                ppo_metrics = eval_policy(env, agent.act, args.eval_episodes, args.eval_seasons, seed=seed)
                base_metrics = eval_policy(env, baseline_pol, args.eval_episodes, args.eval_seasons, seed=seed)

                for name, metrics in [("ppo", ppo_metrics), ("baseline", base_metrics)]:
                    row = {
                        "site": site["name"],
                        "policy": name,
                        "policy_type": v["policy_type"],
                        "debt_penalty": v["debt_penalty"],
                        "fatigue_on": v["fatigue_on"],
                        "seed": seed,
                        **metrics,
                    }
                    rows.append(row)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved diagnostics to {OUTPUT}")


if __name__ == "__main__":
    main()
