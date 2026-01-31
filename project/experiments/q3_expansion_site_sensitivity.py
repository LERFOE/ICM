import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.mdp.action import DEFAULT_ACTION
from project.mdp.config import MDPConfig
from project.solvers.eval import rollout_policy
from project.solvers.rl_ppo import PPOAgent
from project.experiments.utils import build_env


OUTPUT_CSV = Path("project/experiments/output/q3_expansion_sensitivity.csv")
OUTPUT_MD = Path("project/experiments/output/q3_expansion_summary.md")


SITES = [
    {"name": "Toronto", "market_delta": 0.02, "compete_delta": 0, "media_bonus": 0.05},
    {"name": "Denver", "market_delta": -0.02, "compete_delta": 1, "media_bonus": 0.02},
    {"name": "Nashville", "market_delta": -0.03, "compete_delta": 1, "media_bonus": 0.01},
    {"name": "SanDiego", "market_delta": -0.01, "compete_delta": 0, "media_bonus": 0.03},
]


def static_policy(_state):
    return DEFAULT_ACTION


def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for site in SITES:
        cfg = MDPConfig()
        cfg.expansion_market_delta = site["market_delta"]
        cfg.expansion_compete_delta = site["compete_delta"]
        cfg.expansion_media_bonus = site["media_bonus"]

        # build env with calibrated config but override expansion deltas
        try:
            from project.data.integration import calibrate_config_from_data
            cfg, _ = calibrate_config_from_data(cfg)
        except Exception:
            pass
        env = build_env(use_data=True, seed=42)
        env.config = cfg
        env.use_data = True

        baseline = rollout_policy(env, static_policy, episodes=12, max_steps=20)

        agent = PPOAgent(env)
        agent.train(episodes=30)
        learned = rollout_policy(env, agent.act, episodes=12, max_steps=20)

        rows.append(
            {
                "site": site["name"],
                "baseline_terminal": baseline["avg_terminal"],
                "learned_terminal": learned["avg_terminal"],
                "delta_terminal": learned["avg_terminal"] - baseline["avg_terminal"],
                "baseline_cf": baseline["avg_cf"],
                "learned_cf": learned["avg_cf"],
                "delta_cf": learned["avg_cf"] - baseline["avg_cf"],
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
                ]
            )

    with OUTPUT_MD.open("w") as f:
        f.write("# Q3 Expansion Site Sensitivity\n\n")
        f.write("Sites ranked by change in owner terminal value (learned policy vs baseline):\n\n")
        for r in rows:
            f.write(
                f"- {r['site']}: ΔTerminal={r['delta_terminal']:.2f}, ΔCF={r['delta_cf']:.2f}\n"
            )
        f.write("\nInterpretation:\n")
        f.write("- Positive ΔTerminal indicates expansion site is net beneficial to the incumbent owner.\n")
        f.write("- Negative ΔTerminal suggests market dilution or higher competition hurts value.\n")

    print(f"Saved expansion sensitivity to {OUTPUT_CSV} and {OUTPUT_MD}")


if __name__ == "__main__":
    main()
