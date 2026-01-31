import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.mdp.action import ACTION_LABELS, ActionVector
from project.mdp.phase import PHASE_OFFSEASON
from project.solvers.eval import evaluate_action
from project.experiments.utils import build_env


OUTPUT_CSV = Path("project/experiments/output/q2_recruitment_strategy.csv")
OUTPUT_MD = Path("project/experiments/output/q2_recruitment_summary.md")


def main():
    env = build_env(use_data=True, seed=42)
    state = env.reset(use_data=True)
    state.Theta = PHASE_OFFSEASON

    base = state.K
    rows = []

    for a_roster in range(7):
        for a_salary in range(6):
            action = ActionVector(
                a_roster,
                a_salary,
                base[2],
                base[3],
                base[4],
                base[5],
            )
            metrics = evaluate_action(env, state, action, rollouts=40, horizon=8)
            rows.append(
                {
                    "a_roster": a_roster,
                    "a_salary": a_salary,
                    "mean_reward": metrics["mean_reward"],
                    "std_reward": metrics["std_reward"],
                    "mean_terminal": metrics["mean_terminal"],
                }
            )

    rows.sort(key=lambda r: r["mean_terminal"], reverse=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "a_roster",
                "roster_label",
                "a_salary",
                "salary_label",
                "mean_reward",
                "std_reward",
                "mean_terminal",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r["a_roster"],
                    ACTION_LABELS["roster"][r["a_roster"]],
                    r["a_salary"],
                    ACTION_LABELS["salary"][r["a_salary"]],
                    round(r["mean_reward"], 3),
                    round(r["std_reward"], 3),
                    round(r["mean_terminal"], 3),
                ]
            )

    top = rows[:5]
    with OUTPUT_MD.open("w") as f:
        f.write("# Q2 Recruitment Strategy (Offseason)\n\n")
        f.write("Top strategies ranked by owner terminal value:\n\n")
        for i, r in enumerate(top, 1):
            f.write(
                f"{i}. roster={ACTION_LABELS['roster'][r['a_roster']]} | "
                f"salary={ACTION_LABELS['salary'][r['a_salary']]} | "
                f"mean_terminal={r['mean_terminal']:.2f} | "
                f"mean_reward={r['mean_reward']:.2f}\n"
            )
        f.write("\nNotes:\n")
        f.write("- Buyer actions improve competitive state but raise payroll risk.\n")
        f.write("- Seller actions improve flexibility but reduce short-term win%.\n")

    print(f"Saved recruitment results to {OUTPUT_CSV} and {OUTPUT_MD}")


if __name__ == "__main__":
    main()
