import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.mdp.action import ACTION_LABELS
from project.mdp.state import State
from project.solvers.mcts import MCTS
from project.solvers.heuristic_value import heuristic_value
from project.experiments.utils import build_env


OUTPUT_PATH = Path("project/experiments/output/q1_leverage_policy_map.csv")


def build_state(env, leverage: float, macro: int, cf: float) -> State:
    state = env.reset(use_data=True)
    state.F.FV = env.config.base_franchise_value
    state.F.D = leverage * state.F.FV
    state.F.leverage = leverage
    state.F.CF = cf
    state.E.macro = macro
    return state


def main():
    env = build_env(use_data=True, seed=42)
    mcts = MCTS(env, iterations=220, horizon=8, gamma=0.95, value_fn=heuristic_value, rollout_candidates=7)

    leverage_grid = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    cf_grid = [-5.0, 0.0, 5.0, 10.0]
    macro_grid = [0, 1, 2]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "leverage",
                "macro",
                "cash_flow",
                "a_debt",
                "a_debt_label",
                "action_vector",
            ]
        )
        for lam in leverage_grid:
            for macro in macro_grid:
                for cf in cf_grid:
                    state = build_state(env, lam, macro, cf)
                    action = mcts.search(state)
                    writer.writerow(
                        [
                            lam,
                            macro,
                            cf,
                            action.a_debt,
                            ACTION_LABELS["debt"][action.a_debt],
                            action.to_list(),
                        ]
                    )

    print(f"Saved leverage policy map to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
