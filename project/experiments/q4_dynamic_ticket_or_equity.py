import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.mdp.action import DEFAULT_ACTION
from project.solvers.eval import rollout_policy
from project.solvers.rl_ppo import PPOAgent
from project.experiments.utils import build_env
from project.mdp.env import MDPEnv


OUTPUT_MD = Path("project/experiments/output/q4_dynamic_policy_summary.md")


def static_policy(_state):
    return DEFAULT_ACTION


def collect_action_freq(env: MDPEnv, policy, episodes: int = 10, max_steps: int = 20):
    counter = Counter()
    for _ in range(episodes):
        state = env.reset()
        for _ in range(max_steps):
            action = policy(state)
            counter[tuple(action.to_list())] += 1
            state, _, done, _ = env.step(state, action)
            if done:
                break
    return counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ticket", "equity"], default="ticket")
    args = parser.parse_args()

    env = build_env(use_data=True, seed=42)

    baseline = rollout_policy(env, static_policy, episodes=12, max_steps=20)

    agent = PPOAgent(env)
    agent.train(episodes=30)
    learned = rollout_policy(env, agent.act, episodes=12, max_steps=20)

    freq = collect_action_freq(env, agent.act)

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_MD.open("w") as f:
        f.write("# Q4 Dynamic Business Decision\n\n")
        f.write(f"Mode: {args.mode}\n\n")
        f.write("Baseline vs Learned Policy:\n")
        f.write(
            f"- Baseline avg_terminal={baseline['avg_terminal']:.2f}, avg_cf={baseline['avg_cf']:.2f}\n"
        )
        f.write(
            f"- Learned  avg_terminal={learned['avg_terminal']:.2f}, avg_cf={learned['avg_cf']:.2f}\n"
        )
        f.write("\nMost common learned action vectors:\n")
        for action, count in freq.most_common(5):
            f.write(f"- {action}: {count} steps\n")
        f.write("\nInterpretation:\n")
        f.write("- Ticket decisions trade off short-term gate revenue vs long-term brand growth.\n")
        f.write("- Equity decisions trade off cash relief vs permanent dilution.\n")

    print(f"Saved Q4 summary to {OUTPUT_MD}")


if __name__ == "__main__":
    main()
