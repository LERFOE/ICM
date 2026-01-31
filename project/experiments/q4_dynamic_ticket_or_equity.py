import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.mdp.action import ActionVector, DEFAULT_ACTION
from project.solvers.eval import rollout_policy
from project.solvers.rl_ppo import PPOAgent
from project.experiments.utils import build_env
from project.mdp.env import MDPEnv
from project.mdp.mask import action_space_per_dim, mutable_mask


OUTPUT_MD = Path("project/experiments/output/q4_dynamic_policy_summary_{mode}.md")

TRAIN_EPISODES = 12
EVAL_EPISODES = 8
MAX_STEPS = 20


def _nearest_valid(env: MDPEnv, state) -> ActionVector:
    target = ActionVector.from_list(list(state.K))
    if target in env.valid_actions(state):
        return target
    valid = env.valid_actions(state)
    target_list = target.to_list()

    def dist(a: ActionVector) -> int:
        return sum(abs(x - y) for x, y in zip(a.to_list(), target_list))

    return min(valid, key=dist)


def mode_allowed_factory(env: MDPEnv, mode: str):
    target_idx = 2 if mode == "ticket" else 5

    def allowed(state):
        base_action = _nearest_valid(env, state)
        allowed = action_space_per_dim(state.Theta, state.K)
        # freeze non-target dims to current K
        for i in range(len(allowed)):
            if i != target_idx:
                allowed[i] = [base_action.to_list()[i]]
        # apply caps on equity if equity is target
        if target_idx == 5:
            mask = mutable_mask(state.Theta)
            if mask[5] == 1 and env.config.max_equity_action is not None:
                allowed[5] = [a for a in allowed[5] if a <= env.config.max_equity_action]
        return allowed

    return allowed


def hold_policy_factory(env: MDPEnv):
    def policy(state):
        return _nearest_valid(env, state)

    return policy


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

    baseline = rollout_policy(env, hold_policy_factory(env), episodes=EVAL_EPISODES, max_steps=MAX_STEPS)

    allowed_fn = mode_allowed_factory(env, args.mode)
    from project.solvers.rl_ppo import PPOConfig

    ppo_cfg = PPOConfig(steps_per_update=128, epochs=2)
    agent = PPOAgent(env, cfg=ppo_cfg, allowed_fn=allowed_fn)
    agent.train(episodes=TRAIN_EPISODES)
    learned = rollout_policy(env, agent.act, episodes=EVAL_EPISODES, max_steps=MAX_STEPS)

    freq = collect_action_freq(env, agent.act)

    out_path = Path(str(OUTPUT_MD).format(mode=args.mode))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("# Q4 Dynamic Business Decision\n\n")
        f.write(f"Mode: {args.mode}\n\n")
        f.write("Baseline vs Learned Policy (only target dimension changes; others fixed to K):\n")
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

    print(f"Saved Q4 summary to {out_path}")


if __name__ == "__main__":
    main()
