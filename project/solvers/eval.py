from __future__ import annotations
from typing import Callable, Dict
import numpy as np

from project.mdp.action import ActionVector
from project.mdp.env import MDPEnv
from project.mdp.mask import enumerate_valid_actions
from project.mdp.reward import terminal_value
from project.mdp.state import State


PolicyFn = Callable[[State], ActionVector]


def random_policy(state: State, rng: np.random.Generator | None = None, env: MDPEnv | None = None) -> ActionVector:
    rng = rng or np.random.default_rng()
    if env is not None and hasattr(env, "valid_actions"):
        actions = list(env.valid_actions(state))
    else:
        actions = list(enumerate_valid_actions(state.Theta, state.K))
    return actions[rng.integers(0, len(actions))]


def rollout_policy(
    env: MDPEnv,
    policy: PolicyFn,
    episodes: int = 20,
    max_steps: int = 24,
    seed: int = 0,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    total_reward = 0.0
    total_cf = 0.0
    total_fv = 0.0
    total_terminal = 0.0

    for _ in range(episodes):
        state = env.reset(seed=rng.integers(0, 1_000_000))
        ep_reward = 0.0
        for _ in range(max_steps):
            action = policy(state)
            state, reward, done, _ = env.step(state, action, rng)
            ep_reward += reward
            total_cf += state.F.CF
            total_fv += state.F.FV
            if done:
                total_terminal += terminal_value(state.F)
                break
        total_reward += ep_reward

    steps = max_steps * episodes
    return {
        "avg_reward": total_reward / episodes,
        "avg_cf": total_cf / steps,
        "avg_fv": total_fv / steps,
        "avg_terminal": total_terminal / max(1, episodes),
    }


def evaluate_action(
    env: MDPEnv,
    state: State,
    action: ActionVector,
    rollouts: int = 30,
    horizon: int = 8,
    seed: int = 0,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    rewards = []
    terminals = []
    for _ in range(rollouts):
        s = state.copy()
        s_next, reward, done, _ = env.step(s, action, rng)
        total = reward
        steps = 1
        while not done and steps < horizon:
            a = random_policy(s_next, rng, env)
            s_next, r, done, _ = env.step(s_next, a, rng)
            total += r
            steps += 1
        rewards.append(total)
        terminals.append(terminal_value(s_next.F))
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_terminal": float(np.mean(terminals)),
    }
