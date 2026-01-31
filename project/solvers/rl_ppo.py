from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from project.mdp.action import ActionVector
from project.mdp.config import MDPConfig
from project.mdp.env import MDPEnv
from project.mdp.mask import action_space_per_dim


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return exp / np.sum(exp)


@dataclass
class PPOConfig:
    steps_per_update: int = 256
    epochs: int = 4
    gamma: float = 0.95
    clip: float = 0.2
    lr: float = 0.02
    vf_lr: float = 0.02


class LinearPolicy:
    def __init__(self, input_dim: int, action_dims: List[int], seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.weights = [self.rng.normal(0, 0.1, size=(n, input_dim)) for n in action_dims]
        self.biases = [np.zeros(n) for n in action_dims]
        self.v_w = self.rng.normal(0, 0.1, size=(input_dim,))
        self.v_b = 0.0

    def value(self, x: np.ndarray) -> float:
        return float(np.dot(self.v_w, x) + self.v_b)

    def logits(self, x: np.ndarray) -> List[np.ndarray]:
        return [W @ x + b for W, b in zip(self.weights, self.biases)]

    def action_distribution(self, x: np.ndarray, allowed: List[List[int]]) -> List[np.ndarray]:
        logits = self.logits(x)
        probs = []
        for logit, valid in zip(logits, allowed):
            mask = np.full_like(logit, -1e9, dtype=float)
            mask[valid] = 0.0
            p = _softmax(logit + mask)
            probs.append(p)
        return probs

    def sample_action(self, x: np.ndarray, allowed: List[List[int]]) -> Tuple[ActionVector, float, List[np.ndarray]]:
        probs = self.action_distribution(x, allowed)
        actions = []
        logprob = 0.0
        for p, valid in zip(probs, allowed):
            idx = int(self.rng.choice(len(p), p=p))
            actions.append(idx)
            logprob += float(np.log(max(p[idx], 1e-12)))
        return ActionVector(*actions), logprob, probs

    def logprob(self, x: np.ndarray, action: ActionVector, allowed: List[List[int]]) -> Tuple[float, List[np.ndarray]]:
        probs = self.action_distribution(x, allowed)
        action_list = action.to_list()
        logp = 0.0
        for p, a in zip(probs, action_list):
            logp += float(np.log(max(p[a], 1e-12)))
        return logp, probs


class PPOAgent:
    def __init__(self, env: MDPEnv, cfg: PPOConfig | None = None, seed: int = 0, allowed_fn=None):
        self.env = env
        self.cfg = cfg or PPOConfig()
        dummy_state = env.reset(seed=seed)
        self.input_dim = dummy_state.to_vector(env.config).shape[0]
        self.action_dims = [7, 6, 7, 4, 5, 5]
        self.policy = LinearPolicy(self.input_dim, self.action_dims, seed=seed)
        self.allowed_fn = allowed_fn

    def train(self, episodes: int = 50) -> None:
        steps = self.cfg.steps_per_update
        for _ in range(episodes):
            batch = self._collect_batch(steps)
            self._update(batch)

    def _collect_batch(self, steps: int) -> dict:
        states = []
        actions = []
        logprobs = []
        rewards = []
        values = []
        dones = []
        allowed_sets = []

        state = self.env.reset()
        for _ in range(steps):
            x = state.to_vector(self.env.config)
            allowed = self._allowed(state)
            action, logp, _ = self.policy.sample_action(x, allowed)
            value = self.policy.value(x)

            next_state, reward, done, _ = self.env.step(state, action)

            states.append(x)
            actions.append(action)
            logprobs.append(logp)
            rewards.append(reward)
            values.append(value)
            dones.append(done)
            allowed_sets.append(allowed)

            state = next_state if not done else self.env.reset()

        returns, advantages = self._compute_advantages(rewards, values, dones)
        return {
            "states": np.array(states),
            "actions": actions,
            "logprobs": np.array(logprobs),
            "returns": returns,
            "advantages": advantages,
            "allowed_sets": allowed_sets,
        }

    def _compute_advantages(self, rewards, values, dones):
        returns = np.zeros(len(rewards))
        adv = np.zeros(len(rewards))
        running_return = 0.0
        running_adv = 0.0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0.0
                running_adv = 0.0
            running_return = rewards[t] + self.cfg.gamma * running_return
            returns[t] = running_return
            advantage = returns[t] - values[t]
            running_adv = advantage
            adv[t] = running_adv
        # Normalize advantages
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-6)
        return returns, adv

    def _update(self, batch: dict) -> None:
        states = batch["states"]
        actions = batch["actions"]
        old_logprobs = batch["logprobs"]
        returns = batch["returns"]
        advantages = batch["advantages"]
        allowed_sets = batch["allowed_sets"]

        for _ in range(self.cfg.epochs):
            grad_W = [np.zeros_like(w) for w in self.policy.weights]
            grad_b = [np.zeros_like(b) for b in self.policy.biases]
            grad_v = np.zeros_like(self.policy.v_w)
            grad_vb = 0.0

            for i, x in enumerate(states):
                state_vec = x
                action = actions[i]
                allowed = allowed_sets[i]
                # For PPO update, use stored allowed based on current state proxy; this keeps updates simple
                logp, probs = self.policy.logprob(state_vec, action, allowed)
                ratio = np.exp(logp - old_logprobs[i])
                adv = advantages[i]

                if (adv >= 0 and ratio > 1 + self.cfg.clip) or (adv < 0 and ratio < 1 - self.cfg.clip):
                    weight = 0.0
                else:
                    weight = ratio

                # Policy gradients per dimension
                a_list = action.to_list()
                for d, (p, a) in enumerate(zip(probs, a_list)):
                    onehot = np.zeros_like(p)
                    onehot[a] = 1.0
                    grad = (onehot - p)[:, None] * state_vec[None, :]
                    grad_W[d] += weight * adv * grad
                    grad_b[d] += weight * adv * (onehot - p)

                # Value function gradients
                v = self.policy.value(state_vec)
                v_err = v - returns[i]
                grad_v += v_err * state_vec
                grad_vb += v_err

            # Apply gradient steps
            for d in range(len(self.policy.weights)):
                self.policy.weights[d] -= self.cfg.lr * grad_W[d] / max(1, len(states))
                self.policy.biases[d] -= self.cfg.lr * grad_b[d] / max(1, len(states))
            self.policy.v_w -= self.cfg.vf_lr * grad_v / max(1, len(states))
            self.policy.v_b -= self.cfg.vf_lr * grad_vb / max(1, len(states))

    def act(self, state) -> ActionVector:
        x = state.to_vector(self.env.config)
        allowed = self._allowed(state)
        action, _, _ = self.policy.sample_action(x, allowed)
        return action

    def _allowed(self, state):
        if self.allowed_fn is not None:
            return self.allowed_fn(state)
        return action_space_per_dim(state.Theta, state.K)
