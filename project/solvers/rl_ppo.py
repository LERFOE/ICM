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
    policy_type: str = "linear"
    hidden_size: int = 64


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


class MLPPolicy:
    def __init__(self, input_dim: int, action_dims: List[int], hidden_size: int = 64, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.hidden_size = hidden_size
        self.W1 = self.rng.normal(0, 0.1, size=(hidden_size, input_dim))
        self.b1 = np.zeros(hidden_size)
        self.W2 = [self.rng.normal(0, 0.1, size=(n, hidden_size)) for n in action_dims]
        self.b2 = [np.zeros(n) for n in action_dims]
        # value network
        self.v_W1 = self.rng.normal(0, 0.1, size=(hidden_size, input_dim))
        self.v_b1 = np.zeros(hidden_size)
        self.v_W2 = self.rng.normal(0, 0.1, size=(1, hidden_size))
        self.v_b2 = 0.0

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    def _hidden(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        z = self.W1 @ x + self.b1
        h = self._relu(z)
        mask = (z > 0.0).astype(float)
        return h, mask

    def _hidden_value(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        z = self.v_W1 @ x + self.v_b1
        h = self._relu(z)
        mask = (z > 0.0).astype(float)
        return h, mask

    def value(self, x: np.ndarray) -> float:
        h, _ = self._hidden_value(x)
        return float((self.v_W2 @ h)[0] + self.v_b2)

    def logits(self, x: np.ndarray) -> List[np.ndarray]:
        h, _ = self._hidden(x)
        return [W @ h + b for W, b in zip(self.W2, self.b2)]

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
        for p in probs:
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
        if self.cfg.policy_type == "mlp":
            self.policy = MLPPolicy(self.input_dim, self.action_dims, hidden_size=self.cfg.hidden_size, seed=seed)
        else:
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
            if isinstance(self.policy, MLPPolicy):
                grad_W1 = np.zeros_like(self.policy.W1)
                grad_b1 = np.zeros_like(self.policy.b1)
                grad_W2 = [np.zeros_like(w) for w in self.policy.W2]
                grad_b2 = [np.zeros_like(b) for b in self.policy.b2]
                grad_v_W1 = np.zeros_like(self.policy.v_W1)
                grad_v_b1 = np.zeros_like(self.policy.v_b1)
                grad_v_W2 = np.zeros_like(self.policy.v_W2)
                grad_v_b2 = 0.0
            else:
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

                a_list = action.to_list()
                if isinstance(self.policy, MLPPolicy):
                    h, mask = self.policy._hidden(state_vec)
                    delta_h = np.zeros_like(h)
                    for d, (p, a) in enumerate(zip(probs, a_list)):
                        onehot = np.zeros_like(p)
                        onehot[a] = 1.0
                        diff = onehot - p
                        grad_W2[d] += weight * adv * diff[:, None] * h[None, :]
                        grad_b2[d] += weight * adv * diff
                        delta_h += self.policy.W2[d].T @ diff
                    delta_h *= mask
                    grad_W1 += weight * adv * delta_h[:, None] * state_vec[None, :]
                    grad_b1 += weight * adv * delta_h

                    v = self.policy.value(state_vec)
                    v_err = v - returns[i]
                    hv, mask_v = self.policy._hidden_value(state_vec)
                    grad_v_W2 += v_err * hv[None, :]
                    grad_v_b2 += v_err
                    delta_v = (self.policy.v_W2.T * v_err).reshape(-1)
                    delta_v *= mask_v
                    grad_v_W1 += delta_v[:, None] * state_vec[None, :]
                    grad_v_b1 += delta_v
                else:
                    # Policy gradients per dimension (linear)
                    for d, (p, a) in enumerate(zip(probs, a_list)):
                        onehot = np.zeros_like(p)
                        onehot[a] = 1.0
                        grad = (onehot - p)[:, None] * state_vec[None, :]
                        grad_W[d] += weight * adv * grad
                        grad_b[d] += weight * adv * (onehot - p)

                    # Value function gradients (linear)
                    v = self.policy.value(state_vec)
                    v_err = v - returns[i]
                    grad_v += v_err * state_vec
                    grad_vb += v_err

            # Apply gradient steps
            if isinstance(self.policy, MLPPolicy):
                for d in range(len(self.policy.W2)):
                    self.policy.W2[d] -= self.cfg.lr * grad_W2[d] / max(1, len(states))
                    self.policy.b2[d] -= self.cfg.lr * grad_b2[d] / max(1, len(states))
                self.policy.W1 -= self.cfg.lr * grad_W1 / max(1, len(states))
                self.policy.b1 -= self.cfg.lr * grad_b1 / max(1, len(states))
                self.policy.v_W1 -= self.cfg.vf_lr * grad_v_W1 / max(1, len(states))
                self.policy.v_b1 -= self.cfg.vf_lr * grad_v_b1 / max(1, len(states))
                self.policy.v_W2 -= self.cfg.vf_lr * grad_v_W2 / max(1, len(states))
                self.policy.v_b2 -= self.cfg.vf_lr * grad_v_b2 / max(1, len(states))
            else:
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
