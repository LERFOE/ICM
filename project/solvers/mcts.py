from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np

from project.mdp.action import ActionVector
from project.mdp.env import MDPEnv
from project.mdp.mask import enumerate_valid_actions
from project.mdp.state import State


@dataclass
class MCTSNode:
    state: State
    parent: Optional["MCTSNode"] = None
    action: Optional[ActionVector] = None
    children: Dict[ActionVector, "MCTSNode"] = field(default_factory=dict)
    N: int = 0
    W: float = 0.0
    untried: Optional[list] = None

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0


class MCTS:
    def __init__(
        self,
        env: MDPEnv,
        iterations: int = 200,
        horizon: int = 6,
        gamma: float = 0.95,
        c_puct: float = 1.2,
        seed: int = 42,
    ):
        self.env = env
        self.iterations = iterations
        self.horizon = horizon
        self.gamma = gamma
        self.c_puct = c_puct
        self.rng = np.random.default_rng(seed)

    def search(self, root_state: State) -> ActionVector:
        root = MCTSNode(state=root_state)
        for _ in range(self.iterations):
            node = root
            depth = 0
            cumulative_reward = 0.0
            discount = 1.0

            # Selection
            while node.children and depth < self.horizon:
                node = self._select_child(node)
                depth += 1

            # Expansion
            if depth < self.horizon:
                node, reward, done = self._expand(node)
                cumulative_reward += discount * reward
                discount *= self.gamma
                depth += 1

                # Simulation
                if not done:
                    rollout_reward = self._rollout(node.state, self.horizon - depth)
                    cumulative_reward += discount * rollout_reward

            # Backprop
            self._backprop(node, cumulative_reward)

        best_action, _ = self._best_child(root)
        return best_action

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        _, child = self._best_child(node, use_ucb=True)
        return child

    def _best_child(self, node: MCTSNode, use_ucb: bool = False) -> Tuple[ActionVector, MCTSNode]:
        best_score = -1e9
        best_action = None
        best_child = None
        for action, child in node.children.items():
            if use_ucb:
                ucb = child.Q + self.c_puct * np.sqrt(np.log(node.N + 1) / (child.N + 1))
                score = ucb
            else:
                score = child.Q
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        if best_child is None:
            # fallback: pick a random valid action
            action = self._random_action(node.state)
            next_state, reward, done, _ = self.env.step(node.state, action, self.rng)
            child = MCTSNode(state=next_state, parent=node, action=action)
            node.children[action] = child
            return action, child
        return best_action, best_child

    def _expand(self, node: MCTSNode) -> Tuple[MCTSNode, float, bool]:
        if node.untried is None:
            node.untried = list(enumerate_valid_actions(node.state.Theta, node.state.K))
            self.rng.shuffle(node.untried)

        # Progressive widening
        max_children = min(len(node.untried), int(2 + np.sqrt(node.N + 1)))
        if len(node.children) >= max_children and node.children:
            return node, 0.0, False

        if not node.untried:
            return node, 0.0, False

        action = node.untried.pop()
        next_state, reward, done, _ = self.env.step(node.state, action, self.rng)
        child = MCTSNode(state=next_state, parent=node, action=action)
        node.children[action] = child
        return child, reward, done

    def _rollout(self, state: State, depth: int) -> float:
        total = 0.0
        discount = 1.0
        current = state
        for _ in range(depth):
            action = self._random_action(current)
            next_state, reward, done, _ = self.env.step(current, action, self.rng)
            total += discount * reward
            discount *= self.gamma
            current = next_state
            if done:
                break
        return total

    def _random_action(self, state: State) -> ActionVector:
        actions = list(enumerate_valid_actions(state.Theta, state.K))
        return actions[self.rng.integers(0, len(actions))]

    def _backprop(self, node: MCTSNode, reward: float) -> None:
        current = node
        while current is not None:
            current.N += 1
            current.W += reward
            current = current.parent
