import numpy as np
import copy
import random
from src.mdp_definitions import *
from src.mdp_simulation import MDPSimulationEnv

class MCTSNode:
    def __init__(self, state_snapshot=None, parent=None, action_from_parent=None):
        self.state_snapshot = state_snapshot # Deepcopy of Env object or State Dict
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.children = []
        self.visits = 0
        self.value_sum = 0.0
        
    def is_fully_expanded(self):
        # Allow sampling of 5-10 random actions, not all possible (space too big)
        return len(self.children) >= 5 

    def best_child(self, exploration_weight=1.41):
        best_score = -float('inf')
        best_c = None
        for c in self.children:
            exploit = c.value_sum / (c.visits + 1e-6)
            explore = exploration_weight * np.sqrt(np.log(self.visits + 1) / (c.visits + 1e-6))
            score = exploit + explore
            if score > best_score:
                best_score = score
                best_c = c
        return best_c

class MDPSolver:
    def __init__(self, horizon=5):
        self.horizon = horizon
        self.env_model = MDPSimulationEnv() # Internal model for planning

    def plan(self, current_real_env):
        """
        Input: The actual environment maximizing over.
        Returns: Best ActionVector
        """
        root = MCTSNode(state_snapshot=copy.deepcopy(current_real_env))
        
        # Run Simulations
        for _ in range(50): # 50 Iterations
            node = root
            
            # Selection
            while node.is_fully_expanded() and node.children:
                node = node.best_child()
                
            # Expansion
            if not node.is_fully_expanded():
                action = self._sample_random_valid_action(node.state_snapshot)
                new_state_snapshot = copy.deepcopy(node.state_snapshot)
                _, reward, _ = new_state_snapshot.step(action)
                
                # Create Child
                child = MCTSNode(state_snapshot=new_state_snapshot, parent=node, action_from_parent=action)
                node.children.append(child)
                node = child
            
            # Simulation (Rollout)
            rollout_val = self._rollout(node.state_snapshot, depth=3)
            
            # Backprop
            curr = node
            while curr:
                curr.visits += 1
                curr.value_sum += rollout_val
                curr = curr.parent
                
        # Best Action
        best_child = root.best_child(exploration_weight=0.0)
        return best_child.action_from_parent if best_child else self._sample_random_valid_action(current_real_env)

    def _sample_random_valid_action(self, env):
        # Masking based on Phase
        phase = env.phase
        
        # Default restricted
        # a_roster, a_salary, a_ticket, a_marketing, a_debt, a_equity
        
        a_roster = ACT_ROSTER_HOLD
        a_salary = ACT_SALARY_FLOOR
        a_ticket = ACT_TICKET_NORMAL
        a_marketing = ACT_MARKETING_NORMAL
        a_debt = ACT_DEBT_HOLD
        a_equity = ACT_EQUITY_0

        if phase == PHASE_OFFSEASON:
            a_roster = random.choice([0, 1, 2, 3, 4])
            a_salary = random.choice([0, 1, 2, 3])
            a_debt = random.choice([0, 1, 2])
            a_equity = random.choice([0, 1, 2, 3])
            
        elif phase == PHASE_REGULAR:
            a_ticket = random.choice([0, 1, 2, 3, 4])
            a_marketing = random.choice([0, 1, 2])
            
        elif phase == PHASE_TRADE_DEADLINE:
            a_roster = random.choice([2, 3]) # Hold or Buy
            
        return ActionVector(a_roster, a_salary, a_ticket, a_marketing, a_debt, a_equity)

    def _rollout(self, env_snapshot, depth):
        total_reward = 0
        gamma = 0.95
        curr_env = copy.deepcopy(env_snapshot)
        
        for d in range(depth):
            # Random Policy
            action = self._sample_random_valid_action(curr_env)
            _, reward, _ = curr_env.step(action)
            total_reward += (gamma**d) * reward
            
        return total_reward

if __name__ == "__main__":
    # Test Run
    real_env = MDPSimulationEnv()
    solver = MDPSolver()
    
    print("Starting Optimization Year 2024...")
    
    # Run 4 phases (1 year)
    for _ in range(4):
        print(f"Phase: {real_env.phase}")
        best_action = solver.plan(real_env)
        print(f"Selected Action: {best_action}")
        state, reward, _ = real_env.step(best_action)
        print(f"Reward: {reward:.2f}")
        print("-" * 20)
