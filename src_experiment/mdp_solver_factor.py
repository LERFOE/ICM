import numpy as np
import copy
import random
from src_experiment.mdp_definitions import *
from src_experiment.mdp_simulation_factor import ExperimentSimulationEnv

class ExperimentSolver:
    def __init__(self, horizon=5, simulations=200):
        self.horizon = horizon
        self.simulations = simulations

    def plan(self, env):
        """
        Input: ExperimentSimulationEnv
        Returns: Best ActionVector
        """
        # Valid Actions Space Exploration
        # We simplify to a set of "Strategic Packages" to reduce search space
        # Strategy A: Aggressive Growth (Borrow + Buy + Market)
        # Strategy B: Conservative (Hold + Paydown)
        # Strategy C: Rebuild (Sell + Paydown)
        # Strategy D: Status Quo (Hold + Hold)
        
        strategies = [
            ActionVector(ACT_ROSTER_BUY, ACT_SALARY_OVERCAP, ACT_TICKET_NORMAL, ACT_MARKETING_HIGH, ACT_DEBT_BORROW, ACT_EQUITY_0), # A
            ActionVector(ACT_ROSTER_HOLD, ACT_SALARY_FLOOR, ACT_TICKET_NORMAL, ACT_MARKETING_NORMAL, ACT_DEBT_PAYDOWN, ACT_EQUITY_0), # B
            ActionVector(ACT_ROSTER_SELL, ACT_SALARY_FLOOR, ACT_TICKET_DISCOUNT, ACT_MARKETING_MIN, ACT_DEBT_PAYDOWN, ACT_EQUITY_0), # C
            ActionVector(ACT_ROSTER_HOLD, ACT_SALARY_FLOOR, ACT_TICKET_NORMAL, ACT_MARKETING_NORMAL, ACT_DEBT_HOLD, ACT_EQUITY_0), # D
        ]
        
        best_avg_reward = -float('inf')
        best_action = strategies[0]
        
        for strat in strategies:
            total_reward = 0.0
            for _ in range(self.simulations):
                total_reward += self._rollout(env, strat)
            
            avg_reward = total_reward / self.simulations
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_action = strat
                
        return best_action

    def _rollout(self, env, first_action):
        sim_env = env.copy()
        cumulative_reward = 0.0
        discount = 0.95
        
        # Step 1: Fixed First Action
        _, r, _ = sim_env.step(first_action)
        cumulative_reward += r
        
        # Step 2..H: Random Policy (Simplified)
        for t in range(1, self.horizon):
            random_act = ActionVector(
                random.choice([0,1,2]), # Roster
                0, 1, 1, # Salary, Ticket, Marketing
                random.choice([0,1,2]), # Debt
                0
            )
            _, r, _ = sim_env.step(random_act)
            cumulative_reward += (discount ** t) * r
            
        return cumulative_reward
