import pandas as pd
import sys
import os
import numpy as np
import copy

# Ensure we can import from src
sys.path.append(os.getcwd())

from src.mdp_simulation import MDPSimulationEnv, PHASE_OFFSEASON, PHASE_PLAYOFF
from src.mdp_solver import MDPSolver, MCTSNode
from src.mdp_definitions import *

class EnhancedMDPSolver(MDPSolver):
    def __init__(self, horizon=5, simulations=200):
        super().__init__(horizon)
        self.simulations = simulations # Increased from default

    def plan_with_analysis(self, current_real_env):
        """
        Runs MCTS and returns detailed analysis of top choices.
        """
        root = MCTSNode(state_snapshot=copy.deepcopy(current_real_env))
        
        # Run Simulations
        for _ in range(self.simulations): 
            node = root
            # Selection
            while node.is_fully_expanded() and node.children:
                node = node.best_child()
            # Expansion
            if not node.is_fully_expanded():
                action = self._sample_random_valid_action(node.state_snapshot)
                new_state_snapshot = copy.deepcopy(node.state_snapshot)
                _, reward, _ = new_state_snapshot.step(action)
                child = MCTSNode(state_snapshot=new_state_snapshot, parent=node, action_from_parent=action)
                node.children.append(child)
                node = child
            # Simulation (Rollout)
            rollout_val = self._rollout(node.state_snapshot, depth=4) # Deeper rollout
            # Backprop
            curr = node
            while curr:
                curr.visits += 1
                curr.value_sum += rollout_val
                curr = curr.parent
                
        # Analysis
        print(f"  > Thinking Process ({self.simulations} sims):")
        sorted_children = sorted(root.children, key=lambda c: c.visits, reverse=True)
        top_3 = sorted_children[:3]
        
        best_child = None
        best_score = -float('inf')

        for i, child in enumerate(top_3):
            avg_val = child.value_sum / child.visits
            act = child.action_from_parent
            
            # Simple description of action intent
            intent = []
            if act.a_debt == ACT_DEBT_BORROW: intent.append("Borrow")
            elif act.a_debt == ACT_DEBT_PAYDOWN: intent.append("Paydown")
            else: intent.append("HoldDebt")
            
            if act.a_roster == ACT_ROSTER_BUY: intent.append("BuyPlayer")
            elif act.a_roster == ACT_ROSTER_TANK: intent.append("Tank")
            else: intent.append("HoldRoster")
            
            if act.a_equity > 0: intent.append(f"GiveEquity({[0,1,2,5][act.a_equity]}%)")

            print(f"    Option {i+1}: {' + '.join(intent)} -> Exp.Value: {avg_val:.1f} (Visits: {child.visits})")
            
            if avg_val > best_score:
                best_score = avg_val
                best_child = child
                
        return best_child.action_from_parent if best_child else self._sample_random_valid_action(current_real_env)


def run_enhanced_model():
    env = MDPSimulationEnv()
    # Level 1 Upgrade: Deep Horizon (12 steps = 3 years) & High Simulations (1000)
    solver = EnhancedMDPSolver(horizon=12, simulations=1000) 
    
    print("\n=== Enhanced MCM Simulation: 'The Expert Manager' (Deep Thought Version) ===")
    print("-----------------------------------------------------")
    
    # Simulate just 3 critical years to show depth
    for year in range(2024, 2027):
        while env.year == year:
            # Only focus on Offseason for deep analysis
            if env.phase == PHASE_OFFSEASON:
                print(f"\n[Year {env.year} OFFSEASON STRATEGY SESSION]")
                print(f"  Context: Macro={['Recession','Normal','Boom'][env.e_state.macro]}, "
                      f"MediaDealIn={env.e_state.t_media_deal} yrs")
                
                action = solver.plan_with_analysis(env)
                
                print(f"  >> SELECTED STRATEGY: Debt={action.a_debt}, Roster={action.a_roster}")
                env.step(action)
            else:
                # Fast forward other phases
                dummy_action = solver._sample_random_valid_action(env)
                env.step(dummy_action)
                
    print("\nEnhanced simulation complete.")

if __name__ == "__main__":
    run_enhanced_model()
