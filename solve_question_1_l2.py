import numpy as np
import copy
import random
from src.mdp_definitions import *
from src.mdp_definitions_l2 import ActionVector, MicroRosterAction
from src.mdp_simulation_l2 import MDPSimulationEnvL2
from src.mdp_solver import MDPSolver, MCTSNode

class MDPSolverL2(MDPSolver):
    def _sample_random_valid_action(self, env):
        # Base Action
        base_act = super()._sample_random_valid_action(env)
        
        # Add Micro Action if Synergy is bad
        # Check env state directly
        # Calculate current Synergy
        # (We need to reach into GameEngine logic or approximating it)
        
        # Heuristic: If we are in Offseason, maybe try a trade
        micro = None
        if env.phase == 0: # Offseason
            # Randomly propose a "fix synergy" trade
            # 50% chance to propose trading a guard for a big if we have too many guards
            # Simplified for demo
            if random.random() < 0.3:
                micro = MicroRosterAction(
                    transaction_type='TRADE',
                    target_archetype='Rim_Protector',
                    asset_out='Star_Guard'
                )
        
        return ActionVector(
            base_act.a_roster, base_act.a_salary, base_act.a_ticket,
            base_act.a_marketing, base_act.a_debt, base_act.a_equity,
            micro_action=micro
        )

def run_l2_simulation():
    env = MDPSimulationEnvL2()
    # Inject a "Bad Synergy" Roster to start
    # 2 Ball Hegs
    env.roster_df.loc[0, 'USG%'] = 0.35
    env.roster_df.loc[0, 'Pos'] = 'G'
    env.roster_df.loc[1, 'USG%'] = 0.35
    env.roster_df.loc[1, 'Pos'] = 'G'
    
    print("\n=== Level 2: 'Human-Centric' Micro-Decisions ===")
    print("Initial State: 2 Guards with 35% Usage each. Synergy is terrible.")
    
    # Check Initial Perf
    perf = env.game_engine.predict_performance(env.roster_df)
    print(f"Initial Win%: {perf['Win_Pct']:.3f} (Synergy Penalty: {perf['Synergy_Penalty']:.3f})")
    
    solver = MDPSolverL2(horizon=3) # Short horizon but smart actions
    
    # Run 1 Step Planning
    print("\n... Solver analyzing Trade Options ...")
    action = solver.plan(env)
    
    if action.micro_action:
        print(f"Solver Selected Micro-Action: {action.micro_action}")
        env.step(action)
        
        perf_new = env.game_engine.predict_performance(env.roster_df)
        print(f"Post-Trade Win%: {perf_new['Win_Pct']:.3f} (Synergy Penalty: {perf_new['Synergy_Penalty']:.3f})")
        
        if perf_new['Win_Pct'] > perf['Win_Pct']:
            print("SUCCESS: Intelligent Roster Construction improved performance!")
    else:
        print("Solver decided to HOLD.")

if __name__ == "__main__":
    run_l2_simulation()
