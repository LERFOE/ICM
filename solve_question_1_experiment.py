import sys
import os
# Ensure project root is in path
sys.path.append(os.getcwd())

from src_experiment.mdp_simulation_factor import ExperimentSimulationEnv
from src_experiment.mdp_solver_factor import ExperimentSolver
from src_experiment.mdp_definitions import *

def run_experiment_q1():
    print("=== MCM Question 1: Dynamic Decision Model Experiment ===")
    print("Goal: Adjust Leverage based on Macro Economy and Team Performance\n")
    
    env = ExperimentSimulationEnv()
    solver = ExperimentSolver(horizon=5, simulations=100) # Lower sims for speed
    
    # Run a 10-Year Simulation
    history = []
    
    print(f"{'Year':<6} | {'Macro':<10} | {'Leverage':<8} | {'Cash':<8} | {'Win%':<6} | {'Action (D/R)':<15} | {'Profit':<8} | {'Val(M)':<8}")
    print("-" * 100)
    
    # Force Specific Economic Scenarios for Demonstration
    # Years 1-3: Normal
    # Years 4-6: Recession (Stress Test)
    # Years 7-10: Boom (Opportunity)
    
    scenario_map = {
        1: 'Normal', 2: 'Normal', 3: 'Normal',
        4: 'Recession', 5: 'Recession', 6: 'Recession',
        7: 'Boom', 8: 'Boom', 9: 'Boom', 10: 'Boom'
    }
    
    for year in range(1, 11):
        # 1. Force Environment
        env.e_state.macro_economy = scenario_map.get(year, 'Normal')
        
        # 2. Solver Plan
        action = solver.plan(env)
        
        # Log Pre-Step State
        lev_start = env.f_state.leverage
        cash_start = env.f_state.cash
        
        # 3. Step
        _, r, _ = env.step(action)
        
        # Log Results
        act_str = f"Debt:{action.a_debt}/Rost:{action.a_roster}"
        # Decode: Debt 0=Pay, 2=Borrow. Roster 0=Hold, 1=Buy, 2=Sell.
        
        # Profit Calculation for display (Cash delta)
        profit = env.f_state.cash - cash_start - (20.0 if action.a_debt==2 else 0) + (10.0 if action.a_debt==0 else 0)
        # Note: Step updates cash w/ financing, so we adjust back to see operating profit roughly
        
        print(f"{year:<6} | {env.e_state.macro_economy:<10} | {lev_start:.2f}     | {cash_start:.1f}     | {env.last_perf['Win_Pct']:.2f}   | {act_str:<15} | {profit:.1f}     | {env.f_state.franchise_value:.0f}")

        history.append({
            'Year': year,
            'Macro': env.e_state.macro_economy,
            'Action': act_str,
            'Leverage': lev_start,
            'Value': env.f_state.franchise_value
        })

    print("\n=== Analysis ===")
    print("1. Recession Response (Years 4-6):")
    recession_acts = [h['Action'] for h in history if h['Macro'] == 'Recession']
    print(f"   Actions taken: {recession_acts}")
    
    print("2. Boom Response (Years 7-10):")
    boom_acts = [h['Action'] for h in history if h['Macro'] == 'Boom']
    print(f"   Actions taken: {boom_acts}")

if __name__ == "__main__":
    run_experiment_q1()
