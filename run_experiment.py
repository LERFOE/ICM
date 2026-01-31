import pandas as pd
import json
from src.simulation_env import SimulationEnv

def run():
    print("Initializing Simulation Environment...")
    env = SimulationEnv()
    
    # 1. Report Initial State (Strictly per Lunwen.md)
    state = env.get_lunwen_state_vector()
    print("\n[Initial State R_t Check]")
    print(json.dumps(state, indent=2))
    
    # 2. Run Baseline Scenario (Normal Year)
    print("\n[Scenario 1: Baseline Year (No Marketing Boost)]")
    res_base = env.run_season_simulation(marketing_boost=1.0)
    print_results(res_base)
    
    # 3. Run "Caitlin Clark Effect" Scenario
    # Assuming Marketing Boost = 2.5x (Att 4000 -> 10000+)
    print("\n[Scenario 2: The 'Caitlin Clark' Effect (Marketing Boost x2.5)]")
    res_boost = env.run_season_simulation(marketing_boost=2.5)
    print_results(res_boost)
    
    # 4. Compare Valuation Impact
    val_diff = res_boost['Financials']['Valuation_M'] - res_base['Financials']['Valuation_M']
    print(f"\n[Impact Analysis]")
    print(f"Valuation Increase due to Star Power: ${val_diff}M")

def print_results(res):
    perf = res['Performance']
    fin = res['Financials']
    print(f"  Win%: {perf['Win_Pct']:.3f} (Est Wins: {perf['Wins']:.1f})")
    print(f"  SRS:  {perf['SRS']:.3f}")
    print(f"  Att:  {fin['Avg_Att']} (Total: {fin['Total_Att']})")
    print(f"  Rev:  ${fin['Revenue_M']}M")
    print(f"  Exp:  ${fin['Expenses_M']}M")
    print(f"  OpInc:${fin['Operating_Income_M']}M")
    print(f"  Val:  ${fin['Valuation_M']}M")

if __name__ == "__main__":
    run()
