import pandas as pd
import sys
import os

# Ensure we can import from src
sys.path.append(os.getcwd())

from src.mdp_simulation import MDPSimulationEnv, PHASE_OFFSEASON, PHASE_PLAYOFF
from src.mdp_solver import MDPSolver
from src.mdp_definitions import ACT_DEBT_PAYDOWN, ACT_DEBT_BORROW, ACT_DEBT_HOLD

def run_q1_dynamic_decision_model():
    """
    Simulates the Dynamic Decision Model over 5 years.
    demonstrating how the agent adjusts leverage (a_debt) 
    in response to Performance (Win%) and Economy (Macro).
    """
    
    # 1. Initialize the Environment (The World)
    env = MDPSimulationEnv()
    
    # 2. Initialize the Solver (The Brain)
    # Horizon=5 implies the agent looks 5 steps ahead (approx 1.25 years)
    solver = MDPSolver(horizon=5)
    
    history = []
    
    print("\n=== MCM Problem 1: Dynamic Decision Model Simulation ===")
    print("Goal: Maximize Profit & Valuation via Leverage Adjustment")
    print("--------------------------------------------------------")
    
    # Simulate 5 Seasons (2024-2029)
    start_year = env.year
    end_year = start_year + 5
    
    while env.year < end_year:
        current_year = env.year
        phase_name = ["Offseason", "Regular", "TradeDeadline", "Playoff"][env.phase]
        
        # --- 1. Observe State (Situation Assessment) ---
        current_leverage = env.f_state.leverage
        current_win_pct = env.current_win_pct
        current_macro = ["Recession", "Normal", "Boom"][env.e_state.macro]
        current_valuation = env.f_state.franchise_value
        
        # --- 2. Solver Decides (Action Selection) ---
        # The solver runs MCTS simulations to find the action that maximizes future Reward
        action = solver.plan(env)
        
        # --- 3. Interpret Decision ---
        debt_action_str = "HOLD"
        leverage_intent = "Maintain Structure"
        
        if action.a_debt == ACT_DEBT_BORROW: 
            debt_action_str = "BORROW (++D)"
            leverage_intent = "Aggressive Expansion (Invest in Roster/Ops)"
        elif action.a_debt == ACT_DEBT_PAYDOWN: 
            debt_action_str = "PAYDOWN (--D)"
            leverage_intent = "Risk Mitigation (Reduce Interest Burden)"
            
        # --- 4. Execute Action (Transition) ---
        state_dict, reward, _ = env.step(action)
        
        # --- 5. Logging Key Decision Points ---
        # We mainly care about Offseason (Financial Restructuring) and Playoff (Year End Result)
        
        if env.phase == (PHASE_OFFSEASON + 1) % 4: # Just finished Offseason logic
            print(f"\n[Year {current_year} OFFSEASON DECISION]")
            print(f"  Context:  Eco={current_macro}, Win%={current_win_pct:.2f}, Lev={current_leverage:.2%}")
            print(f"  Action:   {debt_action_str} -> {leverage_intent}")
            print(f"  Other:    RosterAction={action.a_roster}, EquityGrant={action.a_equity}")

        if env.phase == (PHASE_PLAYOFF + 1) % 4: # Just finished Playoff/Year-End logic
            # Previous step was Playoff, which calculates full year financials
            cf = env.f_state.cash_flow
            val_growth = env.f_state.valuation_growth
            print(f"[Year {current_year} RESULTS]")
            print(f"  Profit:   ${cf:.2f}M")
            print(f"  Valuation: ${env.f_state.franchise_value:.2f}M (Growth: {val_growth:.1%})")
            
        # Record Data for Analysis
        history.append({
            'Year': current_year,
            'Phase': phase_name,
            'Macro_State': current_macro,
            'Win_Pct': current_win_pct,
            'Leverage': current_leverage,
            'Action_Debt': action.a_debt,
            'Action_Roster': action.a_roster,
            'Cash_Flow': env.f_state.cash_flow,
            'Franchise_Value': env.f_state.franchise_value
        })

    # Save Results
    df = pd.DataFrame(history)
    output_file = "q1_model_output.csv"
    df.to_csv(output_file, index=False)
    print(f"\n Simulation Complete. Detailed tables saved to {output_file}")
    
    # Simple Analysis of Strategy
    print("\n[Strategic Insight Generated]")
    avg_lev_recession = df[df['Macro_State']=='Recession']['Leverage'].mean()
    avg_lev_boom = df[df['Macro_State']=='Boom']['Leverage'].mean()
    
    if pd.notna(avg_lev_recession) and pd.notna(avg_lev_boom):
        print(f"  Avg Leverage in Recession: {avg_lev_recession:.1%}")
        print(f"  Avg Leverage in Boom:      {avg_lev_boom:.1%}")
        if avg_lev_recession < avg_lev_boom:
            print("  -> Model demonstrates 'Counter-Cyclical' risk management (De-leveraging in bad times).")
        else:
            print("  -> Model demonstrates 'Aggressive' growth regardless of cycle.")

if __name__ == "__main__":
    run_q1_dynamic_decision_model()
