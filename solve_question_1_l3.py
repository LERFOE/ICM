import numpy as np
import copy
import random
from src.mdp_definitions import *
from src.mdp_simulation_l2 import MDPSimulationEnvL2
from src.mdp_solver import MDPSolver

class OpponentAgent:
    def __init__(self, strategy_type):
        self.strategy = strategy_type # 'Aggressive', 'Conservative', 'Rebuilding'
        self.cap_space = 1.0 # Millions
        
    def make_bid(self, player_value):
        # Returns a bid amount in Millions
        if self.strategy == 'Aggressive':
            return player_value * 1.2
        elif self.strategy == 'Conservative':
            return player_value * 0.9
        return 0.0

class MultiAgentEnvironment(MDPSimulationEnvL2):
    def __init__(self):
        super().__init__()
        # Initialize 11 Opponents
        self.opponents = []
        for i in range(11):
            strat = random.choice(['Aggressive', 'Conservative', 'Rebuilding'])
            self.opponents.append(OpponentAgent(strat))
            
    def get_market_bidding_intensity(self):
        # Dynamic calculation based on opponents status
        market_heat = 0.0
        for opp in self.opponents:
            if opp.strategy == 'Aggressive':
                market_heat += 2.0
            elif opp.strategy == 'Conservative':
                market_heat += 0.5
        return market_heat
    
    def step(self, action):
        # Update Opponents Strategies randomly (Simulate League dynamics)
        if self.phase == PHASE_OFFSEASON:
            # Randomly shift strategies
            for opp in self.opponents:
                if random.random() < 0.2: # 20% chance to change strategy
                     opp.strategy = random.choice(['Aggressive', 'Conservative', 'Rebuilding'])
                     
            # Update the observed environment state 'bidding_intensity'
            self.e_state.bidding_intensity = self.get_market_bidding_intensity()

        # Handle Micro Action properly if it exists, use base class step
        # Pass the action up the chain
        next_state, reward, done = super().step(action)
        
        # --- MARKET HEAT LOGIC ---
        # If the market is Overheated (High Intensity) and we try to BUY,
        # we suffer a penalty (paying premium, bad contracts).
        # This teaches the agent to be contrarian.
        if action.a_roster == ACT_ROSTER_BUY or action.a_roster == ACT_ROSTER_ALLIN:
             heat = self.get_market_bidding_intensity()
             # Heat ranges from ~5 (Cold) to ~22 (Hot)
             # If Heat > 15, we are in a seller's market.
             if heat > 15.0:
                 # Penalty scales with heat
                 # Buying into a frenzy is -EV.
                 penalty = (heat - 10.0) * 1.5 # e.g. Heat 22 -> (12)*1.5 = -18 Reward
                 reward -= penalty
                 
        return next_state, reward, done

def run_l3_multi_agent():
    env = MultiAgentEnvironment()
    solver = MDPSolver(horizon=5) 
    
    print("\n=== Level 3: Multi-Agent Market Dynamics ===")
    
    # Simulate a "League Expansion" event where everyone has money
    print("\n[Event: New TV Deal Injects Cash -> Everyone becomes Aggressive]")
    for opp in env.opponents:
        opp.strategy = 'Aggressive'
        
    # Check Market Heat
    heat = env.get_market_bidding_intensity()
    env.e_state.bidding_intensity = heat
    print(f"Market Heat Index: {heat} (Extremely High)")
    
    print("Solver Planning...")
    action = solver.plan(env)
    
    act_str = "BUY" if action.a_roster == ACT_ROSTER_BUY else "TRA/SELL" if action.a_roster == ACT_ROSTER_TANK else "HOLD"
    print(f"Solver Decision given High Heat: {act_str}")
    
    if action.a_roster == ACT_ROSTER_TANK:
        print("SUCCESS: Solver chose to SELL assets into the overheated market!")
    elif action.a_roster == ACT_ROSTER_BUY:
        print("FAIL: Solver bought at the top.")
    else:
        print("NEUTRAL: Solver held.")

if __name__ == "__main__":
    run_l3_multi_agent()
