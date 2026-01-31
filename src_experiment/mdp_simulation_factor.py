import numpy as np
import pandas as pd
import copy
import random
from src_experiment.mdp_definitions import *
from src_experiment.game_engine_factor import FactorGameEngine
from src_experiment.player_generator_factor import FactorPlayerGenerator

class FinancialState:
    def __init__(self):
        self.cash = 10.0 # Million (Cash Reserve)
        self.debt = 50.0 # Million
        self.franchise_value = 500.0 # Million
        self.salary_cap = 100.0 
        self.owner_equity_pct = 1.00 # 100%

    @property
    def leverage(self):
        return self.debt / (self.franchise_value + 1e-6)

class EnvState:
    def __init__(self):
        self.macro_economy = 'Normal' # Normal, Recession, Boom
        self.year = 2024
        
class ExperimentSimulationEnv:
    def __init__(self):
        self.game_engine = FactorGameEngine()
        self.player_gen = FactorPlayerGenerator()
        
        self.f_state = FinancialState()
        self.e_state = EnvState()
        
        # Initialize Roster (Balanced Start)
        self.roster_df = self.player_gen.generate_roster({
            'Star': 2, 'Starter': 3, 'Rotation': 4, 'Bench': 4
        })
        
        # Cache performance
        self.last_perf = {'Win_Pct': 0.5, 'SRS': 0.0, 'Synergy': 0.0}

    def step(self, action: ActionVector):
        """
        Transitions the environment based on Action.
        Returns: observation (state), reward, done
        """
        # 1. Execute Financial Actions (Debt)
        self._handle_finance_action(action)
        
        # 2. Execute Roster Actions (Buy/Sell)
        self._handle_roster_action(action)
        
        # 3. Simulate Season (Performance)
        perf = self.game_engine.predict_performance(self.roster_df)
        self.last_perf = perf
        
        # 4. Calculate Economics (Rev, Cost, Profit)
        profit, val_growth = self._calculate_economics(perf, action)
        
        # 5. Update Financial State (Value, Cash)
        self.f_state.cash += profit
        self.f_state.franchise_value *= (1.0 + val_growth)
        
        # Prevent bankruptcy (simplified)
        if self.f_state.cash < -20.0:
            # Emergency loan / Forced sale penalty
            penalty = -50.0
            self.f_state.cash = 10.0
            self.f_state.debt += 30.0
        else:
            penalty = 0.0
            
        # 6. Update Environment (Macro Cycle)
        self._update_macro_economy()
        self.e_state.year += 1
        
        # 7. Reward Calculation
        # Goal: Maximize Profit + Value Growth
        # Reward = Profit (normalized) + Value Change (normalized)
        
        # Normalized Profit ~ 10-20M range
        reward_profit = profit 
        
        # Value delta (Net Equity Change)
        # Equity = FV - Debt
        current_equity = self.f_state.franchise_value - self.f_state.debt
        # We can't easily track prev equity without storing it, but Val Growth approximates it.
        # Let's use Val Growth * Base
        reward_value = (self.f_state.franchise_value * val_growth) * 0.1 # Scale down
        
        total_reward = reward_profit + reward_value + penalty
        
        return self, total_reward, False

    def _handle_finance_action(self, action):
        # Thesis: Leverage adjustment
        if action.a_debt == ACT_DEBT_PAYDOWN:
            # Pay 10M if possible
            payment = min(self.f_state.cash, 10.0)
            self.f_state.debt -= payment
            self.f_state.cash -= payment
        elif action.a_debt == ACT_DEBT_BORROW:
            # Borrow 20M
            self.f_state.debt += 20.0
            self.f_state.cash += 20.0
            
    def _handle_roster_action(self, action):
        # Simplified: Buy = Add 'Starter' quality player, Sell = Remove best player
        if action.a_roster == ACT_ROSTER_BUY:
            # Cost money to sign
            signing_bonus = 5.0
            if self.f_state.cash > signing_bonus:
                self.f_state.cash -= signing_bonus
                # Add good player
                new_player = self.player_gen.generate_player(quality='Star')
                # Replace worst player to keep roster size 13
                self.roster_df = self.roster_df.sort_values('TS%', ascending=True) # Simple sort
                self.roster_df.iloc[0] = pd.Series(new_player) # Replace
                
        elif action.a_roster == ACT_ROSTER_SELL:
            # Gain assets/cash, lose talent
            self.f_state.cash += 5.0 
            # Remove best player, replace with Bench
            new_player = self.player_gen.generate_player(quality='Bench')
            self.roster_df = self.roster_df.sort_values('TS%', ascending=False)
            self.roster_df.iloc[0] = pd.Series(new_player)

    def _calculate_economics(self, perf, action):
        # 1. Revenue
        # Base: 100M. 
        # Macro Factor: Recession=0.8, Boom=1.2
        # Win Factor: +1% Rev per 1% Win over 50%
        macro_mult = 1.0
        if self.e_state.macro_economy == 'Recession': macro_mult = 0.8
        elif self.e_state.macro_economy == 'Boom': macro_mult = 1.2
        
        win_bonus = (perf['Win_Pct'] - 0.5) * 100.0 # e.g. 0.6 -> +10M
        marketing_bonus = 5.0 if action.a_marketing == ACT_MARKETING_HIGH else 0.0
        
        revenue = (100.0 + win_bonus) * macro_mult + marketing_bonus
        
        # 2. Costs
        # Salary
        total_salary = self.roster_df['Salary'].sum()
        # Interest: Rate depends on Leverage (Thesis Section 11)
        # r = r_free + beta * leverage
        r_free = 0.04
        beta = 0.05
        lev = self.f_state.leverage
        interest_rate = r_free + beta * lev
        interest_payment = self.f_state.debt * interest_rate
        
        ops_cost = 20.0 # Fixed
        marketing_cost = 5.0 if action.a_marketing == ACT_MARKETING_HIGH else 0.0
        
        total_cost = total_salary + interest_payment + ops_cost + marketing_cost
        
        profit = revenue - total_cost
        
        # 3. Valuation Growth
        # Driven by Profit Growth + Market Sentiment
        # V_t = gamma * Win_Delta + ...
        # Simplified:
        growth = 0.03 # Inflation
        if profit > 10.0: growth += 0.02
        if perf['Win_Pct'] > 0.6: growth += 0.03
        if self.e_state.macro_economy == 'Boom': growth += 0.05
        if self.e_state.macro_economy == 'Recession': growth -= 0.05
        
        return profit, growth

    def _update_macro_economy(self):
        # Markov Chain Transition
        # Normal -> 10% Recession, 20% Boom
        r = random.random()
        current = self.e_state.macro_economy
        
        if current == 'Normal':
            if r < 0.1: self.e_state.macro_economy = 'Recession'
            elif r > 0.8: self.e_state.macro_economy = 'Boom'
        elif current == 'Recession':
            if r < 0.3: self.e_state.macro_economy = 'Normal' # Recovery
        elif current == 'Boom':
            if r < 0.2: self.e_state.macro_economy = 'Normal' # Cooling
            
    def copy(self):
        new_env = ExperimentSimulationEnv()
        new_env.f_state = copy.deepcopy(self.f_state)
        new_env.e_state = copy.deepcopy(self.e_state)
        new_env.roster_df = self.roster_df.copy()
        new_env.last_perf = self.last_perf
        return new_env

