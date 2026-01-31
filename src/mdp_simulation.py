import numpy as np
import copy
from src.mdp_definitions import *
from src.game_engine import GameEngine
from src.financial_model import FinancialModel
from src.player_generator import generate_synthetic_database, Player

class MDPSimulationEnv:
    def __init__(self):
        self.game_engine = GameEngine()
        self.financial_model = FinancialModel()
        
        # Reset Logic
        self.year = 2024
        self.phase = PHASE_OFFSEASON
        
        # Initialize Roster (State R)
        self.roster_df = generate_synthetic_database(num_teams=1, players_per_team=12) # Just target team for now
        
        # Initialize Financials (State F)
        self.f_state = FinancialState(
            leverage=0.2, # Healthy start
            cash_flow=0.0,
            psi_mean_salary=0.15,
            psi_std_salary=0.05,
            psi_max_salary_ratio=0.25,
            psi_guaranteed=10.0,
            cap_space_avail=0.0,
            tax_status=0,
            valuation_growth=0.0,
            owner_share=1.00, # 100%
            player_equity_pool=0.0,
            franchise_value=90.0, # IND 2024 anchor
            debt_stock=18.0 # 20% of 90
        )
        
        # Initialize Environment (State E)
        self.e_state = EnvState(
            macro=MACRO_NORMAL,
            cap_growth=CAP_GROWTH_SMOOTH,
            i_expansion=0,
            t_media_deal=2, # 2 years to 2026 deal
            mu_size=0.6, # Small Market (Indiana)
            compete_local=0, # Low competition
            n_star_fa=3,
            bidding_intensity=5.0
        )
        
        # Aux variables
        self.current_win_pct = 0.5
        self.current_srs = 0.0

    def step(self, action: ActionVector):
        """
        Executes one transition step in the MDP.
        S_t, A_t -> S_t+1, R_t
        """
        reward = 0.0
        
        # --- 1. Phase Specific Action Execution ---
        if self.phase == PHASE_OFFSEASON:
            self._handle_offseason_logic(action)
        elif self.phase == PHASE_REGULAR:
            self._handle_regular_season_logic(action)
        elif self.phase == PHASE_TRADE_DEADLINE:
            self._handle_trade_deadline(action)
        elif self.phase == PHASE_PLAYOFF:
            reward = self._handle_playoff_and_year_end(action)
        
        # --- 2. Phase Transition ---
        self._advance_phase()
        
        return self._get_full_state_dict(), reward, self.phase == PHASE_OFFSEASON # Done logic if year wrapped? No, continuous.

    def _handle_offseason_logic(self, action):
        # 1. Roster Strategy (Draft/Free Agency)
        # Simplified: If Buying, boost talent. If Tanking, reduce talent gain cap space.
        if action.a_roster == ACT_ROSTER_BUY or action.a_roster == ACT_ROSTER_ALLIN:
             # Simulate signing a star (simplified)
             # Pay cost if cap allows
             pass
        elif action.a_roster == ACT_ROSTER_TANK:
             # Shed salary
             pass
             
        # 2. Equity
        equity_grant = [0.0, 0.01, 0.02, 0.05][action.a_equity]
        if equity_grant > 0:
            self.f_state.owner_share *= (1.0 - equity_grant)
            self.f_state.player_equity_pool += equity_grant
            # Benefit: Reduce current payroll pressure or Boost Morale (Syn)
            
        # 3. Debt (Leverage)
        if action.a_debt == ACT_DEBT_PAYDOWN:
            self.f_state.debt_stock = max(0, self.f_state.debt_stock - 5.0) # Pay 5M
        elif action.a_debt == ACT_DEBT_BORROW:
            self.f_state.debt_stock += 10.0 # Borrow 10M
            
        # Update Leverage
        self.f_state.leverage = self.f_state.debt_stock / (self.f_state.franchise_value + 1e-6)

    def _handle_regular_season_logic(self, action):
        # 1. Game Simulation
        perf = self.game_engine.predict_performance(self.roster_df)
        self.current_win_pct = perf['Win_Pct']
        self.current_srs = perf['SRS']
        
        # 2. Ticket / Marketing
        ticket_mult = [0.9, 1.0, 1.1, 1.2, 1.3][action.a_ticket]
        marketing_inv = [0.5, 1.0, 2.0][action.a_marketing] # Millions cost
        
        # Store for year-end calc
        self.temp_marketing_cost = marketing_inv
        self.temp_ticket_mult = ticket_mult

    def _handle_trade_deadline(self, action):
        # Last chance to adjust roster
        if action.a_roster == ACT_ROSTER_BUY:
            # Boost Syn/Skill slightly for cost
            pass

    def _handle_playoff_and_year_end(self, action):
        # 1. Calculate Financials for the whole year
        
        # Revenue
        # Rev = Base * Macro * Market * Win% * TicketMult
        macro_factor = {0: 0.8, 1: 1.0, 2: 1.15}[self.e_state.macro]
        market_factor = self.e_state.mu_size
        
        economics = self.financial_model.calculate_economics(
            self.current_win_pct,
            self.roster_df['Salary_M'].sum(),
            marketing_boost=self.temp_marketing_cost # Simplified mapping
        )
        
        revenue = economics['Revenue_M'] * macro_factor * market_factor * self.temp_ticket_mult
        
        # Costs
        payroll = self.roster_df['Salary_M'].sum()
        interest = self.f_state.debt_stock * 0.05 # 5% Interest
        # Penalty if leverage high
        if self.f_state.leverage > 0.4:
            interest *= 1.5
            
        expenses = payroll + self.financial_model.cost_charter_flights + self.financial_model.cost_venue + self.temp_marketing_cost + interest
        
        # Cash Flow
        cf = revenue - expenses
        self.f_state.cash_flow = cf
        
        # Valuation Update
        # V_new = V_old * (1 + Growth)
        # Growth driven by Revenue Growth + Media Deal
        
        growth = 0.05 if cf > 0 else -0.05
        # Media Deal Spike
        if self.e_state.t_media_deal == 0:
            growth += 0.40 # 40% spike
            
        self.f_state.franchise_value *= (1.0 + growth)
        self.f_state.valuation_growth = growth
        self.f_state.leverage = self.f_state.debt_stock / self.f_state.franchise_value
        
        # Reward Function (Composite: Profit + Valuation)
        # Owner wants Cash + Equity Value
        actual_owner_value = self.f_state.franchise_value * self.f_state.owner_share
        step_reward = cf + (actual_owner_value * 0.05) # Cash now + 5% of asset value as "holding utility" or just change in value? 
        # Paper says: Maximize Profit AND Value. 
        # Typically Reward = CF + delta(EquityValue)
        
        # --- Time Transition Updates ---
        self._update_environment()
        self._update_roster_ages()
        
        self.year += 1
        return step_reward

    def _update_environment(self):
        # Macro Markov Chain
        # P(Recession|Recession)=0.7, etc
        trans_matrix = [
            [0.7, 0.25, 0.05],
            [0.15, 0.7, 0.15],
            [0.05, 0.3, 0.65]
        ]
        probs = trans_matrix[self.e_state.macro]
        self.e_state.macro = np.random.choice([0, 1, 2], p=probs)
        
        # Media Deal
        if self.e_state.t_media_deal > 0:
            self.e_state.t_media_deal -= 1
        else:
            self.e_state.t_media_deal = 10 # Reset loop
            
        # Expansion
        self.e_state.i_expansion = 1 if self.year in [2026, 2028] else 0

    def _update_roster_ages(self):
        # Age +1
        self.roster_df['Age'] += 1
        # Contract -1
        self.roster_df['Contract_Yrs'] -= 1
        
        # Remove expired
        # (Simplified: Just renew or replace with generic of same level for now to keep roster size constant)
        # In full simulation, this needs complex FA logic.
        # Here we just restore "Contract_Yrs" for simplicity to keep simulation running infinitely
        # unless it's a "Tank" year where we might replace with worse players.
        
        mask_expired = self.roster_df['Contract_Yrs'] <= 0
        self.roster_df.loc[mask_expired, 'Contract_Yrs'] = np.random.randint(1, 4, size=mask_expired.sum())
        
        # Skill decay for Age > 30
        mask_old = self.roster_df['Age'] > 30
        self.roster_df.loc[mask_old, 'WS_40'] *= 0.90 # 10% decline

    def _advance_phase(self):
        self.phase = (self.phase + 1) % 4

    def _get_full_state_dict(self):
        # Construct the massive state dictionary
        return {
            "Year": self.year,
            "Phase": self.phase,
            "R": self.game_engine.predict_performance(self.roster_df), # Simplified R
            "F": asdict(self.f_state),
            "E": asdict(self.e_state)
        }
