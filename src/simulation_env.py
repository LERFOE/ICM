import pandas as pd
import numpy as np
from src.player_generator import generate_synthetic_database
from src.game_engine import GameEngine
from src.financial_model import FinancialModel

class SimulationEnv:
    def __init__(self, target_team_name="Player_0_0"): 
        # By default generator uses generic names "Player_TeamID_Index"
        # We assume Team 0 is the "Target Team" (Indiana equivalent)
        
        self.db = generate_synthetic_database()
        self.target_team_id = 0
        self.game_engine = GameEngine()
        self.financial_model = FinancialModel()
        
        self.current_year = 2024
    
    def get_team_roster(self, team_id):
        # Filter by name pattern or add a 'TeamID' column to generator? 
        # The generator returns a DataFrame. 
        # Actually generator names are 'Player_X_Y'.
        # Let's assume we split by rows or parse name.
        # Ideally generator should return TeamID.
        # I'll parse it here.
        df = self.db.copy()
        # Extract TeamID from "Player_{id}_{idx}"
        df['TeamID'] = df['Name'].apply(lambda x: int(x.split('_')[1]))
        return df[df['TeamID'] == team_id]

    def get_lunwen_state_vector(self):
        """
        Constructs the strict R_t state vector as defined in lunwen.md
        """
        roster = self.get_team_roster(self.target_team_id)
        
        # 1. Q_t: Team Skill Aggregate (Time-weighted)
        agg_ws40, agg_usg, minutes = self.game_engine.calculate_roster_metrics(roster)
        Q_t = {
            'q_ws40': agg_ws40,
            'q_usg': agg_usg
        }
        
        # 2. C_t: Archetype Mix (Simplified to Pos count here)
        C_t = roster['Pos'].value_counts().to_dict()
        
        # 3. P_t: Positional Balance (Same as C_t for now)
        P_t = C_t
        
        # 4. L_t: Contract Maturity
        # Buckets: 1yr, 2yr, 3yr+
        L_t = {
            '1yr': len(roster[roster['Contract_Yrs'] == 1]),
            '2yr': len(roster[roster['Contract_Yrs'] == 2]),
            '3yr+': len(roster[roster['Contract_Yrs'] >= 3])
        }
        
        # 5. A_t: Age Profile
        A_t = {
            'mean_age': roster['Age'].mean(),
            'var_age': roster['Age'].var(),
            'age_28_plus': len(roster[roster['Age'] >= 28])
        }
        
        # 6. Synergies and Perf
        perf = self.game_engine.predict_performance(roster)
        
        return {
            'Q_t': Q_t,
            'C_t': C_t,
            'P_t': P_t,
            'L_t': L_t,
            'A_t': A_t,
            'Win_Pct': perf['Win_Pct'],
            'SRS': perf['SRS'],
            'Syn_t': -perf['Synergy_Penalty'] # Synergy is negative of penalty
        }

    def run_season_simulation(self, marketing_boost=1.0):
        """
        Runs the full pipeline defined in the problem.
        Roster -> GameEngine -> Win% -> FinancialModel -> Outcomes.
        """
        roster = self.get_team_roster(self.target_team_id)
        
        # 1. Basketball Perf
        perf = self.game_engine.predict_performance(roster)
        
        # 2. Financial Perf
        payroll = roster['Salary_M'].sum()
        fin = self.financial_model.calculate_economics(
            perf['Win_Pct'], 
            payroll, 
            marketing_boost=marketing_boost
        )
        
        return {
            'Year': self.current_year,
            'Performance': perf,
            'Financials': fin,
            'Roster_Stats': {
                'Payroll': payroll,
                'Avg_Age': roster['Age'].mean()
            }
        }
