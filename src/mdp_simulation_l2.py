import pandas as pd
import numpy as np
from src.mdp_definitions_l2 import ActionVector, MicroRosterAction
from src.mdp_simulation import MDPSimulationEnv, PHASE_OFFSEASON

class MDPSimulationEnvL2(MDPSimulationEnv):
    def step(self, action: ActionVector):
        # Hijack the roster logic to apply Micro Actions
        if hasattr(action, 'micro_action') and action.micro_action and self.phase == PHASE_OFFSEASON:
            self._apply_micro_roster_move(action.micro_action)
            
        # Call original step (it will use the modified roster)
        return super().step(action)

    def _apply_micro_roster_move(self, move: MicroRosterAction):
        """
        Executes specific player swaps based on Archetypes.
        """
        # Logic: 
        # 1. 'TRADE', 'Star_Guard', 'Pick': Remove Pick, Add Player with high USG small WS
        # 2. 'TRADE', 'Rim_Protector', 'Star_Guard': Remove high USG Guard, Add low USG high DBPM Center
        
        if move.transaction_type == 'TRADE':
            if move.target_archetype == 'Rim_Protector' and move.asset_out == 'Star_Guard':
                # Find a high usage guard to remove
                guards = self.roster_df[self.roster_df['Pos'] == 'G']
                if not guards.empty:
                    # Find highest usage
                    idx_to_remove = guards['USG%'].idxmax()
                    
                    # Create the incoming Center
                    new_player = {
                        'Name': f"Traded_Center_{self.year}",
                        'Pos': 'C',
                        'Age': 26,
                        'Salary_M': guards.loc[idx_to_remove, 'Salary_M'], # Matching salary
                        'Contract_Yrs': 2,
                        'WS_40': 0.150, # Solid
                        'TS%': 0.60,
                        'USG%': 0.15, # Low Usage!
                        'AST%': 0.05,
                        'TRB%': 0.18,
                        'DBPM': 2.5,
                        'TeamID': 0
                    }
                    
                    # Swap
                    self.roster_df = self.roster_df.drop(idx_to_remove)
                    self.roster_df = pd.concat([self.roster_df, pd.DataFrame([new_player])], ignore_index=True)
                    # print(f"  [L2 Action] Traded High-Usage Guard for Rim Protector to fix Synergy.")
