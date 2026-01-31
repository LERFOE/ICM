import numpy as np

class GameEngine:
    def __init__(self):
        # Calibrated Parameters from calibrate_model.py
        self.alpha_srs = 0.0221
        self.beta_srs = 0.5042
        
        # Mapping WS/40 to SRS
        # Win% = 5 * Mean_WS40 (Theoretical)
        # 5 * Mean_WS40 = alpha * SRS + beta
        # SRS = (5 * Mean_WS40 - beta) / alpha
        
    def calculate_roster_metrics(self, roster_df):
        """
        Takes a DataFrame of players (must satisfy Roster constraints).
        Assumes minutes are allocated optimized for best players.
        """
        # Simple Minutes Allocation Logic:
        # Sort by WS/40 descending.
        # Top 5 get 30 min, Next 3 get 15 min, Rest get garbage.
        # Total Min per game = 200.
        
        sorted_roster = roster_df.sort_values('WS_40', ascending=False).reset_index(drop=True)
        
        minutes = np.zeros(len(sorted_roster))
        # Top 5
        minutes[0:5] = 32.0 
        # Next 3
        minutes[5:8] = 13.0
        # Remaining share 1 min or 0
        minutes[8:] = 0.0
        
        # Normalize to 200
        total_assigned = np.sum(minutes)
        if total_assigned > 0:
            minutes = minutes * (200.0 / total_assigned)
            
        # Calculate Weighted Aggregate Metrics
        weights = minutes / 200.0
        
        agg_ws40 = np.sum(sorted_roster['WS_40'] * weights)
        agg_usg = np.sum(sorted_roster['USG%'] * weights)
        
        return agg_ws40, agg_usg, minutes

    def calculate_synergy(self, agg_usg):
        """
        Penalty if usage is too concentrated or too diffuse?
        Actually, high usage players usually overlap.
        If weighted sum of USG% > 25% (roughly 1.25x avg), efficiency might drop?
        League avg USG is 20%. Sum of 5 * 20% = 100%.
        Weighted avg should be ~20%.
        If we hold 'Ball Hogs', Weighted Avg USG > 23-24%.
        """
        # Simple quadratic penalty if Agg USG deviates from optimal ~22%
        # Derived heuristic
        optimal_usg = 0.22
        penalty = 0.0
        if agg_usg > optimal_usg:
            penalty = (agg_usg - optimal_usg) * 5.0 # Arbitrary scalar
            
        return max(0, penalty)

    def predict_performance(self, roster_df):
        """
        Returns: {SRS, Win_Pct, Wins}
        """
        agg_ws40, agg_usg, minutes = self.calculate_roster_metrics(roster_df)
        
        # Base SRS from skill
        # Theoretical Win% from WS
        raw_win_pct = 5.0 * agg_ws40
        
        # Apply Synergy Penalty to the Win% directly or SRS?
        # Let's apply to Win% for simplicity as efficiency loss
        syn_penalty = self.calculate_synergy(agg_usg)
        
        adj_win_pct = raw_win_pct - syn_penalty
        
        # Clamp
        adj_win_pct = max(0.05, min(0.95, adj_win_pct))
        
        # Back-calculate SRS for reporting
        # Win% = alpha * SRS + beta
        # SRS = (Win% - beta) / alpha
        predicted_srs = (adj_win_pct - self.beta_srs) / self.alpha_srs
        
        return {
            'SRS': predicted_srs,
            'Win_Pct': adj_win_pct,
            'Wins': adj_win_pct * 40.0, # 40 Game season
            'Synergy_Penalty': syn_penalty,
            'Agg_WS40': agg_ws40
        }
