import unittest
import pandas as pd
import numpy as np
from game_engine_factor import FactorGameEngine
from player_generator_factor import FactorPlayerGenerator

class TestFactorEngine(unittest.TestCase):
    def setUp(self):
        self.engine = FactorGameEngine()
        self.gen = FactorPlayerGenerator()
        
    def test_superteam_dominance(self):
        """Test that a team of Superstars beats a team of Scrubs."""
        print("\n[Test] Superteam vs Scrubteam")
        
        # 1. Superteam
        df_super = self.gen.generate_roster({'Superstar': 5, 'Star': 3, 'Rotation': 4})
        perf_super = self.engine.predict_performance(df_super)
        print(f"Superteam Win%: {perf_super['Win_Pct']:.3f} | SRS: {perf_super['SRS']:.2f}")
        print(f"Factors: {perf_super['Factors']}")
        
        # 2. Scrubteam
        df_scrub = self.gen.generate_roster({'Bench': 5, 'Rotation': 3, 'Bench': 4}) # Mostly bench
        perf_scrub = self.engine.predict_performance(df_scrub)
        print(f"Scrubteam Win%: {perf_scrub['Win_Pct']:.3f} | SRS: {perf_scrub['SRS']:.2f}")
        
        self.assertGreater(perf_super['Win_Pct'], perf_scrub['Win_Pct'])
        self.assertGreater(perf_super['Win_Pct'], 0.70, "Superteam should be dominant")

    def test_synergy_ballhog_penalty(self):
        """Test that adding too many high-usage players hurts Synergy."""
        print("\n[Test] Synergy Logic: Ballhog Penalty")
        
        # 1. Balanced Team (1 Star, 4 Role Players)
        # Create a DataFrame explicitly to control USG
        balanced_roster = []
        # Main Guy: High USG (35%)
        balanced_roster.append(self._create_specific_player(usg=0.35, ts=0.60, ast=0.30, name="Luka"))
        # Role Players: Low USG (15%)
        for i in range(7):
            balanced_roster.append(self._create_specific_player(usg=0.15, ts=0.55, ast=0.10, name=f"Role{i}"))
            
        df_bal = pd.DataFrame(balanced_roster)
        perf_bal = self.engine.predict_performance(df_bal)
        syn_bal = perf_bal['Synergy']
        
        # 2. Ballhog Team (5 Stars, all want ball)
        # All High USG (30%)
        hog_roster = []
        for i in range(8):
            hog_roster.append(self._create_specific_player(usg=0.30, ts=0.60, ast=0.20, name=f"Hog{i}")) # Low AST relative to USG
            
        df_hog = pd.DataFrame(hog_roster)
        perf_hog = self.engine.predict_performance(df_hog)
        syn_hog = perf_hog['Synergy']
        
        print(f"Balanced Team Synergy: {syn_bal:.3f}")
        print(f"Ballhog Team Synergy: {syn_hog:.3f}")
        
        # Balanced team should have better synergy (higher score)
        self.assertGreater(syn_bal, syn_hog, "Ballhog team should suffer synergy penalty")

    def test_playmaking_bonus(self):
        """Test that high AST% improves Synergy."""
        print("\n[Test] Synergy Logic: Playmaking Bonus")
        
        # 1. Low Playmaking Team
        low_ast_roster = [self._create_specific_player(usg=0.20, ast=0.05) for _ in range(8)]
        perf_low = self.engine.predict_performance(pd.DataFrame(low_ast_roster))
        
        # 2. High Playmaking Team
        high_ast_roster = [self._create_specific_player(usg=0.20, ast=0.30) for _ in range(8)]
        perf_high = self.engine.predict_performance(pd.DataFrame(high_ast_roster))
        
        print(f"Low AST Synergy: {perf_low['Synergy']:.3f}")
        print(f"High AST Synergy: {perf_high['Synergy']:.3f}")
        
        self.assertGreater(perf_high['Synergy'], perf_low['Synergy'])

    def _create_specific_player(self, usg, ast, ts=0.55, trb=0.10, dbpm=0.0, name="Player"):
        """Helper to create raw row."""
        return {
            'Name': name,
            'TS%': ts,
            'USG%': usg,
            'AST%': ast,
            'TRB%': trb,
            'DBPM': dbpm,
            'TOV%': 0.13,
            'Salary': 10.0
        }

if __name__ == '__main__':
    unittest.main()
