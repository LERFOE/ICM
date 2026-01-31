import numpy as np
import pandas as pd
import random

class Player:
    def __init__(self, name, pos, age, salary, contract_years, metrics):
        self.name = name
        self.pos = pos
        self.age = age
        self.salary = salary # In Millions
        self.contract_years = contract_years
        
        # Core Metrics as per user request
        self.ws_40 = metrics.get('ws_40', 0.100) # League avg approx 0.100
        self.ts_pct = metrics.get('ts_pct', 0.530)
        self.usg_pct = metrics.get('usg_pct', 0.200)
        self.ast_pct = metrics.get('ast_pct', 0.150)
        self.trb_pct = metrics.get('trb_pct', 0.100)
        self.dbpm = metrics.get('dbpm', -0.5)
        
        # Derived for simulation
        self.morale = 1.0
        self.fatigue = 0.0

    def to_dict(self):
        return {
            'Name': self.name,
            'Pos': self.pos,
            'Age': self.age,
            'Salary_M': self.salary,
            'Contract_Yrs': self.contract_years,
            'WS_40': self.ws_40,
            'TS%': self.ts_pct,
            'USG%': self.usg_pct,
            'AST%': self.ast_pct,
            'TRB%': self.trb_pct,
            'DBPM': self.dbpm
        }

def generate_synthetic_database(num_teams=12, players_per_team=12):
    """
    Generates a synthetic WNBA player database roughly aligned with 2023 distributions.
    """
    positions = ['G', 'F', 'C']
    roster_types = ['Superstar', 'Star', 'Starter', 'Rotation', 'Bench', 'Deep Bench']
    
    players = []
    
    # Archetype Parameters (Mean, StdDev for WS/40 and USG%)
    archetypes = {
        'Superstar': {'ws_40': (0.250, 0.05), 'usg': (0.28, 0.04), 'sal': (0.20, 0.02)}, # 200k+
        'Star':      {'ws_40': (0.180, 0.03), 'usg': (0.24, 0.03), 'sal': (0.18, 0.02)}, 
        'Starter':   {'ws_40': (0.120, 0.03), 'usg': (0.18, 0.03), 'sal': (0.12, 0.03)},
        'Rotation':  {'ws_40': (0.080, 0.02), 'usg': (0.16, 0.02), 'sal': (0.08, 0.01)},
        'Bench':     {'ws_40': (0.040, 0.02), 'usg': (0.14, 0.02), 'sal': (0.07, 0.005)},
        'Deep Bench':{'ws_40': (0.010, 0.02), 'usg': (0.12, 0.02), 'sal': (0.06, 0.005)}
    }

    # WNBA Salary Hard Cap is approx 1.4M? Individual max ~240k. 
    # Let's normalize Salary to Millions. Max = 0.24M. Min = 0.064M.
    
    for team_id in range(num_teams):
        # Enforce roughly 1 superstar per 2 teams, etc.
        template = ['Starter']*4 + ['Rotation']*4 + ['Bench']*4
        if team_id % 2 == 0:
            template[0] = 'Superstar'
        else:
            template[0] = 'Star'
            
        for i, p_type in enumerate(template):
            stats = archetypes[p_type]
            
            # Generate Metrics
            ws_40 = np.random.normal(stats['ws_40'][0], stats['ws_40'][1])
            usg = np.random.normal(stats['usg'][0], stats['usg'][1])
            salary = np.clip(np.random.normal(stats['sal'][0], stats['sal'][1]), 0.064, 0.245)
            
            pos = random.choice(positions)
            
            # Position bias
            ast_bias = 0.1 if pos == 'G' else 0.0
            trb_bias = 0.1 if pos == 'C' else 0.0
            
            metrics = {
                'ws_40': ws_40,
                'ts_pct': np.random.normal(0.54, 0.04),
                'usg_pct': usg,
                'ast_pct': np.clip(np.random.normal(0.15 + ast_bias, 0.05), 0, 1),
                'trb_pct': np.clip(np.random.normal(0.10 + trb_bias, 0.05), 0, 1),
                'dbpm': np.random.normal(0, 1.5)
            }
            
            p = Player(
                name=f"Player_{team_id}_{i}",
                pos=pos,
                age=random.randint(21, 36),
                salary=round(salary, 3),
                contract_years=random.randint(1, 3),
                metrics=metrics
            )
            players.append(p)
            
    return pd.DataFrame([p.to_dict() for p in players])

if __name__ == "__main__":
    df = generate_synthetic_database()
    print(df.head())
    print(df.groupby('Pos').mean(numeric_only=True))
