import numpy as np
import pandas as pd
import random

class FactorPlayerGenerator:
    def __init__(self):
        # 基础原型参数: (Mean, Std)
        # 对应 src_experiment/game_engine_factor.py 中的 norms
        # TS%, USG%, AST%, TRB%, DBPM, TOV%
        pass # Defined inside methods for clarity

    def generate_player(self, archetype='Random', quality='Rotation'):
        """
        生成除了 WS/40 之外的丰富数据
        """
        if archetype == 'Random':
            archetype = random.choice(['Guard', 'Wing', 'Big'])
            
        # 1. 质量基准 (Quality Tier) -> 决定了总体能力值的高低
        tier_bonus = 0.0
        if quality == 'Superstar': tier_bonus = 1.5
        elif quality == 'Star': tier_bonus = 1.0
        elif quality == 'Starter': tier_bonus = 0.5
        elif quality == 'Rotation': tier_bonus = 0.0
        elif quality == 'Bench': tier_bonus = -0.5
        
        # 2. 原型模板 (Archetype Templates)
        # 用 Z-score 概念定义偏移量 (Base 0 = League Avg)
        
        # Guard: High AST, High USG, Low TRB
        if archetype == 'Guard':
            base_stats = {
                'TS_Z':  0.0, 'USG_Z': 0.5, 'AST_Z': 1.5,
                'TRB_Z': -1.0, 'DBPM_Z': -0.5, 'TOV_Z': 0.5 # High TOV is bad (High Z for TOV)
            }
        # Wing: Balanced
        elif archetype == 'Wing':
            base_stats = {
                'TS_Z':  0.2, 'USG_Z': 0.2, 'AST_Z': 0.0,
                'TRB_Z': -0.2, 'DBPM_Z': 0.5, 'TOV_Z': 0.0
            }
        # Big: High TRB, High DBPM, Low AST
        elif archetype == 'Big':
            base_stats = {
                'TS_Z':  0.5, 'USG_Z': -0.2, 'AST_Z': -1.0,
                'TRB_Z': 1.5, 'DBPM_Z': 1.0, 'TOV_Z': -0.2
            }
            
        # 3. 生成具体数值 (Meta-Stats)
        metrics = {}
        # Mapping back to real values using Engine Norms
        # Copying norms from engine manually for generation
        norms = {
            'TS%':  {'mean': 0.550, 'std': 0.050},
            'USG%': {'mean': 0.200, 'std': 0.050},
            'AST%': {'mean': 0.150, 'std': 0.080},
            'TRB%': {'mean': 0.100, 'std': 0.050},
            'DBPM': {'mean': 0.000, 'std': 2.000},
            'TOV%': {'mean': 0.130, 'std': 0.030}
        }
        
        for k in norms.keys():
            # Key for Z-score dictionary
            z_key = k.replace('%', '') + '_Z' # e.g. TS_Z
            if k == 'DBPM': z_key = 'DBPM_Z'
            
            base_z = base_stats.get(z_key, 0.0)
            
            # Application of Tier Bonus
            # Superstars are good at almost everything (Efficiency, Impact)
            # But specific roles might not scale linearly (e.g. usage)
            
            final_z = base_z + np.random.normal(0, 0.5) # Random Noise
            
            # Tier bonus primarily affects Efficiency (TS), Impact (DBPM), Playmaking (AST)
            if k in ['TS%', 'AST%', 'DBPM', 'TRB%']:
                final_z += tier_bonus
            elif k == 'USG%':
                # Stars have higher usage generally
                final_z += tier_bonus * 0.5
            elif k == 'TOV%':
                # Better players might have lower TOV relative to usage, 
                # but let's keep it noisy.
                pass
                
            # Convert Z back to Value
            val = norms[k]['mean'] + final_z * norms[k]['std']
            
            # Clamping logical bounds
            if '%' in k:
                val = max(0.01, min(0.99, val))
                
            metrics[k] = val
            
        # 4. create Data Row
        row = {
            'Name': f"{quality} {archetype} {random.randint(100,999)}",
            'Archetype': archetype,
            'Quality': quality,
            'TS%': metrics['TS%'],
            'USG%': metrics['USG%'],
            'AST%': metrics['AST%'],
            'TRB%': metrics['TRB%'],
            'DBPM': metrics['DBPM'],
            'TOV%': metrics['TOV%'],
            'Salary': 30.0 if quality=='Superstar' else (20.0 if quality=='Star' else 5.0)
        }
        return row
    
    def generate_roster(self, distribution):
        """
        distribution: dict, e.g. {'Superstar': 1, 'Starter': 4, ...}
        """
        roster = []
        for qual, count in distribution.items():
            for _ in range(count):
                roster.append(self.generate_player(quality=qual))
        
        return pd.DataFrame(roster)
