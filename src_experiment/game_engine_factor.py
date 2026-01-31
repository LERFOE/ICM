import numpy as np
import pandas as pd

class FactorGameEngine:
    def __init__(self):
        # 1. 联盟基准数据 (League Averages & Std for Normalization)
        # 这些数据可以来自真实赛季统计
        self.norms = {
            'TS%':  {'mean': 0.550, 'std': 0.050},
            'USG%': {'mean': 0.200, 'std': 0.050},
            'AST%': {'mean': 0.150, 'std': 0.080},
            'TRB%': {'mean': 0.100, 'std': 0.050},
            'DBPM': {'mean': 0.000, 'std': 2.000},
            'TOV%': {'mean': 0.130, 'std': 0.030}
        }
        
        # 2. 因子权重矩阵 (Factor Loading Matrix)
        # q_i = W * z_i
        # 4 个因子: Scoring, Playmaking, Defense, Gravity
        self.weights = {
            'Scoring':    {'TS%': 0.60, 'USG%': 0.40}, 
            'Playmaking': {'AST%': 0.80, 'TOV%': -0.30}, 
            'Defense':    {'DBPM': 0.60, 'TRB%': 0.40},
            'Gravity':    {'USG%': 0.90, 'TS%': 0.10} # 纯球权吸附力，用于计算冲突
        }
        
        # 3. 胜率模型参数
        # Win% = base_win + alpha * Scoring + beta * Defense + Synergy
        self.coeffs = {
            'alpha_off': 0.35,  # 进攻权重
            'beta_def':  0.35,  # 防守权重
            'gamma_syn': 0.20   # 协同效应权重
        }

    def _z_score(self, value, metric_name):
        mu = self.norms[metric_name]['mean']
        sigma = self.norms[metric_name]['std']
        return (value - mu) / sigma

    def calculate_player_factors(self, pf):
        """
        计算单名球员的因子得分 (Vector q_i)
        pf: pandas Series (Player features)
        """
        # 1. Normalize
        z = {}
        for k in self.norms.keys():
            val = pf.get(k, self.norms[k]['mean'])
            z[k] = self._z_score(val, k)
            
        # 2. Compute Factors
        factors = {}
        for fname, w_dict in self.weights.items():
            f_val = 0.0
            for metric, w in w_dict.items():
                f_val += w * z.get(metric, 0.0)
            factors[fname] = f_val
            
        return factors

    def calculate_team_vectors(self, roster_df, strategy='balanced'):
        """
        计算球队聚合向量 (Vector Q_t)
        包含上场时间分配逻辑
        """
        # 1. 简单的上场时间分配 (基于综合能力)
        # 这里的综合能力可以粗略定义为 Scoring + Defense
        roster_df = roster_df.copy()
        
        # 临时计算评分用于分配时间
        temp_scores = []
        for _, row in roster_df.iterrows():
            f = self.calculate_player_factors(row)
            # 简单评分：攻+防
            score = f['Scoring'] + f['Defense']
            temp_scores.append(score)
        roster_df['Temp_Score'] = temp_scores
        
        # Sort desc
        sorted_roster = roster_df.sort_values('Temp_Score', ascending=False)
        
        # 分配分钟 (Min Distribution)
        # Rotation: 8-9 man
        mins = np.array([34, 32, 30, 28, 24, 20, 16, 10, 6, 0, 0, 0, 0, 0, 0])
        # Truncate or pad
        n = len(sorted_roster)
        if n > 15:
            mins = mins[:n]
        else:
            mins = mins[:n] 
            
        # Normalize to 240 (48*5)
        total_assigned = np.sum(mins)
        mins = mins * (240.0 / total_assigned)
        
        weights = mins / 240.0
        
        # 2. 加权聚合
        team_factors = {
            'Scoring': 0.0,
            'Playmaking': 0.0,
            'Defense': 0.0,
            'Gravity': 0.0
        }
        
        # 遍历主力轮换计算聚合
        for idx, (original_idx, row) in enumerate(sorted_roster.iterrows()):
            w = weights[idx]
            p_factors = self.calculate_player_factors(row)
            
            for k in team_factors.keys():
                team_factors[k] += p_factors[k] * w
                
        return team_factors, weights, sorted_roster

    def calculate_synergy(self, team_factors, roster_df, weights):
        """
        计算协同效应 (Syn_t)
        逻辑：
        1. Ball Movement Bonus: Playmaking 越高越好
        2. Diminishing Returns of Usage: Gravity 总和如果过高，会产生内耗
        """
        # 1. 组织红利
        # Playmaking Z-score 通常在 -2 到 +2 之间.
        # 如果 > 0.5 (全队平均高于联盟水平), 给予奖励
        pm_score = team_factors['Playmaking']
        bonus = 0.0
        if pm_score > 0:
            bonus = pm_score * 0.15 # 线性奖励
            
        # 2. 球权冲突惩罚
        # Gravity (主要由 Usage 构成). 
        # 联盟平均 Usage = 20%, 5人总和 = 100%. Gravity normalized approx 0.
        # 如果 Gravity >> 2.0 (说明全队都是球霸), 惩罚
        g_score = team_factors['Gravity']
        penalty = 0.0
        if g_score > 1.5:
             # 指数惩罚：球权不够分
             penalty = (g_score - 1.5) ** 1.5 * 0.5
             
        synergy = bonus - penalty
        return synergy

    def predict_performance(self, roster_df):
        """
        核心预测 API
        Returns: {SRS, Win_Pct, Synergy_Score, Details}
        """
        # 1. 球队因子聚合
        team_factors, weights, sorted_roster = self.calculate_team_vectors(roster_df)
        
        # 2. 协同计算
        syn_score = self.calculate_synergy(team_factors, sorted_roster, weights)
        
        # 3. 胜率合成
        # 基准各因子贡献
        raw_strength = (
            self.coeffs['alpha_off'] * team_factors['Scoring'] + 
            self.coeffs['beta_def'] * team_factors['Defense']
        )
        
        # 最终强度 = 纸面实力 + 协同
        total_strength = raw_strength + self.coeffs['gamma_syn'] * syn_score
        
        # 4. 映射到胜率
        # Z-score sum to Win%
        # 假设 total_strength ~ N(0, 1) roughly (actually summation var is smaller)
        # Logic: +1.0 strength => ~65% win rate (+1 sigma)
        #        +2.0 strength => ~80% win rate
        #        -2.0 strength => ~20% win rate
        # Simple Logistic curve
        win_pct = 1.0 / (1.0 + np.exp(-1.5 * total_strength))
        
        # Map to SRS (Simple Rating System)
        # SRS approx (Win% - 0.5) * 35 for NBA?
        # NBA SRS range -10 to +10 roughly.
        srs = (win_pct - 0.5) * 30.0
        
        return {
            'Win_Pct': win_pct,
            'SRS': srs,
            'Synergy': syn_score,
            'Factors': team_factors
        }
