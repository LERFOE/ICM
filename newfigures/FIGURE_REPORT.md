# 论文图表报告（newfigures）

本目录中的图表全部由 `project/experiments/generate_paper_figures.py` 生成，服务于 Q1–Q4 的模型构建与评估。图表命名与用途如下。

---

## Q1 杠杆与动态决策（Policy / Evaluation）

### 1) `q1_policy_heatmap_macro0.png` / `q1_policy_heatmap_macro1.png` / `q1_policy_heatmap_macro2.png`
- **类型**：相关性/策略热力图
- **用途**：展示在不同宏观状态下，杠杆政策如何随“现金流/杠杆率”变化
- **数据来源**：`project/experiments/output/q1_leverage_policy_map.csv`
- **解读**：颜色越“高”代表更倾向加杠杆。可用于强调“宏观衰退期去杠杆、繁荣期适度融资”的策略逻辑。

### 2) `q1_violin_win_cash.png`
- **类型**：小提琴图（分布对比）
- **用途**：比较“静态策略 vs 模型策略”在 1–3 赛季的胜率与现金流分布
- **数据来源**：`project/experiments/output/q1_leverage_policy_test_detail.csv`
- **解读**：显示模型策略的分布偏移与方差，强调风险与收益的权衡。

### 3) `q1_radar_season3.png`
- **类型**：雷达图（多指标综合评价）
- **用途**：将胜率/现金流/估值/终值/杠杆/债务/协同等指标统一对比
- **数据来源**：`project/experiments/output/q1_leverage_policy_test_summary.csv`
- **解读**：用于总结“模型 vs 基线策略”综合优劣。

### 4) `q1_roc_win_prediction.png`
- **类型**：ROC 曲线
- **用途**：评估基于 ELO 的胜负预测能力
- **数据来源**：`IND_ELO_O_SOS_game_level.csv`
- **解读**：AUC 越高代表模型胜负判别能力越好。

### 5) `q1_ridgeline_elo.png`
- **类型**：山脊图 / Ridgeline
- **用途**：展示 Indiana 不同赛季 ELO 分布的动态变化
- **数据来源**：`IND_ELO_O_SOS_game_level.csv`
- **解读**：可用于强调长期强度变化与赛季波动。

---

## Q2 招募策略（Draft / FA / Trade）

### 6) `q2_sankey_recruitment.png`
- **类型**：桑基图（流动与过程）
- **用途**：展示不同招募渠道（Draft/FA/Trade）到价值分层的“流向”
- **数据来源**：`project/experiments/output/q2_recruitment_strategy.csv`
- **解读**：可直观看出哪个渠道更容易产出高价值候选。

### 7) `q2_bubble_candidates.png`
- **类型**：气泡图（多维对比）
- **用途**：展示候选球员在“ΔQ（竞技增益）- Owner价值”上的分布，点的大小代表成本
- **数据来源**：`project/experiments/output/q2_recruitment_strategy.csv`
- **解读**：强调“高增益-低成本”的优先目标，以及 Draft/FA/Trade 三类候选差异。

---

## Q3 扩军情景（Site Sensitivity & Winners/Losers）

### 8) `q3_expansion_bar.png`
- **类型**：对比柱状图
- **用途**：比较不同扩军选址对 Indiana 的终值/现金流影响
- **数据来源**：`project/experiments/output/q3_expansion_sensitivity.csv`
- **解读**：用于说明“扩军选址是正向还是负向冲击”。

### 9) `q3_chord_expansion.png`
- **类型**：和弦图风格网络图
- **用途**：展示扩军选址对联盟“受益/受损球队”的关联
- **数据来源**：`project/experiments/output/q3_expansion_sensitivity.csv` + `wnba_attendance.csv` + player-based strength
- **解读**：绿线=受益，红线=受损，强调扩军外溢效应。

---

## Q4 额外业务决策（票价/股权）

### 10) `q4_ticket_policy_hist.png`
- **类型**：动作频次柱状图
- **用途**：展示票价策略在训练后的选择频率
- **数据来源**：PPO 训练（`project/experiments/q4_dynamic_ticket_or_equity.py`）
- **解读**：反映模型偏好的票价区间。

### 11) `q4_equity_policy_hist.png`
- **类型**：动作频次柱状图
- **用途**：展示股权激励策略的选择频率
- **数据来源**：PPO 训练（`project/experiments/q4_dynamic_ticket_or_equity.py`）
- **解读**：用于阐释“股权稀释的谨慎程度”。

---

## 数据结构与多维分布（补充支撑图）

### 12) `player_corr_heatmap.png`
- **类型**：相关性热力图
- **用途**：展示球员指标间相关性（TS%、USG%、AST%、DWS 等）
- **数据来源**：`allplayers.csv`
- **解读**：用于说明特征多样性与变量冗余。

### 13) `player_cluster_parallel_coordinates.png`
- **类型**：平行坐标图
- **用途**：展示 5 类球员聚类的技能特征差异
- **数据来源**：`allplayers.csv` + `project/data/player_kmeans.py`
- **解读**：直观显示“PG/SG/SF/PF/C”类群的技能区分。

### 14) `attendance_streamgraph.png`
- **类型**：堆叠面积图 / Streamgraph
- **用途**：展示联盟顶级球队 attendance 随时间变化
- **数据来源**：`wnba_attendance.csv`
- **解读**：支持“市场环境变化”与票价策略分析。

### 15) `market_bubble_attendance_revenue.png`
- **类型**：气泡图
- **用途**：展示市场规模（出勤）、收入与估值的关系
- **数据来源**：`wnba_attendance.csv` + `wnba_valuations.csv`
- **解读**：可用于财务校准假设和收入-估值关系说明。

---

## 生成方式
运行以下命令可重新生成全部图表：
```
python project/experiments/generate_paper_figures.py
```

> 注：图表使用英文标签以避免字体缺失问题，适合直接放入论文。
