# 论文图表报告（newfigures）

本目录中的图表主要由 `project/experiments/generate_paper_figures.py` 生成；Q4 股权激励相关图由 `project/experiments/q4_equity_mckp.py` 生成。图表命名与用途如下。

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

### 10) `q3_policy_comparison_deltas.png`
- **类型**：对比柱状图（相对基线的增量）
- **用途**：在扩军情景下，对比 PPO / Defensive / Aggressive 相对 Baseline 的终值与累计现金流增量（Season 3）
- **数据来源**：`project/experiments/output/q3_policy_comparison_summary.csv`
- **解读**：用于“拿数据说话”说明：扩军年高薪抢星/高杠杆路径的净收益并不占优，而去杠杆+稳定策略更稳健。

### 11) `q3_offseason_action_debt.png`
- **类型**：热力图（Action Heatmap）
- **用途**：展示扩军年 Offseason 不同策略的债务动作均值（证据化“降低杠杆冲动”）
- **数据来源**：`project/experiments/output/q3_policy_comparison_detail.csv`
- **解读**：颜色越深代表越倾向“加杠杆”，可直接对应论文中 $a_{debt}$ 的策略差异。

### 12) `q3_offseason_action_salary.png`
- **类型**：热力图（Action Heatmap）
- **用途**：展示扩军年 Offseason 不同策略的薪资动作均值（证据化“高薪抢星 vs 稳定薪资”）
- **数据来源**：`project/experiments/output/q3_policy_comparison_detail.csv`
- **解读**：用于说明扩军年竞价上升后，高薪档位的选择频率与财务表现之间的权衡。

### 13) `q3_league_impact_heatmap.png`
- **类型**：相关性/热力图（Impact Heatmap）
- **用途**：量化扩军选址对联盟所有球队的影响（正负影响一目了然）
- **数据来源**：`project/experiments/output/q3_league_impact_allteams.csv`
- **解读**：红色=不利，绿色=有利；用于筛选“最有利/最不利”球队所有者。

---

## Q4 额外业务决策（股权激励 / MCKP）

### 14) `q4_skill_violin.png`
- **类型**：小提琴图（分布对比）
- **用途**：对比 Indiana 与联盟球员的 skill\_score 分布，用于确定星级门槛。
- **数据来源**：`allplayers.csv` + `project/experiments/q4_equity_mckp.py`
- **解读**：显示 Indiana 在联盟分布中的相对位置与门槛区间。

### 15) `q4_trifactor_bars.png`
- **类型**：堆叠柱状图
- **用途**：展示 Tri-Factor 三元驱动（Selection / Competitive / Financial）的贡献结构。
- **数据来源**：`project/experiments/output/q4_equity_mckp/q4_equity_options.csv`
- **解读**：用于解释“为何特定球员被选中”。

### 16) `q4_equity_option_heatmap.png`
- **类型**：热力图（所有者终值）
- **用途**：每名球员 × 不同股权档位的所有者终值变化对比。
- **数据来源**：`project/experiments/output/q4_equity_mckp/q4_equity_options.csv`
- **解读**：颜色越高代表对所有者终值越有利；可观察稀释风险。

### 16.1) `q4_equity_option_win_heatmap.png`
- **类型**：热力图（胜率增益）
- **用途**：展示股权比例带来的胜率增益“甜点区间”。
- **数据来源**：`project/experiments/output/q4_equity_mckp/q4_equity_options.csv`
- **解读**：可见中等股权比例效果最好，过高反而减弱。

### 17) `q4_equity_allocation_lollipop.png`
- **类型**：棒棒糖图（方案输出）
- **用途**：展示 MCKP 输出的最优股权分配。
- **数据来源**：`project/experiments/output/q4_equity_mckp/q4_mckp_solution.csv`
- **解读**：直观看到“给谁 + 给多少”。

### 18) `q4_equity_frontier.png`
- **类型**：折线图（Frontier）
- **用途**：股权上限对 OwnerValue / Win% / CF 的边际影响。
- **数据来源**：`project/experiments/output/q4_equity_mckp/q4_equity_frontier.csv`
- **解读**：用于量化“股权上限”决策。

### 19) `q4_sensitivity_heatmap.png`
- **类型**：敏感性热力图
- **用途**：Tri-Factor 权重扰动下的 OwnerValue 变化。
- **数据来源**：`project/experiments/output/q4_equity_mckp/q4_sensitivity_weights.csv`
- **解读**：证明策略对权重扰动的稳健性。

### 20) `q4_radar_comparison.png`
- **类型**：雷达图
- **用途**：Baseline vs Equity-MCKP 的多指标对比（Win / CF / Owner / Equity）。
- **数据来源**：`project/experiments/output/q4_equity_mckp/q4_equity_frontier.csv`
- **解读**：综合评价股权策略的优势。

> 备注：旧版 PPO 动作频次图 `q4_ticket_policy_hist.png` / `q4_equity_policy_hist.png` 保留为过程记录，但不再是 Q4 主报告图。

---

## 数据结构与多维分布（补充支撑图）

### 21) `player_corr_heatmap.png`
- **类型**：相关性热力图
- **用途**：展示球员指标间相关性（TS%、USG%、AST%、DWS 等）
- **数据来源**：`allplayers.csv`
- **解读**：用于说明特征多样性与变量冗余。

### 22) `player_cluster_parallel_coordinates.png`
- **类型**：平行坐标图
- **用途**：展示 5 类球员聚类的技能特征差异
- **数据来源**：`allplayers.csv` + `project/data/player_kmeans.py`
- **解读**：直观显示“PG/SG/SF/PF/C”类群的技能区分。

### 23) `attendance_streamgraph.png`
- **类型**：堆叠面积图 / Streamgraph
- **用途**：展示联盟顶级球队 attendance 随时间变化
- **数据来源**：`wnba_attendance.csv`
- **解读**：支持“市场环境变化”与票价策略分析。

### 24) `market_bubble_attendance_revenue.png`
- **类型**：气泡图
- **用途**：展示市场规模（出勤）、收入与估值的关系
- **数据来源**：`wnba_attendance.csv` + `wnba_valuations.csv`
- **解读**：可用于财务校准假设和收入-估值关系说明。

---

## 回归拟合权重与敏感性分析（新增）

### 25) `skill_weights_bar.png`
- **类型**：权重柱状图
- **用途**：展示 Ridge/Lasso 拟合得到的 skill\_score 权重方向与大小
- **数据来源**：`project/data/skill_weights.json`
- **解读**：正负号反映指标与目标变量的相关方向；柱高体现重要性。

### 26) `skill_model_mse.png`
- **类型**：模型对比柱状图
- **用途**：比较 Ridge 与 Lasso 的拟合误差
- **数据来源**：`project/data/skill_weights.json`
- **解读**：MSE 越低代表拟合更稳定，本次选用 Ridge。

### 27) `skill_fit_scatter.png`
- **类型**：拟合散点图
- **用途**：展示 Ridge 模型对 Win%、NetRtg、ELO\_proxy 的拟合效果（预测 vs 实际）
- **数据来源**：`allplayers.csv` + `wnba_advanced_stats.csv` + `IND_ELO_O_SOS_season_level.csv`
- **解读**：点越靠近对角线，拟合越好。

### 28) `skill_sensitivity.png`
- **类型**：直方图 + 小提琴图组合
- **用途**：展示权重扰动下的稳定性（Spearman 相关分布 + Top5 重叠率分布）
- **数据来源**：`allplayers.csv`（权重扰动 200 次）
- **解读**：反映招募策略对权重变化的敏感性，Trade 区间波动最大。

---

## 生成方式
运行以下命令可重新生成全部图表：
```
python project/experiments/generate_paper_figures.py
```

> 注：图表使用英文标签以避免字体缺失问题，适合直接放入论文。
