# 实现报告（与 lunwen.md 对照）

## 1. 总体实现概况
本项目以“阶段约束 MDP”为主线实现球队运营决策系统，核心状态为：
`S_t = {R_t, F_t, E_t, Θ_t, K_t}`。模型实现完整覆盖竞技、财务、环境与阶段，并通过 **Action Mask + 冻结配置 K_t** 保证马尔可夫性与赛季窗口约束。

**主要模块**
- `project/mdp/`：状态/动作/阶段/掩码/转移/奖励/环境
- `project/solvers/`：MCTS + PPO + 评估工具
- `project/data/`：数据清洗、校准、玩家聚类与玩家模型
- `project/experiments/`：Q1–Q5实验与策略测试

---

## 2. 数据接入与清洗
已基于当前工作区 CSV 建立数据管道：
- 清洗脚本：`project/data/clean_and_audit.py`
- 清洗输出：`project/data/clean/*.csv`
- 数据审计：`project/data/clean/data_audit.md`
- 参数校准：`project/data/integration.py`
- 校准报告：`project/data/clean/calibration_report.md`

**当前可校准内容**
- attendance vs Win% → `rev_win_beta`
- 票价弹性/营销拉动 → `ticket_elasticity`, `marketing_attendance_beta`
- ELO→胜率逻辑回归 → `win_eta0`, `win_eta1`, `win_eta_sos`
- 市场规模 μ（Indiana vs 联盟）
- 估值 FV 与收入拆分（基于 `wnba_valuations.csv`）

---

## 3. 竞技子系统（使用真实球员数据）
新增玩家聚类模块：
- `project/data/player_kmeans.py`
- `project/experiments/player_cluster_report.py`

**功能**
- 使用真实球员技能向量（`wnba_2023_skill_vector.csv`）进行 KMeans(k=5) 聚类。
- 自动映射 cluster→位置（PG/SG/SF/PF/C）。
- 生成可用于论文的图表：
  - `figures/player_clusters_pca.png`（PCA散点图）
  - `figures/player_cluster_profiles.png`（聚类技能剖面）
- 每支球队按“5个位置 = 5个cluster”选取最优球员组成阵容。

**竞技状态映射**
- `Q_t` 由真实球员指标构造（Off/Def/Play/Reb）
- `C_t` 由 cluster 计数得到
- `P_t` 由位置计数得到
- 对手强度来自全联盟真实球员：按球队构建 5 位置阵容，计算队伍 `ELO/pace/stars` 作为对手摘要

**注意**
当前实现使用 `DWS_40`（由 `DWS/MP` 推导）作为防守信号，训练/聚类所需字段为：`Player, Team, WS/40, TS%, USG%, AST%, TRB%, DWS, MP`（脚本会导出包含 `DWS_40` 的技能文件）。
实际运行基于 `allplayers.csv`（2024 赛季）。

---

## 4. 财务子系统
财务转移严格遵循论文中“CFO 与现金储备分离”与“统一杠杆率”结构：
- `D_{t+1}=D_t+ΔD`
- `FV_{t+1}=FV_t(1+V_t)`
- `s_{t+1}=s_t(1-ω)`
- `λ_{t+1}=D/FV`

**扩展实现**
- 引入票价弹性：收入随价格按 `P^(1−e)` 变化
- 引入营销对出勤/收入的拉动
- 税线与利息约束更细

---

## 5. 动作空间细化（已落地）
原先 3–5 档动作过粗，现扩展为：
- roster: 7档（sell→buy）
- salary: 6档（floor→max_apron）
- ticket: 7档（0.85x→1.30x）
- marketing: 4档（0→10%）
- debt: 5档（-8%→+8% FV）
- equity: 5档（0→3%）

相关更新：`project/mdp/action.py`、`project/mdp/config.py`、`project/mdp/state.py`、`project/mdp/transitions_*`。

---

## 6. 求解器能力提升
MCTS 已加入 **heuristic value** 与 **候选动作贪心 rollout**：
- `project/solvers/heuristic_value.py`
- `project/solvers/mcts.py` 引入 `value_fn` 与 `rollout_candidates`

这显著降低 rollout 方差并增强长程规划能力。

---

## 7. 与 lunwen.md 对照（核心一致性）
**一致部分**
- 状态空间结构：`R/F/E/Theta` 完整实现
- Action Mask + K_t 冻结逻辑完全一致
- 财务转移与统一杠杆率一致
- 奖励函数结构与终值定义一致（在此基础上加入 win% 连续奖励）

**偏差与扩展**
- `K_t` 实现包含 roster 维（论文中只列财务/运营维）
- 奖励函数增加 win% 连续项（论文仅给 playoff 指示项）
- 票价弹性与营销拉动是经验建模（论文未明示）

---

## 8. 为什么仍可能不够“优秀”？问题在哪些部分
1. **赛程与交易细则缺失**：对手匹配与赛程强度未按真实赛程模拟，交易/选秀规则仍为简化机制。
2. **竞技转移仍存在随机噪声**（ELO/SOS 仍部分随机生成），导致策略反馈不够稳定。
3. **财务数据样本太少**（估值与收入只有少量年份，Win%→FV 关系仍被近似），估值回报不够可信。
4. **动作空间虽已细化，但策略仍离散**，真实管理中存在连续决策与非线性约束。
5. **求解器仍非最优**：MCTS 仍是随机树搜索 + 低阶启发式，PPO为线性策略，尚未达到深度策略学习能力。
6. **风险惩罚/终值权重需要经验调参**：过高会压制投资/胜率，过低会导致过度杠杆。

结论：
- **最大瓶颈在赛程/交易机制与财务稀疏数据**（真实赛程缺失 + 估值样本少）
- 次要瓶颈在竞技转移仍含较强随机性

---

## 9. 当前建议的权重与风险参数（管理者辅助模式）
- 终值权重：`terminal_weight = 2.5~3.0`
- 杠杆惩罚：`soft=2.0`, `hard=5.0~6.0`
- 软阈值：`λ_soft=0.30~0.35`，硬阈值：`λ_hard=0.55~0.60`

这能保证“稳健财务 + 合理竞技”之间的折中。

---

## 10. 后续改进路线（按优先级）
1. 基于真实赛程/SOS构建对手强度分布
2. 补齐真实选秀/自由球员/交易规则与合同期限
3. 使用更强求解器（如 MCTS+ValueNet 或 PPO+MLP）
4. 用贝叶斯或层级模型处理财务稀疏数据

---

## 运行方式
1. 生成聚类图表（需球员数据）
   - `python project/experiments/player_cluster_report.py`
2. 重新跑 Q1 测试
   - `python project/experiments/q1_leverage_policy_test.py`
3. 其余实验
   - `python project/experiments/q1_leverage_policy_map.py`
   - `python project/experiments/q2_recruitment_strategy.py`
   - `python project/experiments/q3_expansion_site_sensitivity.py`
   - `python project/experiments/q4_dynamic_ticket_or_equity.py --mode ticket`
   - `python project/experiments/q4_dynamic_ticket_or_equity.py --mode equity`
   - `python project/experiments/q5_letter_generator.py`
