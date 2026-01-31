% =========================
% 最终版：竞技资产状态空间 \mathcal{R}_t（按“10个核心变量”精简版）
% =========================
\begin{comment}
    

\section{最终版竞技资产状态空间 $\mathcal{R}_t$}

\subsection{设计原则}
\begin{itemize}
    \item \textbf{可求解性优先}：用离散分桶/低维向量刻画竞技状态，避免球员逐项指标导致维数灾难。
    \item \textbf{决策相关性}：仅保留对“建队/交易/续约/轮换/杠杆承受能力”有直接影响的竞技变量。
    \item \textbf{可从公开数据映射}：每个变量给出明确的计算方式，保证可由赛季数据或滚动窗口统计得到。
\end{itemize}

\subsection{竞技状态向量的总体形式}

令球队在时刻 $t$ 的竞技资产状态为一个由 $10$ 个核心变量组成的向量：
\begin{equation}
\mathcal{R}_t
=
\Big\{
\mathbf{Q}_t,\;
\mathbf{C}_t,\;
\mathbf{P}_t,\;
\mathbf{L}_t,\;
\mathbf{A}_t,\;
\mathbf{W}_t,\;
\text{ELO}_t,\;
\text{Syn}_t,\;
\mathbf{O}_t,\;
\text{SOS}_t
\Big\}.
\end{equation}

其中前 $1$--$5$ 项刻画球员资产结构（可控内生），第 $6$--$8$ 项刻画球队竞技表现与体系（半内生），第 $9$--$10$ 项刻画赛程对抗环境（外生）。

% -------------------------
% 1) 球员技术能力：低维综合向量
% -------------------------
\subsubsection{1. 球员技术能力聚合向量 $\mathbf{Q}_t$（Team Skill Aggregate）}

为避免逐球员逐指标带来高维状态，先定义每名球员 $i$ 的\textbf{标准化特征}：
\begin{equation}
z_{i,k}
=
\frac{x_{i,k}-\mu_k}{\sigma_k},
\qquad k \in \mathcal{K},
\end{equation}
其中 $x_{i,k}$ 为球员指标（如 TS\%、AST、TOV、DRTG、BPM 等），$\mu_k,\sigma_k$ 为同赛季联盟均值与标准差。

将球员指标压缩为 $d$ 维\textbf{能力因子}（建议 $d=4\sim 6$）：
\begin{equation}
\mathbf{q}_i
=
\mathbf{W}\mathbf{z}_i
\in \mathbb{R}^{d},
\end{equation}
其中 $\mathbf{z}_i=[z_{i,k}]_{k\in\mathcal{K}}$，$\mathbf{W}$ 可由 PCA/因子分析得到，或由专家权重设定。

球队层面的能力聚合向量定义为\textbf{上场时间加权平均}：
\begin{equation}
\mathbf{Q}_t
=
\sum_{i\in \mathcal{R}(t)}
\omega_{i,t}\,\mathbf{q}_i,
\qquad
\omega_{i,t}=\frac{\text{MP}_{i,t}}{\sum_{j\in \mathcal{R}(t)} \text{MP}_{j,t}},
\end{equation}
其中 $\mathcal{R}(t)$ 为当期大名单集合，$\text{MP}_{i,t}$ 为球员在滚动窗口（如最近 $g$ 场）内平均上场时间。

\textbf{解释（示例）}：可取
$\mathbf{q}_i=[q_{i,\text{Off}},q_{i,\text{Def}},q_{i,\text{Play}},q_{i,\text{Reb}}]$
分别代表进攻/防守/组织/篮板四个综合因子。

% -------------------------
% 2) 球员类型分布：聚类标签的直方图
% -------------------------
\subsubsection{2. 球员类型聚类分布 $\mathbf{C}_t$（Roster Archetype Mix）}

对联盟球员做聚类得到类型标签 $c_i\in\{1,\dots,K\}$（建议 $K=6\sim 10$）。定义球队类型构成向量为：
\begin{equation}
\mathbf{C}_t
=
[c_{t,1},\dots,c_{t,K}],
\qquad
c_{t,k}=\sum_{i\in \mathcal{R}(t)} \mathbb{I}(c_i=k).
\end{equation}

该向量刻画“阵容原型结构”（例如：空间型射手、护框中锋、持球核心、3D 侧翼等），用于约束阵容搭配与交易方向。

% -------------------------
% 3) 位置平衡：人数向量 + 约束
% -------------------------
\subsubsection{3. 位置平衡约束向量 $\mathbf{P}_t$（Positional Balance）}

\begin{equation}
\mathbf{P}_t
=
[p_{\text{PG}},p_{\text{SG}},p_{\text{SF}},p_{\text{PF}},p_{\text{C}}],
\qquad
p_{\text{pos}}
=\sum_{i\in \mathcal{R}(t)}\mathbb{I}(\text{pos}_i=\text{pos}).
\end{equation}

\textbf{基本约束}：
\begin{equation}
\sum_{\text{pos}} p_{\text{pos}}=N_{\text{roster}},
\qquad
p_{\text{pos}}\ge 1,\ \forall \text{pos}.
\end{equation}

（若考虑位置模糊，可用“主位置+副位置”将 $\mathbb{I}(\cdot)$ 改为软计数权重。）

% -------------------------
% 4) 合同状态：到期结构的低维化
% -------------------------
\subsubsection{4. 合同状态向量 $\mathbf{L}_t$（Contract Maturity Profile）}

为避免逐球员合同年限带来高维，采用\textbf{到期结构分桶}表示：
\begin{equation}
\mathbf{L}_t
=
[\ell_{0,t},\ell_{1,t},\ell_{2+,t}],
\end{equation}
其中
\begin{equation}
\ell_{0,t}=\sum_{i}\mathbb{I}(l_i=0),\quad
\ell_{1,t}=\sum_{i}\mathbb{I}(l_i=1),\quad
\ell_{2+,t}=\sum_{i}\mathbb{I}(l_i\ge 2),
\end{equation}
$l_i$ 为球员 $i$ 剩余合同年限（年）。

\textbf{可选增强}（若你希望更精确且仍可控）：加入“保障/非保障/选项”比例，但仍保持分桶维度常数。

% -------------------------
% 5) 年龄曲线：年龄结构的低维表示
% -------------------------
\subsubsection{5. 年龄结构向量 $\mathbf{A}_t$（Age Curve Profile）}

采用年龄分布的低维表示：
\begin{equation}
\mathbf{A}_t
=
[a_{\text{mean},t},\ a_{\text{var},t},\ a_{28+,t}],
\end{equation}
其中
\begin{equation}
a_{\text{mean},t}=\frac{1}{N_{\text{roster}}}\sum_{i} a_i,\quad
a_{\text{var},t}=\frac{1}{N_{\text{roster}}}\sum_{i}(a_i-a_{\text{mean},t})^2,\quad
a_{28+,t}=\sum_{i}\mathbb{I}(a_i\ge 28).
\end{equation}

该向量用于刻画“即战力 vs 衰退风险”的结构特征，并可在状态转移中引入年龄衰退函数。

% -------------------------
% 6) 战绩状态：滚动窗口战绩（分桶）
% -------------------------
\subsubsection{6. 当前赛季战绩状态 $\mathbf{W}_t$（Performance Record State）}

为使状态平稳且可比较，使用滚动窗口（最近 $g$ 场）定义：
\begin{equation}
\mathbf{W}_t
=
[\text{Win\%}_t^{(g)},\ \text{Streak}_t,\ \text{RankBin}_t],
\end{equation}
其中 $\text{Win\%}_t^{(g)}$ 为最近 $g$ 场胜率，$\text{Streak}_t$ 为连胜/连败长度，
$\text{RankBin}_t$ 为排名分桶（如：前 1/3、中 1/3、后 1/3）。

% -------------------------
% 7) 战力评级：ELO（或 SRS）单指标
% -------------------------
\subsubsection{7. 球队战力评级 $\text{ELO}_t$（Team Strength Rating）}

采用 ELO 作为单一强度指标：
\begin{equation}
\text{ELO}_{t+1}
=
\text{ELO}_t
+
K_{\text{elo}}\left(W_{\text{actual},t}-W_{\text{expected},t}\right),
\end{equation}
其中 $W_{\text{expected},t}$ 可由 ELO 差异的 logistic 期望胜率给出。

% -------------------------
% 8) 阵容协同：量化“1+1>2”
% -------------------------
\subsubsection{8. 阵容协同效应 $\text{Syn}_t$（Roster Synergy）}

用“实际阵容净效率”减去“按球员加总的基准净效率”定义协同增益：
\begin{equation}
\text{Syn}_t
=
\underbrace{\text{NetRTG}_t^{(g)}}_{\text{阵容整体表现}}
-
\underbrace{\sum_{i\in \mathcal{R}(t)}\omega_{i,t}\,\text{BPM}_{i,t}}_{\text{个体贡献线性叠加基准}},
\end{equation}
其中
$\text{NetRTG}_t^{(g)}=\text{ORTG}_t^{(g)}-\text{DRTG}_t^{(g)}$
为最近 $g$ 场的净效率，$\text{BPM}_{i,t}$ 为球员 box plus-minus（同样可用滚动窗口），
$\omega_{i,t}$ 为上场时间权重。

\textbf{解释}：$\text{Syn}_t>0$ 表示体系加成（“1+1>2”），$\text{Syn}_t<0$ 表示风格冲突或轮换失配。

% -------------------------
% 9) 对手强度分布：剩余赛程 ELO 列表（可再压缩）
% -------------------------
\subsubsection{9. 对手强度分布 $\tilde{\mathbf{O}}_t$（Opponent Strength Statistics）}

原始赛程长度 $m$ 为变量，为保证状态空间维数固定，定义统计量向量：
\begin{equation}
\tilde{\mathbf{O}}_t = [\mu_{\text{opp}}, \sigma_{\text{opp}}, \text{N}_{\text{elite}}]
\end{equation}
其中 $\mu_{\text{opp}}$ 为剩余对手平均 ELO，$\sigma_{\text{opp}}$ 为标准差，$\text{N}_{\text{elite}}$ 为剩余赛程中强队（ELO $>1600$）的数量。

\textbf{注}：为保持状态维度常数，可用统计量替代原始向量：
\begin{equation}
\tilde{\mathbf{O}}_t=
[\overline{o}_t,\ \text{Var}(o)_t,\ o^{(90\%)}_t],
\end{equation}
分别为均值、方差与 90\% 分位数（“强敌尾部风险”）。

% -------------------------
% 10) 赛程难度：SOS（单指标）
% -------------------------
\subsubsection{10. 赛程难度系数 $\text{SOS}_t$（Strength of Schedule）}

\begin{equation}
\text{SOS}_t
=
\frac{1}{m}\sum_{j=1}^{m}\frac{o_j}{\text{ELO}_{\text{avg}}},
\end{equation}
其中 $\text{ELO}_{\text{avg}}$ 为联盟平均 ELO。

（可选）若需体现压缩赛程惩罚，可在环境状态中另加背靠背数 $N_{\text{B2B}}$，或将其并入 $\text{SOS}_t$ 的修正项。

% -------------------------
% 数学表示总结（竞技状态仅此 10 项）
% -------------------------
\subsection{竞技状态空间总结}

最终，竞技资产状态空间采用常数维、可计算、可用于动态规划/ADP/MCTS 的表示：
\begin{equation}
\boxed{
\mathcal{R}_t
=
\Big\{
\mathbf{Q}_t,\;
\mathbf{C}_t,\;
\mathbf{P}_t,\;
\mathbf{L}_t,\;
\mathbf{A}_t,\;
\mathbf{W}_t,\;
\text{ELO}_t,\;
\text{Syn}_t,\;
\tilde{\mathbf{O}}_t,\;
\text{SOS}_t
\Big\}.
}
\end{equation}

\textbf{维度说明（示例）}：若 $\mathbf{Q}_t\in\mathbb{R}^4$，$\mathbf{C}_t\in\mathbb{R}^K$（$K=8$），则总体维度约为
$4+8+5+3+3+3+1+1+3+1=32$，
其中 $\tilde{\mathbf{O}}_t$ 为3维统计量，保持状态空间维数恒定，显著提升可求解性。
\section{建模建议}

\begin{enumerate}
    \item \textbf{降维处理}：使用主成分分析(PCA)将技术指标压缩至5-10个综合因子
    \item \textbf{分层建模}：战术层（单场比赛模拟）$\rightarrow$ 赛季层（82场积分）$\rightarrow$ 多年期（资产演化）
    \item \textbf{蒙特卡洛模拟}：对伤病/选秀/交易等随机事件进行1000+次情景推演
    \item \textbf{敏感性分析}：识别核心杠杆变量（如顶薪球员健康度、选秀权价值）
\end{enumerate}

\section{最终版财务状态空间 $\mathcal{F}_t$（精简核心版）}

\subsection{设计原则}
\begin{itemize}
    \item \textbf{只保留直接影响决策且无法从其他变量推导的核心变量}
    \item \textbf{明确区分存量（Balance Sheet）与流量（Cash Flow）}
    \item \textbf{与竞技状态 $\mathcal{R}_t$ 建立清晰的因果映射}
\end{itemize}

\subsection{核心财务状态向量}

$$\mathcal{F}_t = \{\lambda_t, \text{CF}_t, \mathbf{\Psi}_t, \text{Cap}_t, V_t, \text{Equity}_t\}$$

\subsubsection{1. 杠杆率 $\lambda_t \in [0, 1]$}

\textbf{定义}：
\begin{equation}
\lambda_t = \frac{D_t}{D_t + E_t}
\end{equation}

其中：
\begin{itemize}
    \item $D_t$: 总债务存量（银行贷款、债券等）
    \item $E_t$: 所有者权益（球队市场估值）
\end{itemize}

\textbf{状态分级}（用于策略分支）：
\begin{itemize}
    \item Safe: $\lambda < 0.3$（可积极投资，融资成本低）
    \item Balanced: $0.3 \leq \lambda < 0.6$（需谨慎管理）
    \item Distressed: $\lambda \geq 0.6$（被迫收缩，面临强制出售风险）
\end{itemize}

\textbf{转移机制}：
\begin{equation}
\lambda_{t+1} = \frac{D_t + \Delta D_t - \text{Repayment}_t}{(D_t + \Delta D_t - \text{Repayment}_t) + (E_t + \Delta E_t)}
\end{equation}

其中：
\begin{itemize}
    \item $\Delta D_t$: 当期新增债务（动作变量）
    \item $\text{Repayment}_t$: 债务偿还
    \item $\Delta E_t$: 估值变化（受竞技成绩和媒体周期影响）
\end{itemize}

\textbf{利息成本}：
\begin{equation}
\text{Interest}_t = D_t \times r_{\text{debt},t}
\end{equation}

其中债务利率随杠杆率上升：
\begin{equation}
r_{\text{debt},t} = r_{\text{risk-free}} + \beta \cdot \lambda_t + \delta \cdot \mathbb{I}(\text{CF}_t < 0)
\end{equation}

$\beta \approx 0.05$（杠杆惩罚系数），$\delta \approx 0.02$（流动性危机额外溢价）

\textbf{备注}：这是题目要求的"leverage调整"的核心量化指标。

---

\subsubsection{2. 经营性现金流 $\text{CF}_t \in \mathbb{R}$}

\textbf{定义}（简化的现金流瀑布）：
\begin{equation}
\text{CF}_t = \text{Rev}_t - \text{Cost}_t - \text{Tax}_t - \text{Interest}_t
\end{equation}

\textbf{收入总额}（二元结构）：
\begin{equation}
\text{Rev}_t = \underbrace{\text{Rev}_{\text{Performance}}}_{\text{竞技驱动}} + \underbrace{\text{Rev}_{\text{Star}}}_{\text{人气驱动}}
\end{equation}

\begin{itemize}
    \item $\text{Rev}_{\text{Performance}} = \alpha_1 \cdot \text{Win\%}_t + \alpha_2 \cdot \text{Playoff}_t$
    
    其中 $\alpha_1 \approx 2$M（胜率每提升10\%带来200万美元收入），$\alpha_2 \in [0.5, 5]$M（季后赛轮次奖金）
    
    \item $\text{Rev}_{\text{Star}} = \beta_1 \cdot \text{StarPower}_t + \beta_2 \cdot \text{MediaExposure}_t$
    (注：$\text{StarPower}$ 与 $\text{MediaExposure}$ 为基于 $\mathcal{R}_t$ 与 $\mathcal{E}_t$ 的衍生观测变量，不作为独立状态维度)
    
    其中 $\beta_1 \approx 1.5$M（顶级球星带来150万美元额外收入），$\beta_2$（全国转播场次系数）
\end{itemize}

\textbf{成本总额}（三大块）：
\begin{equation}
\text{Cost}_t = \text{Salary}_{\text{Total}} + \text{Ops}_{\text{Fixed}} + \text{Venue}_t
\end{equation}

\begin{itemize}
    \item $\text{Salary}_{\text{Total}} = \sum_{i=1}^{N_{\text{roster}}} \text{Salary}_i$（所有球员合同总和，直接来自财务状态 $\mathbf{\Psi}_t$）

    \item $\text{Ops}_{\text{Fixed}}$: 固定运营成本（教练、管理层、差旅，约3-5M美元/年）
    \item $\text{Venue}_t$: 场馆成本
    \begin{equation}
    \text{Venue}_t = \begin{cases}
    \text{Rent}_{\text{annual}}, & \text{if 租赁模式} \\
    \frac{\text{VenueValue}}{T_{\text{depreciation}}} + \text{Maintenance}_t, & \text{if 自有模式}
    \end{cases}
    \end{equation}
\end{itemize}

\textbf{税收支出}（简化）：
\begin{equation}
\text{Tax}_t = \max\left(0, (\text{Rev}_t - \text{Cost}_t - \text{Interest}_t) \times r_{\text{corporate}}\right)
\end{equation}

其中 $r_{\text{corporate}} \approx 25\%$（联邦+州综合税率）

\textbf{关键阈值}：
\begin{itemize}
    \item $\text{CF}_t > 0$: 健康状态，可积累现金储备
    \item $\text{CF}_t < 0$: 亏损状态，需动用储备或融资
    \item $\text{CF}_t < -C_{\text{min}}$: 触发"强制出售"吸收态（如连续3年亏损超5M）
\end{itemize}

\textbf{现金储备演化}：
\begin{equation}
\text{Cash}_{t+1} = \text{Cash}_t + \text{CF}_t + \Delta D_t - \text{Repayment}_t - \text{Dividends}_t
\end{equation}

\textbf{备注}：这是短期生存能力的核心指标，决定球队能否"熬过"重建期。

---

\subsubsection{3. 薪资与合同结构向量 $\mathbf{\Psi}_t$（Salary Structure）}

\begin{equation}
\mathbf{\Psi}_t = [\mu_{S,t},\ \sigma_{S,t},\ r_{\text{max},t},\ S_{\text{commit},t}]
\end{equation}

\begin{itemize}
    \item $\mu_{S,t}$ (\textbf{Salary Level})：平均薪资水平，作为计算总薪资的基准。
    \item $\sigma_{S,t}$ (\textbf{Salary Distribution})：薪资分布特征（如方差或基尼系数），反映薪资结构的集中度（避免“中产陷阱”）。
    \item $r_{\text{max},t}$ (\textbf{Max Salary Ratio})：顶薪球员薪资占比，衡量头部球星对空间的占用。
    \item $S_{\text{commit},t}$ (\textbf{Contract Amount})：未来保障合同总金额，反映长期财务刚性。
\end{itemize}

---

\subsubsection{4. 工资帽状态 $\text{Cap}_t = \{S_{\text{used}}, S_{\text{avail}}, \text{Tax}_{\text{Status}}\}$}

\textbf{已用薪资总额}：
\begin{equation}
S_{\text{used}} = \sum_{i=1}^{N_{\text{roster}}} \text{Salary}_i
\end{equation}

\textbf{可用薪资空间}：
\begin{equation}
S_{\text{avail}} = \text{Cap}_{\text{limit}} - S_{\text{used}}
\end{equation}

\textbf{税收状态}（离散化为3档）：
\begin{equation}
\text{Tax}_{\text{Status}} = \begin{cases}
0 \quad (\text{Under Cap}), & \text{if } S_{\text{used}} < \text{Cap}_{\text{limit}} \\
1 \quad (\text{Taxpayer}), & \text{if } \text{Cap}_{\text{limit}} \leq S_{\text{used}} < \text{Apron}_1 \\
2 \quad (\text{Hard Apron}), & \text{if } S_{\text{used}} \geq \text{Apron}_2
\end{cases}
\end{equation}

\textbf{奢侈税惩罚}（若适用）：
\begin{equation}
\text{LuxuryTax}_t = \begin{cases}
0, & \text{if Tax\_Status} = 0 \\
\sum_{k} \text{Bracket}_k \times \text{Rate}_k, & \text{if Tax\_Status} \geq 1
\end{cases}
\end{equation}

其中税率为累进制：
\begin{itemize}
    \item 首个\$5M超出部分：税率1.5倍（\$1.50/\$1.00）
    \item 次个\$5M：税率2.0倍
    \item 超过\$10M：税率3.0倍以上
    \item 若连续3年纳税（Repeater Tax）：税率再乘1.5倍
\end{itemize}

\textbf{动作空间约束}：
\begin{itemize}
    \item Tax\_Status = 0: 可全额使用中产特例，可大额交易
    \item Tax\_Status = 1: 部分交易特例失效
    \item Tax\_Status = 2: 禁止打包交易、首轮签交易受限、无中产特例
\end{itemize}

\textbf{工资帽增长}（来自环境状态）：
\begin{equation}
\text{Cap}_{\text{limit},t+1} = \text{Cap}_{\text{limit},t} \times (1 + g_{\text{cap}})
\end{equation}

其中 $g_{\text{cap}} \in \{8\%, 12\%, 40\%\}$（平滑增长 vs 媒体合约跳跃）

\textbf{备注}：这是硬约束，直接限制动作空间 $\mathcal{A}_t$。

---

\subsubsection{5. 品牌估值增速 $V_t \in \mathbb{R}$（替代绝对估值）}

\textbf{定义}（相对于上一期的估值变化率，以小数表示）：
\begin{equation}
V_t = \frac{E_t - E_{t-1}}{E_{t-1}}
\end{equation}

\textbf{驱动因素}（回归模型）：
\begin{equation}
V_t = \gamma_1 \cdot \Delta\text{Win\%} + \gamma_2 \cdot \text{MediaDeal}_{\text{Spike}} + \gamma_3 \cdot \text{StarAcquisition} + \gamma_4 \cdot \text{CF}_t + \epsilon
\end{equation}

参数估计（示例）：
\begin{itemize}
    \item $\gamma_1 \approx 0.05$（胜率提升10\%导致估值增长5\%）
    \item $\gamma_2 \approx 0.30$（新媒体合约生效导致估值跳跃30\%）
    \item $\gamma_3 \approx 0.15$（获得超级球星导致估值增长15\%）
    \item $\gamma_4 \approx 0.0002$（每百万美元利润增加0.02\%估值）
\end{itemize}

\textbf{状态分级}：
\begin{itemize}
    \item 高增长（$V > 0.20$）：估值飙升期（如获得Caitlin Clark）
    \item 稳定（$0 < V < 0.20$）：正常发展
    \item 贬值（$V < 0$）：球队价值下滑
\end{itemize}

\textbf{长期估值累积}：
\begin{equation}
E_t = E_0 \times \prod_{\tau=1}^{t} (1 + V_{\tau})
\end{equation}

\textbf{备注}：这捕捉题目要求的"特许经营权品牌资产"，用增速而非绝对值避免数值过大。

---

\subsubsection{6. 股权结构与球员权益 $\text{Equity}_t = \{\text{Owner}_{\text{share}}, \text{Player}_{\text{equity}}, \text{Dilution}_t\}$}

\textbf{核心问题}：题目明确提到"球员股权可作为薪资替代方案"（profit participation, equity stake等）

\textbf{所有者权益占比}：
\begin{equation}
\text{Owner}_{\text{share},t} = \frac{E_{\text{owner}}}{E_t} \times 100\%
\end{equation}

初始状态：$\text{Owner}_{\text{share},0} = 100\%$

\textbf{球员股权池}（员工持股计划，ESOP）：
\begin{equation}
\text{Player}_{\text{equity},t} = \sum_{i \in \text{Granted}} \omega_i
\end{equation}

其中 $\omega_i$ 为球员 $i$ 的持股比例（通常 $0.1\% \sim 5\%$）

\textbf{股权授予条件}：
\begin{itemize}
    \item \textbf{类型1：收入分成}（Revenue Sharing）
    
    球员 $i$ 获得当年收入的额外分成：
    \begin{equation}
    \text{Bonus}_i = \text{Rev}_t \times \omega_i \times \mathbb{I}(\text{Rev}_t > \text{Threshold})
    \end{equation}
    
    \item \textbf{类型2：利润分成}（Profit Participation）
    
    球员 $i$ 获得当年净利润的分成：
    \begin{equation}
    \text{Bonus}_i = \max(0, \text{CF}_t) \times \omega_i
    \end{equation}
    
    \item \textbf{类型3：长期股权}（Equity Stake）
    
    球员 $i$ 获得球队 $\omega_i\%$ 的永久股权，退役后仍享有：
    \begin{itemize}
        \item 年度分红权（如球队盈利时按比例分配）
        \item 增值收益权（球队出售时按估值分成）
    \end{itemize}
    
    \textbf{估值退出收益}：
    \begin{equation}
    \text{Payout}_i = (E_T - E_0) \times \omega_i \quad \text{（球队出售时）}
    \end{equation}
\end{itemize}

\textbf{股权稀释动态}：
\begin{equation}
\text{Owner}_{\text{share},t+1} = \text{Owner}_{\text{share},t} \times \frac{E_t}{E_t + \Delta E_{\text{new}}}
\end{equation}

当授予新股权或引入外部投资者时发生稀释

\textbf{决策权衡}（动作空间中的选择）：

对于顶薪球员 $i$，所有者可选择：
\begin{enumerate}
    \item \textbf{纯现金合同}：
    \begin{equation}
    \text{Salary}_i = \$500k/\text{year}, \quad \omega_i = 0\%
    \end{equation}
    
    \item \textbf{现金+股权混合}：
    \begin{equation}
    \text{Salary}_i = \$350k/\text{year}, \quad \omega_i = 2\%, \quad \text{期望总价值} \approx \$500k
    \end{equation}
    
    优势：当期现金流压力减小（CF\_t改善），但未来利润需分享
    
    \item \textbf{股权激励型}（适用于重建期）：
    \begin{equation}
    \text{Salary}_i = \$250k/\text{year}, \quad \omega_i = 5\%, \quad \text{Vesting Period} = 4\text{年}
    \end{equation}
    
    若球队4年内估值从\$100M增至\$200M，球员额外获得：
    \begin{equation}
    \text{Gain}_i = (\$200M - \$100M) \times 5\% = \$5M
    \end{equation}
\end{enumerate}

\textbf{约束条件}：
\begin{itemize}
    \item \textbf{总股权池上限}：
    \begin{equation}
    \sum_i \omega_i \leq \omega_{\max} \quad (\omega_{\max} \approx 10\%-15\%)
    \end{equation}
    
    避免过度稀释导致所有者失去控制权
    
    \item \textbf{流动性要求}：
    
    股权计划不能导致现金流枯竭：
    \begin{equation}
    \text{Cash}_t - \text{Salary}_{\text{Total}} \geq C_{\min}
    \end{equation}
    
    \item \textbf{球员接受度}：
    
    只有当期望总价值≥纯现金合同时，球员才接受股权方案：
    \begin{equation}
    \text{Salary}_i + \mathbb{E}[\text{Equity Gain}_i] \geq \text{MarketValue}_i
    \end{equation}
\end{itemize}

\textbf{股权退出机制}（Liquidity Events）}：

\begin{enumerate}
    \item \textbf{球队出售}：
    
    所有股权持有者按比例分享交易对价
    
    \item \textbf{回购条款}（Buyback）：
    
    球员退役/交易时，球队有权按公允价值回购股权：
    \begin{equation}
    \text{Buyback Price} = E_t \times \omega_i \times (1 - \text{Discount})
    \end{equation}
    
    Discount通常10-20\%（流动性折扣）
    
    \item \textbf{分红政策}：
    
    若球队选择不分红（将利润全部再投资），股权持有者无当期收益，只能等待长期增值
\end{enumerate}

\textbf{财务影响建模}：

\textbf{对现金流的影响}：
\begin{equation}
\text{CF}_t^{\text{with equity}} = \text{CF}_t^{\text{no equity}} + \underbrace{(\text{Salary}_{\text{saved}} - \text{Dividends}_{\text{paid}})}_{\text{净效应}}
\end{equation}

\begin{itemize}
    \item 短期：节省现金工资，CF\_t改善
    \item 长期：需支付分红或面临回购压力，CF\_t恶化
\end{itemize}

\textbf{对估值的影响}：
\begin{equation}
E_t^{\text{diluted}} = E_t \times (1 - \sum_i \omega_i)
\end{equation}

所有者实际拥有的价值被稀释

\textbf{最优化问题}（所有者视角）：

在给定薪资帽约束下，选择股权授予比例以最大化期望净现值：
\begin{equation}
\max_{\{\omega_i\}} \mathbb{E}\left[ \sum_{t=0}^{T} \frac{\text{CF}_t \times (1 - \sum_i \omega_i)}{(1+r)^t} + \frac{E_T \times (1 - \sum_i \omega_i)}{(1+r)^T} \right]
\end{equation}

约束条件：
\begin{align}
&\sum_i \text{Salary}_i \leq \text{Cap}_{\text{limit}} \\
&\sum_i \omega_i \leq \omega_{\max} \\
&\text{Salary}_i + \mathbb{E}[\text{Equity Gain}_i] \geq \text{MarketValue}_i \quad \forall i
\end{align}

\textbf{备注}：股权激励在初创/重建期球队中特别有效，因为估值增长空间大，球员愿意"赌未来"。

---

\subsection{财务状态的极简表示}

\begin{equation}
\mathcal{F}_t = \begin{cases}
\lambda_t \in [0, 1] & \text{（杠杆率）} \\
\text{CF}_t \in \mathbb{R} & \text{（经营现金流）} \\
\mathbf{\Psi}_t \in \mathbb{R}^4 & \text{（薪资结构：均值/方差/顶薪比/保障金额）} \\
S_{\text{avail}} \in [0, \text{Cap}_{\text{limit}}] & \text{（可用薪资空间）} \\
\text{Tax}_{\text{Status}} \in \{0, 1, 2\} & \text{（税收档位）} \\
V_t \in \mathbb{R} & \text{（品牌估值增速）} \\
\text{Equity}_t \in \mathbb{R}^2 & \text{（所有者权益占比与球员持股池）}
\end{cases}
\end{equation}

\textbf{总维度}：约9维（与 Section 10 统一状态空间保持一致）。

---

\section{最终版环境状态空间 $\mathcal{E}_t$（精简核心版）}

\subsection{设计原则}
\begin{itemize}
    \item \textbf{只保留"外生冲击"和"不可控约束"}
    \item \textbf{宏观经济用单一指标代理}
    \item \textbf{联盟政策聚焦于"阶跃式变化"（如扩军、工资帽暴涨）}
\end{itemize}

\subsection{核心环境状态向量}

$$\mathcal{E}_t = \{\text{Macro}_t, \text{League}_t, \text{Market}_t, \text{FA}_{\text{Market},t}\}$$

---

\subsubsection{1. 宏观经济景气度 $\text{Macro}_t \in \{\text{衰退}, \text{正常}, \text{繁荣}\}$}

\textbf{定义}（离散化的经济周期指标）：
\begin{itemize}
    \item \textbf{衰退}：消费者支出下降，门票收入-20\%，赞助合同难续约
    \item \textbf{正常}：基准状态
    \item \textbf{繁荣}：娱乐消费旺盛，门票溢价+15\%，新赞助商涌入
\end{itemize}

\textbf{转移概率}（马尔可夫链）：
\begin{equation}
P(\text{Macro}_{t+1} | \text{Macro}_t) = \begin{bmatrix}
0.7 & 0.25 & 0.05 \\
0.15 & 0.7 & 0.15 \\
0.05 & 0.3 & 0.65
\end{bmatrix}
\end{equation}

行顺序：\{衰退, 正常, 繁荣\}

\textbf{影响机制}：
\begin{equation}
\text{Rev}_t = \text{Rev}_{\text{base}} \times \begin{cases}
0.8, & \text{if Macro}_t = \text{衰退} \\
1.0, & \text{if Macro}_t = \text{正常} \\
1.15, & \text{if Macro}_t = \text{繁荣}
\end{cases}
\end{equation}

\textbf{备注}：用三状态马尔可夫链替代复杂的GDP/CPI/利率多维建模。

---

\subsubsection{2. 联盟政策周期 $\text{League}_t = \{\text{Cap}_{\text{Growth}}, I_{\text{Expansion}}, T_{\text{MediaDeal}}\}$}

\textbf{工资帽增长模式 Cap\_Growth}：
\begin{equation}
\text{Cap}_{\text{Growth}} \in \{\text{平滑}, \text{跳跃}\}
\end{equation}

\begin{itemize}
    \item \textbf{平滑增长}（Smooth）：每年 $g_{\text{cap}} = 8\%-12\%$（可预测）
    \item \textbf{跳跃增长}（Spike）：某年突增 $g_{\text{cap}} = 40\%$（2026年WNBA新媒体合约）
\end{itemize}

\textbf{工资帽演化}：
\begin{equation}
\text{Cap}_{\text{limit},t+1} = \text{Cap}_{\text{limit},t} \times (1 + g_{\text{cap}})
\end{equation}

---

\textbf{扩军时间表 $I_{\text{Expansion}}$}：
\begin{equation}
I_{\text{Expansion},t} = \begin{cases}
1, & \text{if } t \in \{2026, 2028, ...\} \\
0, & \text{otherwise}
\end{cases}
\end{equation}

\textbf{扩军影响}（一次性冲击）：

\begin{enumerate}
    \item \textbf{扩军费分红}：
    \begin{equation}
    \text{CF}_t = \text{CF}_t + \frac{\text{ExpansionFee}}{N_{\text{teams}}} \quad \text{（一次性+5M美元）}
    \end{equation}
    
    \item \textbf{人才稀释}：
    
    自由市场球员平均技能下降：
    \begin{equation}
    \mathbb{E}[S_i^{\text{FA}}] = \mathbb{E}[S_i^{\text{FA}}]_{\text{baseline}} \times 0.95
    \end{equation}
    
    \item \textbf{保护名单要求}：
    
    每队需提交保护名单 $P_{\text{Protected}} \subseteq \{1, 2, ..., 12\}$，通常规则：
    \begin{equation}
    |P_{\text{Protected}}| = 6 \sim 8
    \end{equation}
    
    未保护球员可被新球队选走（每队最多失去1名球员）
\end{enumerate}

---

\textbf{媒体合约倒计时 $T_{\text{MediaDeal}}$}：
\begin{equation}
T_{\text{MediaDeal}} \in \{5, 4, 3, 2, 1, 0\}
\end{equation}

\textbf{合约生效触发}：

当 $T_{\text{MediaDeal}} = 0$ 时触发"估值跳跃"：
\begin{equation}
V_t = V_{t-1} + \text{Spike}_{\text{media}} \quad (\text{Spike} \approx 30\%-50\%)
\end{equation}

同时：
\begin{equation}
\text{Rev}_{\text{media},t} = \text{Rev}_{\text{media},t-1} \times (1 + g_{\text{media}})
\end{equation}

其中 $g_{\text{media}} \approx 200\%-300\%$（2026年WNBA新合约预期）

\textbf{倒计时更新}：
\begin{equation}
T_{\text{MediaDeal},t+1} = \begin{cases}
T_{\text{MediaDeal},t} - 1, & \text{if } T_{\text{MediaDeal},t} > 0 \\
10, & \text{if } T_{\text{MediaDeal},t} = 0 \quad \text{（重置为下个周期）}
\end{cases}
\end{equation}

\textbf{备注}：这三个子变量共同描述"联盟层面的制度环境"。

---

\subsubsection{3. 本队市场位置 $\text{Market}_t$（准静态参数）}

\textbf{市场规模系数 $\mu_{\text{size}} \in [0.5, 2.0]$}：

\begin{itemize}
    \item \textbf{超大市场}（纽约、洛杉矶）：$\mu_{\text{size}} = 2.0$
    \item \textbf{大市场}（芝加哥、旧金山）：$\mu_{\text{size}} = 1.5$
    \item \textbf{中等市场}（西雅图、凤凰城）：$\mu_{\text{size}} = 1.0$
    \item \textbf{小市场}（印第安纳、康涅狄格）：$\mu_{\text{size}} = 0.6$
\end{itemize}

\textbf{影响公式}：
\begin{equation}
\text{Rev}_{\text{gate}} = \text{Base} \times \mu_{\text{size}} \times f(\text{Win\%}, \text{StarPower})
\end{equation}

---

\textbf{本地竞争强度 $\text{Compete}_{\text{local}} \in \{\text{低}, \text{中}, \text{高}\}$}：

\begin{itemize}
    \item \textbf{低}：本城市无其他主流职业球队（如印第安纳，WNBA是夏季唯一主角）
    \item \textbf{中}：有1-2支竞争球队
    \item \textbf{高}：体育市场高度饱和（纽约9支职业球队）
\end{itemize}

\textbf{修正因子}：
\begin{equation}
\text{MediaShare} = \frac{1}{1 + 0.2 \times N_{\text{competitors}}}
\end{equation}

\textbf{综合市场效应}：
\begin{equation}
\text{Rev}_t = \text{Rev}_{\text{base}} \times \mu_{\text{size}} \times \text{MediaShare}
\end{equation}

\textbf{备注}：$\mu_{\text{size}}$ 和 $\text{Compete}_{\text{local}}$ 在球队建立时确定，不随时间变化（因此严格说是"参数"而非"状态"，但为了建模完整性放在这里）。

---

\subsubsection{4. 自由市场热度 $\text{FA}_{\text{Market},t} = \{N_{\text{Star}}, \text{Bidding}_t\}$}

\textbf{顶级球星供给数量}：
\begin{equation}
N_{\text{Star}} = |\{i : C_i \in \{\text{核心得分手}, \text{全能侧翼}\} \land l_i = 0\}|
\end{equation}

其中 $C_i$ 为球员类型（来自竞技状态 $\mathcal{R}_t$），$l_i$ 为剩余合同年限

---

\textbf{竞价激烈度}（其他球队的薪资空间总和）：
\begin{equation}
\text{Bidding}_t = \sum_{j \neq \text{own}} S_{\text{avail},j}
\end{equation}

\textbf{影响}：
\begin{itemize}
    \item $N_{\text{Star}}$ 高且 $\text{Bidding}$ 低：买方市场，可低价签约
    \item $N_{\text{Star}}$ 低且 $\text{Bidding}$ 高：卖方市场，溢价严重
\end{itemize}

\textbf{签约成本函数}：
\begin{equation}
\text{Contract}_{\text{offer}} = \text{FairValue}_i \times \left(1 + 0.5 \times \frac{\text{Bidding}_t}{\text{Bidding}_{\text{avg}}}\right) \times \left(1 - 0.3 \times \frac{N_{\text{Star}}}{N_{\text{Star,avg}}}\right)
\end{equation}

\textbf{状态转移}：

每个休赛期，自由市场状态重置：
\begin{align}
N_{\text{Star},t+1} &= |\{i : l_i = 0 \text{ at end of season } t\}| \\
\text{Bidding}_{t+1} &= \sum_{j} (\text{Cap}_{\text{limit}} - S_{\text{used},j})
\end{align}

\textbf{备注}：这捕捉"其他球队决策的集体效应"，是博弈论元素的简化表达。

---

\subsection{环境状态的极简表示（最终方案）}

\begin{equation}
\mathcal{E}_t = \begin{cases}
\text{Macro}_t \in \{\text{衰退}, \text{正常}, \text{繁荣}\} & \text{（经济周期）} \\
\text{Cap}_{\text{Growth}} \in \{\text{平滑}, \text{跳跃}\} & \text{（工资帽增长模式）} \\
I_{\text{Expansion}} \in \{0, 1\} & \text{（扩军年标记）} \\
T_{\text{MediaDeal}} \in \{0, 1, 2, ..., 5\} & \text{（媒体合约倒计时）} \\
\mu_{\text{size}} \in [0.5, 2.0] & \text{（市场规模系数）} \\
\text{Compete}_{\text{local}} \in \{\text{低}, \text{中}, \text{高}\} & \text{（本地竞争强度）} \\
N_{\text{Star,FA}} \in \mathbb{Z}_+ & \text{（自由市场球星数）} \\
\text{Bidding}_t \in \mathbb{R}_+ & \text{（竞价强度，百万美元）}
\end{cases}
\end{equation}

\textbf{总维度}：8维（相比原方案的15+维大幅精简）

---

\section{状态空间集成设计}

\subsection{完整状态向量 $\mathcal{S}_t = \{\mathcal{R}_t, \mathcal{F}_t, \mathcal{E}_t, \Theta_t\}$}

\begin{itemize}
    \item \textbf{竞技状态 $\mathcal{R}_t$}：10维（球员技能、阵容结构、健康度、选秀权）
    \item \textbf{财务状态 $\mathcal{F}_t$}：9维（杠杆率、现金流、薪资结构、配置状态等）
    \item \textbf{环境状态 $\mathcal{E}_t$}：8维（经济、联盟政策、市场、自由市场）
    \item \textbf{阶段状态 $\Theta_t$}：1维（休赛期/常规赛/交易期/季后赛）
\end{itemize}

\textbf{总维度}：约30维（满足精确控制需求的同时保持计算可行性）

---


\section{MDP 统一状态空间的形式化定义}

为解决跨周期的动态决策问题（涵盖赛季内的滚动决策与休赛期的资产重构），并精确响应联盟赛程规则与扩军事件，系统的总体状态空间定义为竞技、财务、环境三个子空间与阶段状态的笛卡尔积：
\begin{equation}
\mathcal{S} = \mathcal{R} \times \mathcal{F} \times \mathcal{E} \times \Theta
\end{equation}

\subsection{1. 赛季阶段状态 $\Theta_t$}
该变量充当MDP的“决策模式控制器”，决定了模型在当前时刻的可行动作空间 $\mathcal{A}(s_t)$ 及状态转移逻辑：
\begin{equation}
\Theta_t \in \{\text{Offseason},\ \text{Regular},\ \text{TradeDeadline},\ \text{Playoff}\}
\end{equation}
\begin{itemize}
    \item \textbf{Offseason (休赛期)}：决策核心为资产重构。允许选秀（含扩军选秀响应）、自由市场竞价、股权结构调整及长期杠杆规划。
    \item \textbf{Regular (常规赛)}：决策核心为运营优化。允许双向合同签约、票价动态调整、短期伤病特例申请，不可进行核心阵容变更（除非交易期）。
    \item \textbf{TradeDeadline (交易截止日)}：决策核心为博弈。最后调整阵容窗口，面临信息不对称与高溢价风险。
    \item \textbf{Playoff (季后赛)}：决策核心为风险控制。花名册冻结，仅允许战术层面调整与场馆运营优化。
\end{itemize}

\subsection{2. 竞技资产状态 $\mathcal{R}_t$}
采用前文确定的 10 维核心竞技特征向量：
\begin{equation}
\mathcal{R}_t = \Big\{ \mathbf{Q}_t,\; \mathbf{C}_t,\; \mathbf{P}_t,\; \mathbf{L}_t,\; \mathbf{A}_t,\; \mathbf{W}_t,\; \text{ELO}_t,\; \text{Syn}_t,\; \tilde{\mathbf{O}}_t,\; \text{SOS}_t \Big\}
\end{equation}

\subsection{3. 财务状态 $\mathcal{F}_t$}
为了解决 MDP 闭合性问题并统一杠杆定义，我们显式引入\textbf{配置状态} $\mathbf{K}_t$ 和\textbf{所有者权益} $s_t$：
\begin{equation}
\mathcal{F}_t = \Big\{ FV_t,\ D_t,\ \lambda_t,\ \text{CF}_t,\ \mathbf{\Psi}_t,\ S_{\text{avail},t},\ \text{Tax}_{\text{Status},t},\ V_t,\ s_t,\ \mathbf{K}_t \Big\}
\end{equation}

其中关键变量修正定义如下：
\begin{itemize}
    \item $\lambda_t = \frac{D_t}{FV_t}$：\textbf{统一杠杆率}。分母采用特许经营权价值（Franchise Value）$FV_t$，与后续终值函数 $FV_T-D_T$ 保持一致；
    \item $s_t$：\textbf{所有者持股比例}（Owner's Share），取值 $s_t\in[0,1]$，初始 $s_0=1$，随股权激励 $a_{\text{equity}}$ 逐年摊薄；
    \item $\mathbf{K}_t$：\textbf{经营配置状态}（Configuration State）。记录当前生效的票价、营销、分红、薪资和债务档位，用于保证 Action Masking 的马尔可夫性：
    \begin{equation}
        \mathbf{K}_t = [k_{\text{ticket}},\ k_{\text{marketing}},\ k_{\text{equity}},\ k_{\text{salary}},\ k_{\text{debt}}]
    \end{equation}
    \item $\mathbf{\Psi}_t$：薪资与合同结构向量（含 $\mu_{S,t},\ \sigma_{S,t},\ r_{\text{max},t},\ S_{\text{commit},t}$）；
    \item $V_t$：本期品牌估值增长率（Valuation Growth Rate）。
\end{itemize}

\subsection{4. 环境状态 $\mathcal{E}_t$}
包含 8 个核心外生变量（含扩军标记 $I_{\text{Expansion}}$，与休赛期阶段 $\Theta_t=\text{Offseason}$ 结合触发扩军选秀逻辑）：
\begin{equation}
\mathcal{E}_t = \Big\{ \text{Macro}_t,\ \text{Cap}_{\text{Growth}},\ I_{\text{Expansion}},\ T_{\text{MediaDeal}},\ \mu_{\text{size}},\ \text{Compete}_{\text{local}},\ N_{\text{Star,FA}},\ \text{Bidding}_t \Big\}
\end{equation}

\section{MDP 动作空间与决策流定义}

为了克服传统“概念列表式”动作空间维度可变、稀疏性高的问题，本文将决策变量标准化为**可计算的低维离散动作向量** $\mathbf{a}_t$。这使得问题可直接适配于 DQN 或 PPO 等强化学习算法。

\subsection{1. 统一动作向量定义}
在任意时间步 $t$，动作向量定义为 6 维整型向量：
\begin{equation}
\mathbf{a}_t = \{a_{\text{roster}},\; a_{\text{salary}},\; a_{\text{ticket}},\; a_{\text{marketing}},\; a_{\text{debt}},\; a_{\text{equity}}\}
\end{equation}
每个分量对应一个离散化的决策档位，具体定义如下：

\subsection{2. 分量详解}

\subsubsection{1) 竞技构建决策 $a_{\text{roster}}$ (Roster Strategy)}
决定球队在转会市场、选秀大会或交易截止日的操作方向与力度。
\begin{itemize}
    \item \textbf{空间定义}：$a_{\text{roster}} \in \{0, 1, 2, 3, 4\}$
    \begin{itemize}
        \item 0 (\textbf{Seller-Aggressive}): “清仓重建”。积极送出老将换取选秀权（$\downarrow \mathbf{Q}_t, \uparrow \mathbf{C}_t$）。
        \item 1 (\textbf{Seller-Conservative}): “微调避税”。送出特定合同以降低薪资。
        \item 2 (\textbf{Hold}): “保持现状”。不进行主动交易操作。
        \item 3 (\textbf{Buyer-Conservative}): “补强”。用次轮签或中产特例补充角色球员。
        \item 4 (\textbf{Buyer-Aggressive}): “梭哈争冠”。送出未来资产换取即战力巨星（$\uparrow \mathbf{Q}_t, \downarrow \mathbf{C}_t$）。
    \end{itemize}
    \item \textbf{阶段约束}：在 $\Theta_t = \text{Playoff}$ 或非交易期，该分量被强制锁定为 2 (Hold)。
\end{itemize}

\subsubsection{2) 薪资空间管理 $a_{\text{salary}}$ (Spending Target)}
设定本周期的目标薪资支出水平，间接控制签约动作。
\begin{itemize}
    \item \textbf{空间定义}：$a_{\text{salary}} \in \{0, 1, 2, 3\}$
    \begin{itemize}
        \item 0 (\textbf{Floor}): 仅满足穷人线（Cap Min），追求利润最大化。
        \item 1 (\textbf{Over-Cap}): 超过工资帽但低于税线，使用全额中产。
        \item 2 (\textbf{Taxpayer}): 进入奢侈税区间，保留核心阵容。
        \item 3 (\textbf{Apron}): 突破硬工资帽（Apron），不惜重金缴纳罚款打造豪门。
    \end{itemize}
\end{itemize}

\subsubsection{3) 票务与商业定价 $a_{\text{ticket}}$ (Ticket Pricing)}
\begin{itemize}
    \item \textbf{空间定义}：$a_{\text{ticket}} \in \{0, 1, 2, 3, 4\}$，对应倍率 $M_{\text{ticket}} \in \{0.9,\ 1.0,\ 1.1,\ 1.2,\ 1.3\}$
    \item \textbf{通过机制}：票价 = $\text{BasePrice} \times M_{\text{ticket}}$。
    \item \textbf{权衡}：高倍率提升短期 $Rev_{\text{gate}}$，但降低上座率并长期损害 $\mu_{\text{size}}$（球迷基础）。
\end{itemize}

\subsubsection{4) 市场营销投入 $a_{\text{marketing}}$ (Marketing Investment)}
\begin{itemize}
    \item \textbf{空间定义}：$a_{\text{marketing}} \in \{0, 1, 2\}$
    \begin{itemize}
        \item 0 (\textbf{Low}): 维持最低曝光，无额外支出。
        \item 1 (\textbf{Medium}): 投入营收的 5\% 用于推广，平稳提升 $V_t$。
        \item 2 (\textbf{High}): 投入营收的 10\% 进行全美推广（或流媒体合作），显著提升 $V_t$ 和 $\text{MediaExposure}$。
    \end{itemize}
\end{itemize}

\subsubsection{5) 债务杠杆调整 $a_{\text{debt}}$ (Leverage Control)}
\begin{itemize}
    \item \textbf{空间定义}：$a_{\text{debt}} \in \{0, 1, 2\}$
    \begin{itemize}
        \item 0 (\textbf{Deleverage}): 优先偿还债务（$\Delta D < 0$）。
        \item 1 (\textbf{Maintain}): 借新还旧，维持当前杠杆率。
        \item 2 (\textbf{Leverage Up}): 激进融资（$\Delta D > 0$），用于球场建设或支付巨额薪资。
    \end{itemize}
\end{itemize}

\subsubsection{6) 股权激励授予 $a_{\text{equity}}$ (Equity Dilution)}
针对题目核心特征（WNBA球员权益），决定稀释多少股权给予球员。
\begin{itemize}
    \item \textbf{空间定义}：$a_{\text{equity}} \in \{0, 1, 2, 3\}$，对应授予比例 $\omega \in \{0\%,\ 1\%,\ 2\%,\ 5\%\}$
    \item \textbf{作用}：授予越高，$\mathbf{\Psi}_t$ 中的现金薪资压力越小，球员满意度/留队率越高，但所有者剩余价值（Terminal Value）受损。
\end{itemize}

\subsection{3. 动作有效性与阶段约束 (Action Masking \& Markov Consistency)}
为确保决策的现实可行性且不破坏 Markov 性，我们使用\textbf{配置状态} $\mathbf{K}_t$ 作为约束基准，而非依赖上一时刻动作 $a_{t-1}$。
只有在当前阶段允许改变的动作分量才可变，其余分量必须强制等于当前配置状态 $\mathbf{K}_t$ 对应的值。

\begin{table}[h]
\centering
\caption{各赛季阶段允许的动作维度}
\begin{tabular}{|l|c|c|c|c|}
\hline
\textbf{动作分量} & \textbf{Offseason (休赛期)} & \textbf{Regular (常规赛)} & \textbf{TradeDeadline (交易期)} & \textbf{Playoff (季后赛)} \\ \hline
$a_{\text{roster}}$ & \checkmark & - & \checkmark & - \\ \hline
$a_{\text{salary}}$ & \checkmark & - & \checkmark & - \\ \hline
$a_{\text{ticket}}$ & \checkmark & - & - & - \\ \hline
$a_{\text{marketing}}$ & \checkmark & \checkmark & \checkmark & \checkmark \\ \hline
$a_{\text{debt}}$ & \checkmark & - & - & - \\ \hline
$a_{\text{equity}}$ & \checkmark & \checkmark & - & - \\ \hline
\end{tabular}
\end{table}

\textbf{掩码机制}：
\begin{equation}
\mathcal{A}_{\text{valid}}(s_t) = \Big\{ \mathbf{a} \in \mathcal{A} \ \Big|\ \forall j,\ \text{Mutable}(j, \Theta_t) \ \lor\ a_j = \mathbf{K}_t[j] \Big\}
\end{equation}
这确保了在冻结期（如季后赛），Agent 只能延续上一时刻的状态配置（$a_j = k_j$），消除了对 $a_{t-1}$ 的非马尔可夫依赖。

\section{系统动力学：状态转移方程}

\subsection{1. 赛季阶段演化 (Phase Dynamics)}
阶段变量 $\Theta_t$ 遵循确定性的事件驱动逻辑：$\Theta_{t+1} = \text{Next}(\Theta_t)$。

\subsection{2. 竞技状态转移 (Competitive Dynamics)}
（此部分保持不变：$\mathbf{Q}_{t+1}$ 受交易影响，$\mathbf{A}_{t+1}$ 自然老化，$\mathbf{L}_{t+1}$ 合同滚动，战绩受 ELO 驱动）

\subsection{3. 财务状态转移 (Financial Dynamics)}
为彻底解决变量定义冲突，本节严格遵循财务状态空间（Section 11）的定义体系，引入**所有者权益分离**逻辑。

\subsubsection{(1) 现金流双轨制：运营 vs 储备}
\textbf{经营性现金流 (CFO)} $CF_t$ 是衡量当期盈利能力的核心指标，用于奖励函数：
\begin{equation}
\text{CF}_t = \text{Rev}_t(\mathbf{a}_t) - \text{Cost}_{\text{ops}} - \text{Interest}_t - \text{Tax}_t
\end{equation}
其中 $\text{Cost}_{\text{ops}}$ 包含运营支出与现金薪资：
\begin{equation}
\text{Cost}_{\text{ops}} = \text{Ops}_{\text{fixed}} + S_{\text{total}} \cdot (1 - \omega(a_{\text{equity}}))
\end{equation}
注意：此处 $\omega$ 体现了股权对现金薪资的替代作用（PMA条款），直接提升 $CF_t$（分子），从而提升当期奖励。

\textbf{现金储备 (Cash Reserves)} 的更新需包含融资活动，使用\textbf{净现金流}：
\begin{equation}
\text{Cash}_{t+1} = \text{Cash}_t + \text{CF}_t + \underbrace{\Delta D(a_{\text{debt}}) - \text{Repayment}}_{\text{融资活动}}
\end{equation}
\textbf{关键区分}：奖励函数只奖励 $CF_t$（赚来的钱），不奖励 $\Delta D$（借来的钱），彻底杜绝了“借债刷分”的漏洞。

\subsubsection{(2) 价值体系重构：企业价值 vs 所有者权益}
为避免重复计算稀释，我们将价值分解为三个独立变量：
\begin{itemize}
    \item \textbf{特许经营权价值 (Franchise Value, FV)}：即企业整体价值（Enterprise Value）。
    \begin{equation}
    FV_{t+1} = FV_t \times (1 + V_t)
    \end{equation}
    其中 $V_t$ 仍为增值率。
    
    \item \textbf{归属所有者份额 (Owner's Share, $s_t$)}：
    初始 $s_0=1$。每期股权授予 $a_{\text{equity}}$ 会导致永久性稀释：
    \begin{equation}
    s_{t+1} = s_t \times (1 - \omega(a_{\text{equity}}))
    \end{equation}
    
    \item \textbf{债务存量 (Debt, $D_t$)}：
    独立演化：$D_{t+1} = D_t + \Delta D$。
\end{itemize}

\subsubsection{(3) 杠杆率更新}
\begin{equation}
\lambda_{t+1} = \frac{D_{t+1}}{FV_{t+1}}
\end{equation}
此处与 Section 11 保持严格一致，分母为企业价值。

\subsubsection{(4) 配置状态更新}
配置状态 $\mathbf{K}_t$ 仅在动作可变的阶段被更新，否则保持不变（Freeze）：
\begin{equation}
K_{j, t+1} = \begin{cases} 
a_{j, t} & \text{若 } \text{Mutable}(j, \Theta_t) \\
K_{j, t} & \text{否则 (Frozen)}
\end{cases}
\end{equation}
这一机制确保了即使动作 $a_t$ 在下一时刻改变，如果处于冻结期，系统状态也不会响应非法变化；结合 Action Masking，实际上 Agent 只能选择 $a_t = K_t$。

\subsection{4. 环境状态转移 (Environmental Dynamics)}
（保持马尔可夫链驱动逻辑不变）

---
% =================================================================================================
% SECTION 13: REWARD FUNCTION (REDESIGNED)
% =================================================================================================
\section{目标函数与奖励结构}

\subsection{1. 单步奖励函数 $R_t$}
\begin{equation}
R_t = w_{\text{CF}} \cdot \frac{\text{CF}_t}{\sigma_{\text{CF}}} 
+ w_{\text{val}} \cdot V_t 
+ w_{\text{win}} \cdot \mathbb{I}(\Theta_t=\text{Playoff})
- P_{\text{risk}}(\lambda_t)
\end{equation}
\textbf{严格定义}：
\begin{itemize}
    \item $CF_t$：Section 12.3.1 定义的经营性现金流（不含债权本金）；
    \item $\sigma_{\text{CF}}$：\textbf{现金流归一化常数}（Scaling Factor）。取历史财务数据的标准差（如 \$5M），用于消除量级差异；
    \item $V_t$：\textbf{品牌估值增长率}（Valuation Growth Rate）。直接奖励资产增值速度；
    \item $P_{\text{risk}}(\lambda_t)$：统一使用 $\lambda_t = \frac{D_t}{FV_t}$ 计算的风险惩罚项。
\end{itemize}

\subsection{2. 终值函数 (Terminal Value)}
这是模型最关键的修正。为反映“所有者最终能拿走的财富”，我们计算**归属所有者的净权益**：
\begin{equation}
\text{TerminalValue}(S_T) = s_T \times \left( FV_T - D_T \right)
\end{equation}
其中 $s_T\in[0,1]$ 为所有者最终持股比例，$FV_T$ 与 $D_T$ 为终止时刻的特许经营权价值与债务存量，均属于财务状态 $\mathcal{F}_T$ 的分量。

\textbf{逻辑闭环}：
\begin{itemize}
    \item $FV_T - D_T$ 是最终时刻的**总权益价值 (Total Equity)**。
    \item $s_T$ 是经过 $T$ 年连续稀释后的**所有者持股比例**。
    \item 此公式清晰、简洁且完全避免了重复扣减的问题。Agent 必须权衡：是用股权（降低 $s_T$）来换取薪资空间（提升 $Win\% \to FV_T$），还是保留股权（维持 high $s_T$）但承担现金压力。
\end{itemize}


# 最终版：竞技资产状态空间 $\mathcal{R}_t$（按“10个核心变量”精简版）

<!-- 原 LaTeX 的 \begin{comment}...\end{comment} 已转换为 Markdown 文档主体 -->

## 竞技资产状态空间 $\mathcal{R}_t$

### 设计原则

- **可求解性优先**：用离散分桶/低维向量刻画竞技状态，避免球员逐项指标导致维数灾难。
- **决策相关性**：仅保留对“建队/交易/续约/轮换/杠杆承受能力”有直接影响的竞技变量。
- **可从公开数据映射**：每个变量给出明确的计算方式，保证可由赛季数据或滚动窗口统计得到。

### 竞技状态向量的总体形式

令球队在时刻 $t$ 的竞技资产状态为一个由 $10$ 个核心变量组成的向量：

$$
\mathcal{R}_t
=
\Big\{
\mathbf{Q}_t,\;
\mathbf{C}_t,\;
\mathbf{P}_t,\;
\mathbf{L}_t,\;
\mathbf{A}_t,\;
\mathbf{W}_t,\;
\text{ELO}_t,\;
\text{Syn}_t,\;
\mathbf{O}_t,\;
\text{SOS}_t
\Big\}.
$$

其中前 $1$--$5$ 项刻画球员资产结构（可控内生），第 $6$--$8$ 项刻画球队竞技表现与体系（半内生），第 $9$--$10$ 项刻画赛程对抗环境（外生）。

---

#### 1. 球员技术能力聚合向量 $\mathbf{Q}_t$（Team Skill Aggregate）

为避免逐球员逐指标带来高维状态，先定义每名球员 $i$ 的**标准化特征**：

$$
z_{i,k}
=
\frac{x_{i,k}-\mu_k}{\sigma_k},
\qquad k \in \mathcal{K},
$$

其中 $x_{i,k}$ 为球员指标（如 TS\%、AST、TOV、DRTG、BPM 等），$\mu_k,\sigma_k$ 为同赛季联盟均值与标准差。

将球员指标压缩为 $d$ 维**能力因子**（建议 $d=4\sim 6$）：

$$
\mathbf{q}_i
=
\mathbf{W}\mathbf{z}_i
\in \mathbb{R}^{d},
$$

其中 $\mathbf{z}_i=[z_{i,k}]_{k\in\mathcal{K}}$，$\mathbf{W}$ 可由 PCA/因子分析得到，或由专家权重设定。

球队层面的能力聚合向量定义为**上场时间加权平均**：

$$
\mathbf{Q}_t
=
\sum_{i\in \mathcal{R}(t)}
\omega_{i,t}\,\mathbf{q}_i,
\qquad
\omega_{i,t}=\frac{\text{MP}_{i,t}}{\sum_{j\in\mathcal{R}(t)} \text{MP}_{j,t}},
$$

其中 $\mathcal{R}(t)$ 为当期大名单集合，$\text{MP}_{i,t}$ 为球员在滚动窗口（如最近 $g$ 场）内平均上场时间。

**解释（示例）**：可取 $\mathbf{q}_i=[q_{i,\text{Off}},q_{i,\text{Def}},q_{i,\text{Play}},q_{i,\text{Reb}}]$ 分别代表进攻/防守/组织/篮板四个综合因子。

---

#### 2. 球员类型聚类分布 $\mathbf{C}_t$（Roster Archetype Mix）

对联盟球员做聚类得到类型标签 $c_i\in\{1,\dots,K\}$（建议 $K=6\sim 10$）。定义球队类型构成向量为：

$$
\mathbf{C}_t
=
[c_{t,1},\dots,c_{t,K}],
\qquad
c_{t,k}=\sum_{i\in \mathcal{R}(t)} \mathbb{I}(c_i=k).
$$

该向量刻画“阵容原型结构”（例如：空间型射手、护框中锋、持球核心、3D 侧翼等），用于约束阵容搭配与交易方向。

---

#### 3. 位置平衡约束向量 $\mathbf{P}_t$（Positional Balance）

$$
\mathbf{P}_t
=
[p_{\text{PG}},p_{\text{SG}},p_{\text{SF}},p_{\text{PF}},p_{\text{C}}],
\qquad
p_{\text{pos}}
=\sum_{i\in \mathcal{R}(t)}\mathbb{I}(\text{pos}_i=\text{pos}).
$$

**基本约束**：

$$
\sum_{\text{pos}} p_{\text{pos}}=N_{\text{roster}},
\qquad
p_{\text{pos}}\ge 1,\ \forall\ \text{pos}.
$$

（若考虑位置模糊，可用“主位置+副位置”将 $\mathbb{I}(\cdot)$ 改为软计数权重。）

---

#### 4. 合同状态向量 $\mathbf{L}_t$（Contract Maturity Profile）

为避免逐球员合同年限带来高维，采用**到期结构分桶**表示：

$$
\mathbf{L}_t
=
[\ell_{0,t},\ell_{1,t},\ell_{2+,t}],
$$

其中

$$
\ell_{0,t}=\sum_{i}\mathbb{I}(l_i=0),\quad
\ell_{1,t}=\sum_{i}\mathbb{I}(l_i=1),\quad
\ell_{2+,t}=\sum_{i}\mathbb{I}(l_i\ge 2),
$$

$l_i$ 为球员 $i$ 剩余合同年限（年）。

**可选增强**：加入“保障/非保障/选项”比例，但仍保持分桶维度常数。

---

#### 5. 年龄结构向量 $\mathbf{A}_t$（Age Curve Profile）

采用年龄分布的低维表示：

$$
\mathbf{A}_t
=
[a_{\text{mean},t},\ a_{\text{var},t},\ a_{28+,t}],
$$

其中

$$
a_{\text{mean},t}=\frac{1}{N_{\text{roster}}}\sum_{i} a_i,\quad
a_{\text{var},t}=\frac{1}{N_{\text{roster}}}\sum_{i}(a_i-a_{\text{mean},t})^2,\quad
a_{28+,t}=\sum_{i}\mathbb{I}(a_i\ge 28).
$$

该向量用于刻画“即战力 vs 衰退风险”的结构特征，并可在状态转移中引入年龄衰退函数。

---

#### 6. 当前赛季战绩状态 $\mathbf{W}_t$（Performance Record State）

为使状态平稳且可比较，使用滚动窗口（最近 $g$ 场）定义：

$$
\mathbf{W}_t
=
[\text{Win\%}_t^{(g)},\ \text{Streak}_t,\ \text{RankBin}_t],
$$

其中 $\text{Win\%}_t^{(g)}$ 为最近 $g$ 场胜率，$\text{Streak}_t$ 为连胜/连败长度，$\text{RankBin}_t$ 为排名分桶（如：前 1/3、中 1/3、后 1/3）。

---

#### 7. 球队战力评级 $\text{ELO}_t$（Team Strength Rating）

采用 ELO 作为单一强度指标：

$$
\text{ELO}_{t+1}
=
\text{ELO}_t
+ 
K_{\text{elo}}\left(W_{\text{actual},t}-W_{\text{expected},t}\right),
$$

其中 $W_{\text{expected},t}$ 可由 ELO 差异的 logistic 期望胜率给出。

---

#### 8. 阵容协同效应 $\text{Syn}_t$（Roster Synergy）

用“实际阵容净效率”减去“按球员加总的基准净效率”定义协同增益：

$$
\text{Syn}_t
=
\underbrace{\text{NetRTG}_t^{(g)}}_{\text{阵容整体表现}}
-
\underbrace{\sum_{i\in \mathcal{R}(t)}\omega_{i,t}\,\text{BPM}_{i,t}}_{\text{个体贡献线性叠加基准}},
$$

其中 $\text{NetRTG}_t^{(g)}=\text{ORTG}_t^{(g)}-\text{DRTG}_t^{(g)}$ 为最近 $g$ 场的净效率，$\text{BPM}_{i,t}$ 为球员 box plus-minus（同样可用滚动窗口），$\omega_{i,t}$ 为上场时间权重。

**解释**：$\text{Syn}_t>0$ 表示体系加成（“1+1>2”），$\text{Syn}_t<0$ 表示风格冲突或轮换失配。

---

#### 9. 对手强度分布 $\tilde{\mathbf{O}}_t$（Opponent Strength Statistics）

原始赛程长度 $m$ 为变量，为保证状态空间维数固定，定义统计量向量：

$$
\tilde{\mathbf{O}}_t = [\mu_{\text{opp}}, \sigma_{\text{opp}}, \text{N}_{\text{elite}}]
$$

其中 $\mu_{\text{opp}}$ 为剩余对手平均 ELO，$\sigma_{\text{opp}}$ 为标准差，$\text{N}_{\text{elite}}$ 为剩余赛程中强队（ELO $>1600$）的数量。

**注**：为保持状态维度常数，可用统计量替代原始向量：

$$
\tilde{\mathbf{O}}_t=
[\overline{o}_t,\ \text{Var}(o)_t,\ o^{(90\%)}_t],
$$

分别为均值、方差与 90\% 分位数（“强敌尾部风险”）。

---

#### 10. 赛程难度系数 $\text{SOS}_t$（Strength of Schedule）

$$
\text{SOS}_t
=
\frac{1}{m}\sum_{j=1}^{m}\frac{o_j}{\text{ELO}_{\text{avg}}},
$$

其中 $\text{ELO}_{\text{avg}}$ 为联盟平均 ELO。

（可选）若需体现压缩赛程惩罚，可在环境状态中另加背靠背数 $N_{\text{B2B}}$，或将其并入 $\text{SOS}_t$ 的修正项。

---

### 竞技状态空间总结

最终，竞技资产状态空间采用常数维、可计算、可用于动态规划/ADP/MCTS 的表示：

$$
\boxed{
\mathcal{R}_t
=
\Big\{
\mathbf{Q}_t,\;
\mathbf{C}_t,\;
\mathbf{P}_t,\;
\mathbf{L}_t,\;
\mathbf{A}_t,\;
\mathbf{W}_t,\;
\text{ELO}_t,\;
\text{Syn}_t,\;\
\tilde{\mathbf{O}}_t,\;
\text{SOS}_t
\Big\}.
}
$$

**维度说明（示例）**：若 $\mathbf{Q}_t\in\mathbb{R}^4$，$\mathbf{C}_t\in\mathbb{R}^K$（$K=8$），则总体维度约为

$$
4+8+5+3+3+3+1+1+3+1=32,
$$

其中 $\tilde{\mathbf{O}}_t$ 为 3 维统计量，保持状态空间维数恒定，显著提升可求解性。

---

## 建模建议

1. **降维处理**：使用主成分分析（PCA）将技术指标压缩至 5–10 个综合因子。
2. **分层建模**：战术层（单场比赛模拟）$\rightarrow$ 赛季层（82场积分）$\rightarrow$ 多年期（资产演化）。
3. **蒙特卡洛模拟**：对伤病/选秀/交易等随机事件进行 1000+ 次情景推演。
4. **敏感性分析**：识别核心杠杆变量（如顶薪球员健康度、选秀权价值）。

---

## 最终版财务状态空间 $\mathcal{F}_t$（精简核心版）

### 设计原则

- **只保留直接影响决策且无法从其他变量推导的核心变量**
- **明确区分存量（Balance Sheet）与流量（Cash Flow）**
- **与竞技状态 $\mathcal{R}_t$ 建立清晰的因果映射**

### 核心财务状态向量

$$
\mathcal{F}_t = \{\lambda_t, \text{CF}_t, \mathbf{\Psi}_t, \text{Cap}_t, V_t, \text{Equity}_t\}
$$

#### 1. 杠杆率 $\lambda_t \in [0, 1]$

**定义**：

$$
\lambda_t = \frac{D_t}{D_t + E_t}
$$

其中：

- $D_t$：总债务存量（银行贷款、债券等）
- $E_t$：所有者权益（球队市场估值）

**状态分级（用于策略分支）**：

- Safe：$\lambda < 0.3$（可积极投资，融资成本低）
- Balanced：$0.3 \leq \lambda < 0.6$（需谨慎管理）
- Distressed：$\lambda \geq 0.6$（被迫收缩，面临强制出售风险）

**转移机制**：

$$
\lambda_{t+1} = \frac{D_t + \Delta D_t - \text{Repayment}_t}{(D_t + \Delta D_t - \text{Repayment}_t) + (E_t + \Delta E_t)}
$$

其中：

- $\Delta D_t$：当期新增债务（动作变量）
- $\text{Repayment}_t$：债务偿还
- $\Delta E_t$：估值变化（受竞技成绩和媒体周期影响）

**利息成本**：

$$
\text{Interest}_t = D_t \times r_{\text{debt},t}
$$

其中债务利率随杠杆率上升：

$$
r_{\text{debt},t} = r_{\text{risk-free}} + \beta \cdot \lambda_t + \delta \cdot \mathbb{I}(\text{CF}_t < 0)
$$

$\beta \approx 0.05$（杠杆惩罚系数），$\delta \approx 0.02$（流动性危机额外溢价）

**备注**：这是题目要求的 leverage 调整的核心量化指标。

---

#### 2. 经营性现金流 $\text{CF}_t \in \mathbb{R}$

**定义（简化的现金流瀑布）**：

$$
\text{CF}_t = \text{Rev}_t - \text{Cost}_t - \text{Tax}_t - \text{Interest}_t
$$

**收入总额（二元结构）**：

$$
\text{Rev}_t = \underbrace{\text{Rev}_{\text{Performance}}}_{\text{竞技驱动}} + \underbrace{\text{Rev}_{\text{Star}}}_{\text{人气驱动}}
$$

- $\text{Rev}_{\text{Performance}} = \alpha_1 \cdot \text{Win\%}_t + \alpha_2 \cdot \text{Playoff}_t$

  其中 $\alpha_1 \approx 2$M（胜率每提升 10\% 带来 200 万美元收入），$\alpha_2 \in [0.5, 5]$M（季后赛轮次奖金）

- $\text{Rev}_{\text{Star}} = \beta_1 \cdot \text{StarPower}_t + \beta_2 \cdot \text{MediaExposure}_t$

  注：$\text{StarPower}$ 与 $\text{MediaExposure}$ 为基于 $\mathcal{R}_t$ 与 $\mathcal{E}_t$ 的衍生观测变量，不作为独立状态维度。

  其中 $\beta_1 \approx 1.5$M（顶级球星带来 150 万美元额外收入），$\beta_2$（全国转播场次系数）

**成本总额（三大块）**：

$$
\text{Cost}_t = \text{Salary}_{\text{Total}} + \text{Ops}_{\text{Fixed}} + \text{Venue}_t
$$

- $\text{Salary}_{\text{Total}} = \sum_{i=1}^{N_{\text{roster}}} \text{Salary}_i$（所有球员合同总和，直接来自财务状态 $\mathbf{\Psi}_t$）
- $\text{Ops}_{\text{Fixed}}$：固定运营成本（教练、管理层、差旅，约 3–5M 美元/年）
- $\text{Venue}_t$：场馆成本

$$
\text{Venue}_t = \begin{cases}
\text{Rent}_{\text{annual}}, & \text{if 租赁模式} \\
\frac{\text{VenueValue}}{T_{\text{depreciation}}} + \text{Maintenance}_t, & \text{if 自有模式}
\end{cases}
$$

**税收支出（简化）**：

$$
\text{Tax}_t = \max\left(0, (\text{Rev}_t - \text{Cost}_t - \text{Interest}_t) \times r_{\text{corporate}}\right)
$$

其中 $r_{\text{corporate}} \approx 25\%$（联邦+州综合税率）。

**关键阈值**：

- $\text{CF}_t > 0$：健康状态，可积累现金储备
- $\text{CF}_t < 0$：亏损状态，需动用储备或融资
- $\text{CF}_t < -C_{\text{min}}$：触发“强制出售”吸收态（如连续 3 年亏损超 5M）

**现金储备演化**：

$$
\text{Cash}_{t+1} = \text{Cash}_t + \text{CF}_t + \Delta D_t - \text{Repayment}_t - \text{Dividends}_t
$$

**备注**：这是短期生存能力的核心指标，决定球队能否“熬过”重建期。

---

#### 3. 薪资与合同结构向量 $\mathbf{\Psi}_t$（Salary Structure）

$$
\mathbf{\Psi}_t = [\mu_{S,t},\ \sigma_{S,t},\ r_{\text{max},t},\ S_{\text{commit},t}]
$$

- $\mu_{S,t}$（**Salary Level**）：平均薪资水平，作为计算总薪资的基准。
- $\sigma_{S,t}$（**Salary Distribution**）：薪资分布特征（如方差或基尼系数），反映薪资结构的集中度（避免“中产陷阱”）。
- $r_{\text{max},t}$（**Max Salary Ratio**）：顶薪球员薪资占比，衡量头部球星对空间的占用。
- $S_{\text{commit},t}$（**Contract Amount**）：未来保障合同总金额，反映长期财务刚性。

---

#### 4. 工资帽状态 $\text{Cap}_t = \{S_{\text{used}}, S_{\text{avail}}, \text{Tax}_{\text{Status}}\}$

**已用薪资总额**：

$$
S_{\text{used}} = \sum_{i=1}^{N_{\text{roster}}} \text{Salary}_i
$$

**可用薪资空间**：

$$
S_{\text{avail}} = \text{Cap}_{\text{limit}} - S_{\text{used}}
$$

**税收状态（离散化为 3 档）**：

$$
\text{Tax}_{\text{Status}} = \begin{cases}
0 \quad (\text{Under Cap}), & \text{if } S_{\text{used}} < \text{Cap}_{\text{limit}} \\
1 \quad (\text{Taxpayer}), & \text{if } \text{Cap}_{\text{limit}} \leq S_{\text{used}} < \text{Apron}_1 \\
2 \quad (\text{Hard Apron}), & \text{if } S_{\text{used}} \geq \text{Apron}_2
\end{cases}
$$

**奢侈税惩罚（若适用）**：

$$
\text{LuxuryTax}_t = \begin{cases}
0, & \text{if Tax\_Status} = 0 \\
\sum_{k} \text{Bracket}_k \times \text{Rate}_k, & \text{if Tax\_Status} \geq 1
\end{cases}
$$

其中税率为累进制：

- 首个 $\$5M$ 超出部分：税率 1.5 倍（$\$1.50/\$1.00$）
- 次个 $\$5M$：税率 2.0 倍
- 超过 $\$10M$：税率 3.0 倍以上
- 若连续 3 年纳税（Repeater Tax）：税率再乘 1.5 倍

**动作空间约束**：

- Tax\_Status = 0：可全额使用中产特例，可大额交易
- Tax\_Status = 1：部分交易特例失效
- Tax\_Status = 2：禁止打包交易、首轮签交易受限、无中产特例

**工资帽增长（来自环境状态）**：

$$
\text{Cap}_{\text{limit},t+1} = \text{Cap}_{\text{limit},t} \times (1 + g_{\text{cap}})
$$

其中 $g_{\text{cap}} \in \{8\%, 12\%, 40\%\}$（平滑增长 vs 媒体合约跳跃）。

**备注**：这是硬约束，直接限制动作空间 $\mathcal{A}_t$。

---

#### 5. 品牌估值增速 $V_t \in \mathbb{R}$（替代绝对估值）

**定义（相对于上一期的估值变化率，以小数表示）**：

$$
V_t = \frac{E_t - E_{t-1}}{E_{t-1}}
$$

**驱动因素（回归模型）**：

$$
V_t = \gamma_1 \cdot \Delta\text{Win\%} + \gamma_2 \cdot \text{MediaDeal}_{\text{Spike}} + \gamma_3 \cdot \text{StarAcquisition} + \gamma_4 \cdot \text{CF}_t + \epsilon
$$

参数估计（示例）：

- $\gamma_1 \approx 0.05$（胜率提升 10\% 导致估值增长 5\%）
- $\gamma_2 \approx 0.30$（新媒体合约生效导致估值跳跃 30\%）
- $\gamma_3 \approx 0.15$（获得超级球星导致估值增长 15\%）
- $\gamma_4 \approx 0.0002$（每百万美元利润增加 0.02\% 估值）

**状态分级**：

- 高增长（$V > 0.20$）：估值飙升期（如获得 Caitlin Clark）
- 稳定（$0 < V < 0.20$）：正常发展
- 贬值（$V < 0$）：球队价值下滑

**长期估值累积**：

$$
E_t = E_0 \times \prod_{\tau=1}^{t} (1 + V_{\tau})
$$

**备注**：这捕捉题目要求的“特许经营权品牌资产”，用增速而非绝对值避免数值过大。

---

#### 6. 股权结构与球员权益 $\text{Equity}_t = \{\text{Owner}_{\text{share}}, \text{Player}_{\text{equity}}, \text{Dilution}_t\}$

**核心问题**：题目明确提到“球员股权可作为薪资替代方案”（profit participation, equity stake 等）。

**所有者权益占比**：

$$
\text{Owner}_{\text{share},t} = \frac{E_{\text{owner}}}{E_t} \times 100\%
$$

初始状态：$\text{Owner}_{\text{share},0} = 100\%$。

**球员股权池（员工持股计划，ESOP）**：

$$
\text{Player}_{\text{equity},t} = \sum_{i \in \text{Granted}} \omega_i
$$

其中 $\omega_i$ 为球员 $i$ 的持股比例（通常 $0.1\% \sim 5\%$）。

**股权授予条件**：

- **类型 1：收入分成**（Revenue Sharing）

  $$
  \text{Bonus}_i = \text{Rev}_t \times \omega_i \times \mathbb{I}(\text{Rev}_t > \text{Threshold})
  $$

- **类型 2：利润分成**（Profit Participation）

  $$
  \text{Bonus}_i = \max(0, \text{CF}_t) \times \omega_i
  $$

- **类型 3：长期股权**（Equity Stake）

  球员 $i$ 获得球队 $\omega_i\%$ 的永久股权，退役后仍享有：

  - 年度分红权（如球队盈利时按比例分配）
  - 增值收益权（球队出售时按估值分成）

  **估值退出收益**：

  $$
  \text{Payout}_i = (E_T - E_0) \times \omega_i \quad \text{（球队出售时）}
  $$

**股权稀释动态**：

$$
\text{Owner}_{\text{share},t+1} = \text{Owner}_{\text{share},t} \times \frac{E_t}{E_t + \Delta E_{\text{new}}}
$$

当授予新股权或引入外部投资者时发生稀释。

**决策权衡（动作空间中的选择）**：

对于顶薪球员 $i$，所有者可选择：

1. **纯现金合同**：

   $$
   \text{Salary}_i = \$500k/\text{year}, \quad \omega_i = 0\%
   $$

2. **现金+股权混合**：

   $$
   \text{Salary}_i = \$350k/\text{year}, \quad \omega_i = 2\%, \quad \text{期望总价值} \approx \$500k
   $$

   优势：当期现金流压力减小（$\text{CF}_t$ 改善），但未来利润需分享。

3. **股权激励型**（适用于重建期）：

   $$
   \text{Salary}_i = \$250k/\text{year}, \quad \omega_i = 5\%, \quad \text{Vesting Period} = 4\text{年}
   $$

   若球队 4 年内估值从 $\$100M$ 增至 $\$200M$，球员额外获得：

   $$
   \text{Gain}_i = (\$200M - \$100M) \times 5\% = \$5M
   $$

**约束条件**：

- **总股权池上限**：

  $$
  \sum_i \omega_i \leq \omega_{\max} \quad (\omega_{\max} \approx 10\%-15\%)
  $$

  避免过度稀释导致所有者失去控制权。

- **流动性要求**：

  $$
  \text{Cash}_t - \text{Salary}_{\text{Total}} \geq C_{\min}
  $$

- **球员接受度**：

  $$
  \text{Salary}_i + \mathbb{E}[\text{Equity Gain}_i] \geq \text{MarketValue}_i
  $$

**股权退出机制（Liquidity Events）**：

1. **球队出售**：所有股权持有者按比例分享交易对价。
2. **回购条款（Buyback）**：球员退役/交易时，球队有权按公允价值回购股权：

   $$
   \text{Buyback Price} = E_t \times \omega_i \times (1 - \text{Discount})
   $$

   Discount 通常 10–20\%（流动性折扣）。

3. **分红政策**：若球队选择不分红（将利润全部再投资），股权持有者无当期收益，只能等待长期增值。

**财务影响建模**：

- **对现金流的影响**：

  $$
  \text{CF}_t^{\text{with equity}} = \text{CF}_t^{\text{no equity}} + (\text{Salary}_{\text{saved}} - \text{Dividends}_{\text{paid}})
  $$

  - 短期：节省现金工资，$\text{CF}_t$ 改善
  - 长期：需支付分红或面临回购压力，$\text{CF}_t$ 恶化

- **对估值的影响**：

  $$
  E_t^{\text{diluted}} = E_t \times \left(1 - \sum_i \omega_i\right)
  $$

  所有者实际拥有的价值被稀释。

**最优化问题（所有者视角）**：

在给定薪资帽约束下，选择股权授予比例以最大化期望净现值：

$$
\max_{\{\omega_i\}} \mathbb{E}\left[ \sum_{t=0}^{T} \frac{\text{CF}_t \times (1 - \sum_i \omega_i)}{(1+r)^t} + \frac{E_T \times (1 - \sum_i \omega_i)}{(1+r)^T} \right]
$$

约束条件：

$$
\begin{aligned}
&\sum_i \text{Salary}_i \leq \text{Cap}_{\text{limit}} \\
&\sum_i \omega_i \leq \omega_{\max} \\
&\text{Salary}_i + \mathbb{E}[\text{Equity Gain}_i] \geq \text{MarketValue}_i \quad \forall i
\end{aligned}
$$

**备注**：股权激励在初创/重建期球队中特别有效，因为估值增长空间大，球员愿意“赌未来”。

---

### 财务状态的极简表示

$$
\mathcal{F}_t = \begin{cases}
\lambda_t \in [0, 1] & \text{（杠杆率）} \\
\text{CF}_t \in \mathbb{R} & \text{（经营现金流）} \\
\mathbf{\Psi}_t \in \mathbb{R}^4 & \text{（薪资结构：均值/方差/顶薪比/保障金额）} \\
S_{\text{avail}} \in [0, \text{Cap}_{\text{limit}}] & \text{（可用薪资空间）} \\
\text{Tax}_{\text{Status}} \in \{0, 1, 2\} & \text{（税收档位）} \\
V_t \in \mathbb{R} & \text{（品牌估值增速）} \\
\text{Equity}_t \in \mathbb{R}^2 & \text{（所有者权益占比与球员持股池）}
\end{cases}
$$

**总维度**：约 9 维（与 Section 10 统一状态空间保持一致）。

---

## 最终版环境状态空间 $\mathcal{E}_t$（精简核心版）

### 设计原则

- **只保留“外生冲击”和“不可控约束”**
- **宏观经济用单一指标代理**
- **联盟政策聚焦于“阶跃式变化”（如扩军、工资帽暴涨）**

### 核心环境状态向量

$$
\mathcal{E}_t = \{\text{Macro}_t, \text{League}_t, \text{Market}_t, \text{FA}_{\text{Market},t}\}
$$

---

#### 1. 宏观经济景气度 $\text{Macro}_t \in \{\text{衰退}, \text{正常}, \text{繁荣}\}$

**定义（离散化的经济周期指标）**：

- **衰退**：消费者支出下降，门票收入 -20\%，赞助合同难续约
- **正常**：基准状态
- **繁荣**：娱乐消费旺盛，门票溢价 +15\%，新赞助商涌入

**转移概率（马尔可夫链）**：

$$
P(\text{Macro}_{t+1} | \text{Macro}_t) = \begin{bmatrix}
0.7 & 0.25 & 0.05 \\
0.15 & 0.7 & 0.15 \\
0.05 & 0.3 & 0.65
\end{bmatrix}
$$

行顺序：\{衰退, 正常, 繁荣\}

**影响机制**：

$$
\text{Rev}_t = \text{Rev}_{\text{base}} \times \begin{cases}
0.8, & \text{if Macro}_t = \text{衰退} \\
1.0, & \text{if Macro}_t = \text{正常} \\
1.15, & \text{if Macro}_t = \text{繁荣}
\end{cases}
$$

**备注**：用三状态马尔可夫链替代复杂的 GDP/CPI/利率多维建模。

---

#### 2. 联盟政策周期 $\text{League}_t = \{\text{Cap}_{\text{Growth}}, I_{\text{Expansion}}, T_{\text{MediaDeal}}\}$

**工资帽增长模式 Cap\_Growth**：

$$
\text{Cap}_{\text{Growth}} \in \{\text{平滑}, \text{跳跃}\}
$$

- **平滑增长（Smooth）**：每年 $g_{\text{cap}} = 8\%\text{–}12\%$（可预测）
- **跳跃增长（Spike）**：某年突增 $g_{\text{cap}} = 40\%$（2026年 WNBA 新媒体合约）

**工资帽演化**：

$$
\text{Cap}_{\text{limit},t+1} = \text{Cap}_{\text{limit},t} \times (1 + g_{\text{cap}})
$$

**扩军时间表 $I_{\text{Expansion}}$**：

$$
I_{\text{Expansion},t} = \begin{cases}
1, & \text{if } t \in \{2026, 2028, ...\} \\
0, & \text{otherwise}
\end{cases}
$$

**扩军影响（一次性冲击）**：

1. **扩军费分红**：

   $$
   \text{CF}_t = \text{CF}_t + \frac{\text{ExpansionFee}}{N_{\text{teams}}} \quad \text{（一次性+5M美元）}
   $$

2. **人才稀释**：自由市场球员平均技能下降：

   $$
   \mathbb{E}[S_i^{\text{FA}}] = \mathbb{E}[S_i^{\text{FA}}]_{\text{baseline}} \times 0.95
   $$

3. **保护名单要求**：每队需提交保护名单 $P_{\text{Protected}} \subseteq \{1, 2, ..., 12\}$，通常规则：

   $$
   |P_{\text{Protected}}| = 6 \sim 8
   $$

   未保护球员可被新球队选走（每队最多失去 1 名球员）。

**媒体合约倒计时 $T_{\text{MediaDeal}}$**：

$$
T_{\text{MediaDeal}} \in \{5, 4, 3, 2, 1, 0\}
$$

**合约生效触发**：当 $T_{\text{MediaDeal}} = 0$ 时触发“估值跳跃”：

$$
V_t = V_{t-1} + \text{Spike}_{\text{media}} \quad (\text{Spike} \approx 30\%\text{–}50\%)
$$

同时：

$$
\text{Rev}_{\text{media},t} = \text{Rev}_{\text{media},t-1} \times (1 + g_{\text{media}})
$$

其中 $g_{\text{media}} \approx 200\%\text{–}300\%$（2026年 WNBA 新合约预期）。

**倒计时更新**：

$$
T_{\text{MediaDeal},t+1} = \begin{cases}
T_{\text{MediaDeal},t} - 1, & \text{if } T_{\text{MediaDeal},t} > 0 \\
10, & \text{if } T_{\text{MediaDeal},t} = 0 \quad \text{（重置为下个周期）}
\end{cases}
$$

**备注**：这三个子变量共同描述“联盟层面的制度环境”。

---

#### 3. 本队市场位置 $\text{Market}_t$（准静态参数）

**市场规模系数 $\mu_{\text{size}} \in [0.5, 2.0]$**：

- **超大市场**（纽约、洛杉矶）：$\mu_{\text{size}} = 2.0$
- **大市场**（芝加哥、旧金山）：$\mu_{\text{size}} = 1.5$
- **中等市场**（西雅图、凤凰城）：$\mu_{\text{size}} = 1.0$
- **小市场**（印第安纳、康涅狄格）：$\mu_{\text{size}} = 0.6$

**影响公式**：

$$
\text{Rev}_{\text{gate}} = \text{Base} \times \mu_{\text{size}} \times f(\text{Win\%}, \text{StarPower})
$$

**本地竞争强度 $\text{Compete}_{\text{local}} \in \{\text{低}, \text{中}, \text{高}\}$**：

- **低**：本城市无其他主流职业球队（如印第安纳，WNBA 是夏季唯一主角）
- **中**：有 1–2 支竞争球队
- **高**：体育市场高度饱和（纽约 9 支职业球队）

**修正因子**：

$$
\text{MediaShare} = \frac{1}{1 + 0.2 \times N_{\text{competitors}}}
$$

**综合市场效应**：

$$
\text{Rev}_t = \text{Rev}_{\text{base}} \times \mu_{\text{size}} \times \text{MediaShare}
$$

**备注**：$\mu_{\text{size}}$ 和 $\text{Compete}_{\text{local}}$ 在球队建立时确定，不随时间变化（因此严格说是“参数”而非“状态”，但为了建模完整性放在这里）。

---

#### 4. 自由市场热度 $\text{FA}_{\text{Market},t} = \{N_{\text{Star}}, \text{Bidding}_t\}$

**顶级球星供给数量**：

$$
N_{\text{Star}} = |\{i : C_i \in \{\text{核心得分手}, \text{全能侧翼}\} \land l_i = 0\}|
$$

其中 $C_i$ 为球员类型（来自竞技状态 $\mathcal{R}_t$），$l_i$ 为剩余合同年限。

**竞价激烈度（其他球队的薪资空间总和）**：

$$
\text{Bidding}_t = \sum_{j \neq \text{own}} S_{\text{avail},j}
$$

**影响**：

- $N_{\text{Star}}$ 高且 $\text{Bidding}$ 低：买方市场，可低价签约
- $N_{\text{Star}}$ 低且 $\text{Bidding}$ 高：卖方市场，溢价严重

**签约成本函数**：

$$
\text{Contract}_{\text{offer}} = \text{FairValue}_i \times \left(1 + 0.5 \times \frac{\text{Bidding}_t}{\text{Bidding}_{\text{avg}}}\right) \times \left(1 - 0.3 \times \frac{N_{\text{Star}}}{N_{\text{Star,avg}}}\right)
$$

**状态转移**：每个休赛期，自由市场状态重置：

$$
\begin{aligned}
N_{\text{Star},t+1} &= |\{i : l_i = 0 \text{ at end of season } t\}| \\
\text{Bidding}_{t+1} &= \sum_{j} (\text{Cap}_{\text{limit}} - S_{\text{used},j})
\end{aligned}
$$

**备注**：这捕捉“其他球队决策的集体效应”，是博弈论元素的简化表达。

---

### 环境状态的极简表示（最终方案）

$$
\mathcal{E}_t = \begin{cases}
\text{Macro}_t \in \{\text{衰退}, \text{正常}, \text{繁荣}\} & \text{（经济周期）} \\
\text{Cap}_{\text{Growth}} \in \{\text{平滑}, \text{跳跃}\} & \text{（工资帽增长模式）} \\
I_{\text{Expansion}} \in \{0, 1\} & \text{（扩军年标记）} \\
T_{\text{MediaDeal}} \in \{0, 1, 2, ..., 5\} & \text{（媒体合约倒计时）} \\
\mu_{\text{size}} \in [0.5, 2.0] & \text{（市场规模系数）} \\
\text{Compete}_{\text{local}} \in \{\text{低}, \text{中}, \text{高}\} & \text{（本地竞争强度）} \\
N_{\text{Star,FA}} \in \mathbb{Z}_+ & \text{（自由市场球星数）} \\
\text{Bidding}_t \in \mathbb{R}_+ & \text{（竞价强度，百万美元）}
\end{cases}
$$

**总维度**：8 维（相比原方案的 15+ 维大幅精简）。

---

## 状态空间集成设计

### 完整状态向量 $\mathcal{S}_t = \{\mathcal{R}_t, \mathcal{F}_t, \mathcal{E}_t, \Theta_t\}$

- **竞技状态 $\mathcal{R}_t$**：10 维（球员技能、阵容结构、健康度、选秀权）
- **财务状态 $\mathcal{F}_t$**：9 维（杠杆率、现金流、薪资结构、配置状态等）
- **环境状态 $\mathcal{E}_t$**：8 维（经济、联盟政策、市场、自由市场）
- **阶段状态 $\Theta_t$**：1 维（休赛期/常规赛/交易期/季后赛）

**总维度**：约 30 维（满足精确控制需求的同时保持计算可行性）。

---

# MDP 统一状态空间的形式化定义

为解决跨周期的动态决策问题（涵盖赛季内的滚动决策与休赛期的资产重构），并精确响应联盟赛程规则与扩军事件，系统的总体状态空间定义为竞技、财务、环境三个子空间与阶段状态的笛卡尔积：

$$
\mathcal{S} = \mathcal{R} \times \mathcal{F} \times \mathcal{E} \times \Theta
$$

## 1. 赛季阶段状态 $\Theta_t$

该变量充当 MDP 的“决策模式控制器”，决定了模型在当前时刻的可行动作空间 $\mathcal{A}(s_t)$ 及状态转移逻辑：

$$
\Theta_t \in \{\text{Offseason},\ \text{Regular},\ \text{TradeDeadline},\ \text{Playoff}\}
$$

- **Offseason（休赛期）**：决策核心为资产重构。允许选秀（含扩军选秀响应）、自由市场竞价、股权结构调整及长期杠杆规划。
- **Regular（常规赛）**：决策核心为运营优化。允许双向合同签约、票价动态调整、短期伤病特例申请，不可进行核心阵容变更（除非交易期）。
- **TradeDeadline（交易截止日）**：决策核心为博弈。最后调整阵容窗口，面临信息不对称与高溢价风险。
- **Playoff（季后赛）**：决策核心为风险控制。花名册冻结，仅允许战术层面调整与场馆运营优化。

## 2. 竞技资产状态 $\mathcal{R}_t$

采用前文确定的 10 维核心竞技特征向量：

$$
\mathcal{R}_t = \Big\{ \mathbf{Q}_t,\ \mathbf{C}_t,\ \mathbf{P}_t,\ \mathbf{L}_t,\ \mathbf{A}_t,\ \mathbf{W}_t,\ \text{ELO}_t,\ \text{Syn}_t,\ \tilde{\mathbf{O}}_t,\ \text{SOS}_t \Big\}
$$

## 3. 财务状态 $\mathcal{F}_t$

为了解决 MDP 闭合性问题并统一杠杆定义，我们显式引入**配置状态** $\mathbf{K}_t$ 和**所有者权益** $s_t$：

$$
\mathcal{F}_t = \Big\{ FV_t,\ D_t,\ \lambda_t,\ \text{CF}_t,\ \mathbf{\Psi}_t,\ S_{\text{avail},t},\ \text{Tax}_{\text{Status},t},\ V_t,\ s_t,\ \mathbf{K}_t \Big\}
$$

其中关键变量修正定义如下：

- $\lambda_t = \frac{D_t}{FV_t}$：**统一杠杆率**。分母采用特许经营权价值（Franchise Value）$FV_t$，与后续终值函数 $FV_T-D_T$ 保持一致；
- $s_t$：**所有者持股比例**（Owner's Share），取值 $s_t\in[0,1]$，初始 $s_0=1$，随股权激励 $a_{\text{equity}}$ 逐年摊薄；
- $\mathbf{K}_t$：**经营配置状态**（Configuration State）。记录当前生效的票价、营销、分红、薪资和债务档位，用于保证 Action Masking 的马尔可夫性：

  $$
  \mathbf{K}_t = [k_{\text{ticket}},\ k_{\text{marketing}},\ k_{\text{equity}},\ k_{\text{salary}},\ k_{\text{debt}}]
  $$

- $\mathbf{\Psi}_t$：薪资与合同结构向量（含 $\mu_{S,t},\ \sigma_{S,t},\ r_{\text{max},t},\ S_{\text{commit},t}$）；
- $V_t$：本期品牌估值增长率（Valuation Growth Rate）。

## 4. 环境状态 $\mathcal{E}_t$

包含 8 个核心外生变量（含扩军标记 $I_{\text{Expansion}}$，与休赛期阶段 $\Theta_t=\text{Offseason}$ 结合触发扩军选秀逻辑）：

$$
\mathcal{E}_t = \Big\{ \text{Macro}_t,\ \text{Cap}_{\text{Growth}},\ I_{\text{Expansion}},\ T_{\text{MediaDeal}},\ \mu_{\text{size}},\ \text{Compete}_{\text{local}},\ N_{\text{Star,FA}},\ \text{Bidding}_t \Big\}
$$

---

# MDP 动作空间与决策流定义

为了克服传统“概念列表式”动作空间维度可变、稀疏性高的问题，本文将决策变量标准化为**可计算的低维离散动作向量** $\mathbf{a}_t$。这使得问题可直接适配于 DQN 或 PPO 等强化学习算法。

## 1. 统一动作向量定义

在任意时间步 $t$，动作向量定义为 6 维整型向量：

$$
\mathbf{a}_t = \{a_{\text{roster}},\ a_{\text{salary}},\ a_{\text{ticket}},\ a_{\text{marketing}},\ a_{\text{debt}},\ a_{\text{equity}}\}
$$

每个分量对应一个离散化的决策档位，具体定义如下。

## 2. 分量详解

### 1) 竞技构建决策 $a_{\text{roster}}$（Roster Strategy）

决定球队在转会市场、选秀大会或交易截止日的操作方向与力度。

- **空间定义**：$a_{\text{roster}} \in \{0, 1, 2, 3, 4\}$

  - 0（**Seller-Aggressive**）：清仓重建。积极送出老将换取选秀权（$\downarrow \mathbf{Q}_t,\ \uparrow \mathbf{C}_t$）。
  - 1（**Seller-Conservative**）：微调避税。送出特定合同以降低薪资。
  - 2（**Hold**）：保持现状。不进行主动交易操作。
  - 3（**Buyer-Conservative**）：补强。用次轮签或中产特例补充角色球员。
  - 4（**Buyer-Aggressive**）：梭哈争冠。送出未来资产换取即战力巨星（$\uparrow \mathbf{Q}_t,\ \downarrow \mathbf{C}_t$）。

- **阶段约束**：在 $\Theta_t = \text{Playoff}$ 或非交易期，该分量被强制锁定为 2（Hold）。

### 2) 薪资空间管理 $a_{\text{salary}}$（Spending Target）

设定本周期的目标薪资支出水平，间接控制签约动作。

- **空间定义**：$a_{\text{salary}} \in \{0, 1, 2, 3\}$

  - 0（**Floor**）：仅满足穷人线（Cap Min），追求利润最大化。
  - 1（**Over-Cap**）：超过工资帽但低于税线，使用全额中产。
  - 2（**Taxpayer**）：进入奢侈税区间，保留核心阵容。
  - 3（**Apron**）：突破硬工资帽（Apron），不惜重金缴纳罚款打造豪门。

### 3) 票务与商业定价 $a_{\text{ticket}}$（Ticket Pricing）

- **空间定义**：$a_{\text{ticket}} \in \{0, 1, 2, 3, 4\}$，对应倍率 $M_{\text{ticket}} \in \{0.9,\ 1.0,\ 1.1,\ 1.2,\ 1.3\}$
- **通过机制**：票价 = $\text{BasePrice} \times M_{\text{ticket}}$。
- **权衡**：高倍率提升短期 $\text{Rev}_{\text{gate}}$，但降低上座率并长期损害 $\mu_{\text{size}}$（球迷基础）。

### 4) 市场营销投入 $a_{\text{marketing}}$（Marketing Investment）

- **空间定义**：$a_{\text{marketing}} \in \{0, 1, 2\}$

  - 0（**Low**）：维持最低曝光，无额外支出。
  - 1（**Medium**）：投入营收的 5\% 用于推广，平稳提升 $V_t$。
  - 2（**High**）：投入营收的 10\% 进行全美推广（或流媒体合作），显著提升 $V_t$ 和 $\text{MediaExposure}$。

### 5) 债务杠杆调整 $a_{\text{debt}}$（Leverage Control）

- **空间定义**：$a_{\text{debt}} \in \{0, 1, 2\}$

  - 0（**Deleverage**）：优先偿还债务（$\Delta D < 0$）。
  - 1（**Maintain**）：借新还旧，维持当前杠杆率。
  - 2（**Leverage Up**）：激进融资（$\Delta D > 0$），用于球场建设或支付巨额薪资。

### 6) 股权激励授予 $a_{\text{equity}}$（Equity Dilution）

针对题目核心特征（WNBA 球员权益），决定稀释多少股权给予球员。

- **空间定义**：$a_{\text{equity}} \in \{0, 1, 2, 3\}$，对应授予比例 $\omega \in \{0\%,\ 1\%,\ 2\%,\ 5\%\}$
- **作用**：授予越高，$\mathbf{\Psi}_t$ 中的现金薪资压力越小，球员满意度/留队率越高，但所有者剩余价值（Terminal Value）受损。

## 3. 动作有效性与阶段约束（Action Masking & Markov Consistency）

为确保决策的现实可行性且不破坏 Markov 性，我们使用**配置状态** $\mathbf{K}_t$ 作为约束基准，而非依赖上一时刻动作 $a_{t-1}$。只有在当前阶段允许改变的动作分量才可变，其余分量必须强制等于当前配置状态 $\mathbf{K}_t$ 对应的值。

| 动作分量 | Offseason（休赛期） | Regular（常规赛） | TradeDeadline（交易期） | Playoff（季后赛） |
|---|---:|---:|---:|---:|
| $a_{\text{roster}}$ | ✓ | - | ✓ | - |
| $a_{\text{salary}}$ | ✓ | - | ✓ | - |
| $a_{\text{ticket}}$ | ✓ | - | - | - |
| $a_{\text{marketing}}$ | ✓ | ✓ | ✓ | ✓ |
| $a_{\text{debt}}$ | ✓ | - | - | - |
| $a_{\text{equity}}$ | ✓ | ✓ | - | - |

**掩码机制**：

$$
\mathcal{A}_{\text{valid}}(s_t) = \Big\{ \mathbf{a} \in \mathcal{A} \ \Big|\ \forall j,\ \text{Mutable}(j, \Theta_t)\ \lor\ a_j = \mathbf{K}_t[j] \Big\}
$$

这确保了在冻结期（如季后赛），Agent 只能延续上一时刻的状态配置（$a_j = k_j$），消除了对 $a_{t-1}$ 的非马尔可夫依赖。

---

# 系统动力学：状态转移方程

## 1. 赛季阶段演化（Phase Dynamics）

阶段变量 $\Theta_t$ 遵循确定性的事件驱动逻辑：$\Theta_{t+1} = \text{Next}(\Theta_t)$。

## 2. 竞技状态转移（Competitive Dynamics）

（此部分保持不变：$\mathbf{Q}_{t+1}$ 受交易影响，$\mathbf{A}_{t+1}$ 自然老化，$\mathbf{L}_{t+1}$ 合同滚动，战绩受 ELO 驱动。）

## 3. 财务状态转移（Financial Dynamics）

为彻底解决变量定义冲突，本节严格遵循财务状态空间（Section 11）的定义体系，引入**所有者权益分离**逻辑。

### (1) 现金流双轨制：运营 vs 储备

**经营性现金流（CFO）** $CF_t$ 是衡量当期盈利能力的核心指标，用于奖励函数：

$$
\text{CF}_t = \text{Rev}_t(\mathbf{a}_t) - \text{Cost}_{\text{ops}} - \text{Interest}_t - \text{Tax}_t
$$

其中 $\text{Cost}_{\text{ops}}$ 包含运营支出与现金薪资：

$$
\text{Cost}_{\text{ops}} = \text{Ops}_{\text{fixed}} + S_{\text{total}} \cdot (1 - \omega(a_{\text{equity}}))
$$

注意：此处 $\omega$ 体现了股权对现金薪资的替代作用（PMA 条款），直接提升 $\text{CF}_t$（分子），从而提升当期奖励。

**现金储备（Cash Reserves）** 的更新需包含融资活动，使用**净现金流**：

$$
\text{Cash}_{t+1} = \text{Cash}_t + \text{CF}_t + \underbrace{\Delta D(a_{\text{debt}}) - \text{Repayment}}_{\text{融资活动}}
$$

**关键区分**：奖励函数只奖励 $\text{CF}_t$（赚来的钱），不奖励 $\Delta D$（借来的钱），彻底杜绝了“借债刷分”的漏洞。

### (2) 价值体系重构：企业价值 vs 所有者权益

为避免重复计算稀释，我们将价值分解为三个独立变量：

- **特许经营权价值（Franchise Value, FV）**：即企业整体价值（Enterprise Value）。

  $$
  FV_{t+1} = FV_t \times (1 + V_t)
  $$

- **归属所有者份额（Owner's Share, $s_t$）**：初始 $s_0=1$。每期股权授予 $a_{\text{equity}}$ 会导致永久性稀释：

  $$
  s_{t+1} = s_t \times (1 - \omega(a_{\text{equity}}))
  $$

- **债务存量（Debt, $D_t$）**：独立演化：$D_{t+1} = D_t + \Delta D$。

### (3) 杠杆率更新

$$
\lambda_{t+1} = \frac{D_{t+1}}{FV_{t+1}}
$$

此处与 Section 11 保持严格一致，分母为企业价值。

### (4) 配置状态更新

配置状态 $\mathbf{K}_t$ 仅在动作可变的阶段被更新，否则保持不变（Freeze）：

$$
K_{j, t+1} = \begin{cases}
 a_{j, t} & \text{若 } \text{Mutable}(j, \Theta_t) \\
 K_{j, t} & \text{否则（Frozen）}
\end{cases}
$$

这一机制确保了即使动作 $a_t$ 在下一时刻改变，如果处于冻结期，系统状态也不会响应非法变化；结合 Action Masking，实际上 Agent 只能选择 $a_t = K_t$。

## 4. 环境状态转移（Environmental Dynamics）

（保持马尔可夫链驱动逻辑不变。）

---

# 目标函数与奖励结构

## 1. 单步奖励函数 $R_t$

$$
R_t = w_{\text{CF}} \cdot \frac{\text{CF}_t}{\sigma_{\text{CF}}}
+ w_{\text{val}} \cdot V_t
+ w_{\text{win}} \cdot \mathbb{I}(\Theta_t=\text{Playoff})
- P_{\text{risk}}(\lambda_t)
$$

**严格定义**：

- $\text{CF}_t$：Section 12.3.1 定义的经营性现金流（不含债权本金）；
- $\sigma_{\text{CF}}$：**现金流归一化常数**（Scaling Factor）。取历史财务数据的标准差（如 $\$5M$），用于消除量级差异；
- $V_t$：**品牌估值增长率**（Valuation Growth Rate）。直接奖励资产增值速度；
- $P_{\text{risk}}(\lambda_t)$：统一使用 $\lambda_t = \frac{D_t}{FV_t}$ 计算的风险惩罚项。

## 2. 终值函数（Terminal Value）

这是模型最关键的修正。为反映“所有者最终能拿走的财富”，我们计算**归属所有者的净权益**：

$$
\text{TerminalValue}(S_T) = s_T \times \left( FV_T - D_T \right)
$$

其中 $s_T\in[0,1]$ 为所有者最终持股比例，$FV_T$ 与 $D_T$ 为终止时刻的特许经营权价值与债务存量，均属于财务状态 $\mathcal{F}_T$ 的分量。

**逻辑闭环**：

- $FV_T - D_T$ 是最终时刻的**总权益价值（Total Equity）**。
- $s_T$ 是经过 $T$ 年连续稀释后的**所有者持股比例**。
- 此公式清晰、简洁且完全避免了重复扣减的问题。Agent 必须权衡：是用股权（降低 $s_T$）来换取薪资空间（提升 $\text{Win\%} \to FV_T$），还是保留股权（维持 high $s_T$）但承担现金压力。

---

# Q2–Q4 建模与实现（详细版，供论文末尾直接使用）

> 说明：本节与正文状态/动作/奖励定义保持一致，所有实验均基于 **Indiana Fever (IND)**。球员数据来自 **2024 赛季 `allplayers.csv`**；财务部分使用“联赛级回归 + Indiana 校准”的参数（与数据校准章节一致）。

---

## Q2 招募策略（Draft / Free Agency / Trade）

### 1) 球员特征与技能表征

**原始特征**（`allplayers.csv`）：
$$
\{WS/40, TS\%, USG\%, AST\%, TRB\%, DWS, MP\}
$$

**防守信号替代**（避免 DBPM 缺失）：
$$
DWS\_40_i = \frac{DWS_i}{MP_i}\times 40
$$

**标准化**（联盟内 z-score）：
$$
z_{i,k} = \frac{x_{i,k}-\mu_k}{\sigma_k}
$$

**球员综合得分（skill\_score）这个分物种球员类别，先把他们kmeans之后再**（回归拟合权重）：
$$
skill_i = \sum_{k\in \mathcal{K}} w_k\, z_{i,k}
$$


### 2) 怎样衡量球员的能力：

为了更准确地捕捉现代篮球中球员的功能差异，我们对 2023 赛季 WNBA 全体球员的六维技能向量进行了 KMeans ($K=5$) 聚类分析。通过分析各聚类的原始统计特征（Raw Statistics），我们识别出了以下五种典型的球员风格，并将其映射到传统的“位置”标签以适应阵容构建需求：

**表 1：WNBA 球员技能聚类统计表 (Cluster Profiles - Raw Stats)**

| Cluster (Tag) | WS/40 | TS% | USG% | AST% | TRB% | DWS/40 | **风格定义 (Player Archetype)** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Cluster 3 (PG)** | **0.175** | **57.1%** | **25.1%** | **26.8%** | 9.5% | 0.062 | **核心持球大核 (Elite Playmaker)**<br>高使用率与高助攻并存，是球队进攻的绝对发起点与核心得分手。 |
| **Cluster 0 (SG)** | 0.133 | 54.8% | 19.7% | 11.7% | **16.7%** | **0.078** | **攻防全能球星 (Two-Way Star)**<br>拥有极高的防守贡献与篮板保护能力，同时兼具优秀的得分效率，通常是顶级的全能前锋或摇摆人。 |
| **Cluster 1 (PF)** | 0.081 | 54.3% | 16.7% | 11.6% | 8.5% | 0.047 | **优质主力/拼图 (Solid Starter)**<br>各项数据均衡且高效，能够提供稳定的攻防贡献，是强队不可或缺的主力轮换。 |
| **Cluster 4 (SF)** | 0.020 | 49.9% | 17.9% | 18.2% | 5.9% | 0.009 | **外线角色球员 (Perimeter Role)**<br>效率与防守贡献相对平庸，通常负责外线过渡或替补轮转。 |
| **Cluster 2 (C)** | -0.132 | 23.3% | 15.2% | 6.1% | 10.7% | 0.019 | **边缘/低效球员 (Deep Bench)**<br>样本中表现为低效率和负贡献值，通常对应上场时间极少的边缘球员或未适应环境的新秀。 |

该逻辑实现于 `project/data/player_kmeans.py`。虽然使用了这一映射，但我们的模型并未局限于僵化的位置，而是允许通过 `skill_score` 进行跨位置的人才比较，以实现“最佳天赋积累”策略。

### 3) Indiana 阵容构建与缺口识别

- 从 `Team=IND` 选出 12 人 roster；
- 首发：每个 cluster 取 skill\_score 最高球员；
- 替补：全队 skill\_score 高者补齐；
- **位置缺口**按“当前位置平均 skill\_score 最低的优先补强”排序。

### 4) 招募池定义（Draft / FA / Trade）

基于 skill\_score 分位数构建三类候选池（排除 IND 球员）：
$$
\text{Draft}: [0, 45\%],\quad
\text{FA}: [75\%, 100\%],\quad
\text{Trade}: [60\%, 90\%]
$$
分别对应“低成本潜力 / 即战力 / 中高强度补强”。

### 5) 所有者视角价值模型（核心）

**候选替换规则**：若候选位置存在，则替换该位置最低 skill\_score；否则替换全队最低。

竞技增益：
$$
\Delta Q = \overline{Q}_{new}-\overline{Q}_{old}
$$

胜率增益（启发式）：
$$
\Delta Win \approx 0.08\tanh(\Delta Q)
$$

星级判定：
$$
Star_i = \mathbb{I}(skill_i \ge q_{0.85})
$$

成本代理：
$$
Cost_i = c_{pool}\cdot skill_i,\quad
c_{Draft}=0.4,\;c_{FA}=1.2,\;c_{Trade}=0.9
$$

**Owner 价值函数**：
$$
V_i = 1.2\Delta Q + 0.6\cdot Star_i - 0.8\cdot Cost_i
$$


### 6) 权重回归拟合与敏感性分析

**(a) 团队特征构造（分钟加权）**

对每支球队 $T$ 的聚合特征：
$$
X_{T,k} = \sum_{i\in T} w_i\,z_{i,k},\quad
w_i=\frac{MP_i}{\sum_{j\in T} MP_j}
$$

**(b) 目标变量（Win% / NetRtg / ELO）**

- Win% 与 NetRtg 来自 `wnba_advanced_stats.csv` 最近赛季（2023）。  
- 联盟 ELO 不完整，因此使用 **IND 历史 ELO 与 NetRtg/Win% 回归**得到 ELO 代理：
$$
ELO\_{proxy} = 1526.89 + 5.541\cdot NetRtg + 2.237\cdot Win\%
$$

**(c) 多目标回归（Ridge / Lasso）**

令 $Y=[Win\%, NetRtg, ELO\_{proxy}]$，拟合：
$$
\min_{W}\ \|Y - XW\|_2^2 + \alpha\|W\|_2^2 \quad (Ridge)
$$
或
$$
\min_{W}\ \|Y - XW\|_2^2 + \alpha\|W\|_1 \quad (Lasso)
$$

本次数据下 Ridge 的误差略低于 Lasso，因此采用 Ridge 结果并在目标维度上取平均得到权重：
$$
w = [-0.043,\ 0.018,\ 0.020,\ 0.085,\ -0.349,\ -0.485]
$$

注意，这些数值经过了双重标准化（Z-Score Normalization）处理，它们代表的是“相对偏差值”（Standard Deviations from Mean），而不是“原始绝对值”。
原始数据如下：
对应 $[WS/40, TS\%, USG\%, AST\%, TRB\%, DWS\_{40}]$（L1 归一化）。

**(d) 敏感性分析（权重扰动）**

以 $w$ 为中心加入噪声并归一化（200 次）：
- 全体球员 rank 的 Spearman 相关：均值 **0.976**，标准差 **0.020**
- 招募 Top5 重叠率：
  - Draft：**0.878 ± 0.134**
  - Free Agency：**0.700 ± 0.148**
  - Trade：**0.474 ± 0.181**

结果表明：整体排名较稳定，但 Trade 区间对权重更敏感。

### 7) 求解流程与输出

**流程**：
1) 构建 IND 阵容与位置缺口；
2) 生成 Draft/FA/Trade 候选池；
3) 对每名候选执行“替换—评估—打分”；
4) 以 $V_i$ 排序输出招募清单。

**实现脚本**：`project/experiments/q2_recruitment_strategy.py`  
**输出文件**：
- `project/experiments/output/q2_recruitment_strategy.csv`
- `project/experiments/output/q2_recruitment_summary.md`

### 8) 解决问题的方式（对应题意）

Q2 要求“基于联赛标准操作制定招募策略，并从所有者利润角度评估”。该实现将 Draft/FA/Trade 明确分流，并用 $V_i$ 将竞技收益、明星效应与成本风险统一度量，直接输出可执行的招募方案与利弊分析。

---

## Q3 扩军情景与选址敏感性

### 1) 扩军冲击如何“写入状态”（环境层）与为什么要这样写

本题的关键是：\textbf{扩军并不是一个“连续微扰”，而是一个在赛季边界发生的离散外生冲击}。因此我们将扩军冲击作为环境状态 $\mathcal{E}_t$ 的一次性跳变，并要求该跳变\textbf{发生在进入扩军赛季的 Offseason 之前}，使总经理能在休赛期决策时“提前看到”冲击。

代码实现位置：`project/mdp/transitions_env.py` 的 `env_transition(...)`，并在 `project/mdp/env.py` 中以“先推进阶段/年份，再更新环境”的方式调用，使 $\Theta_{t+1}=\text{Offseason}$ 时的环境冲击可被下一步决策直接观测。

#### (a) 扩军选址三元组（情景参数）

我们用一个三元组对“新球队选址”进行参数化（代码：`project/experiments/q3_expansion_site_sensitivity.py`）：
$$
(\Delta_{market},\ \Delta_{compete},\ media\_bonus)
$$
解释如下：
\begin{itemize}
    \item $\Delta_{market}$：扩军对 Indiana 市场规模因子 $\mu_{\text{size}}$ 的净影响（可能为负：市场被分流；也可能为正：联盟关注度提升带来外溢）。
    \item $\Delta_{compete}$：扩军对 Indiana 本地竞争强度 $\text{Compete}_{local}$ 的影响（越大表示“附近新增竞争者/同类娱乐分流”越强）。
    \item $media\_bonus$：扩军年媒体合作/转播关注带来的“当期媒体收入+品牌增长”提升幅度。
\end{itemize}

#### (b) 一次性环境跳变（扩军年进入 Offseason 时触发）

当进入扩军年份 $y\in\mathcal{Y}_{expansion}$ 且阶段切换到 $\Theta_{t+1}=\text{Offseason}$ 时，环境状态在代码中执行一次性更新：
$$
\mu_{t+1}=\text{clip}(\mu_t+\Delta_{market}),\qquad
compete_{t+1}=\text{clip}(compete_t+\Delta_{compete})
$$
其中：
\begin{itemize}
    \item $\mu_t$ 对应代码中的 `EnvState.mu_size`，并在财务转移中作为门票收入的市场放大因子；
    \item $compete_t$ 对应 `EnvState.compete_local`，在财务转移中削弱门票需求（见下文）。
\end{itemize}

同时我们将扩军对自由市场的两条典型机制“定量化”写入：
$$
N_{\text{Star,FA},t+1}=\max\{0,\ N_{\text{Star,FA},t}+\Delta_{\text{Star,FA}}\}+\varepsilon_{FA},
$$
$$
\text{Bidding}_{t+1}=\max\{0,\ \text{Bidding}_t+\Delta_{\text{bidding}}\}+\varepsilon_{bid},
$$
解释如下：
\begin{itemize}
    \item $N_{\text{Star,FA}}$（代码 `EnvState.n_star_fa`）表示“当期自由市场可获得的顶级球员数量（稀缺度）”，扩军使其减少（默认 $\Delta_{\text{Star,FA}}=-2$）。
    \item $\text{Bidding}$（代码 `EnvState.bidding_intensity`）表示“竞价强度/溢价水平”，扩军使其上升（默认 $\Delta_{\text{bidding}}=+2$）。
    \item 噪声项：$\varepsilon_{FA}\sim\text{round}(\mathcal{N}(0,1))$，$\varepsilon_{bid}\sim\mathcal{N}(0,0.5)$，用于刻画扩军年自由市场供给与竞价的不确定性。
\end{itemize}

#### (c) 扩军冲击的“具体数值”（Indiana 的基准与各选址对比）

由数据校准得到 Indiana 的基准市场因子约为 $\mu_0\approx 0.615$（对应 `calibrate_config_from_data()`），且初始 $\text{Compete}_{local}=0$，$N_{\text{Star,FA}}=3$，$\text{Bidding}=5$。各选址情景的“确定性部分”变化为：
\begin{itemize}
    \item Toronto：$\Delta_{market}=+0.02,\ \Delta_{compete}=0$，因此 $\mu\approx 0.635$；
    \item Denver：$\Delta_{market}=-0.02,\ \Delta_{compete}=1$，因此 $\mu\approx 0.595,\ compete=1$；
    \item Nashville：$\Delta_{market}=-0.03,\ \Delta_{compete}=1$，因此 $\mu\approx 0.585,\ compete=1$；
    \item SanDiego：$\Delta_{market}=-0.01,\ \Delta_{compete}=0$，因此 $\mu\approx 0.605$。
\end{itemize}
并且扩军年会使 $N_{\text{Star,FA}}:3\to 1$（再叠加随机波动）、$\text{Bidding}:5\to 7$（再叠加随机波动）。

### 2) “自由市场供给减少/竞价强度上升”到底影响了哪些变量（竞技层传播链）

扩军并不会直接修改 Indiana 的胜率，而是通过“阵容更新的边际效率”传导至竞技状态 $\mathcal{R}_t$。该机制在 `project/mdp/transitions_comp.py` 的 `roster_update(...)` 中以\textbf{可计算的乘法因子}落地：
$$
\Delta \mathbf{Q}\ \propto\ \Delta_{\text{roster}}(a_{\text{roster}})
\cdot f_{\text{salary}}(a_{\text{salary}})
\cdot f_{\text{FA}}(N_{\text{Star,FA}})
\cdot f_{\text{bid}}(\text{Bidding}).
$$
其中两条与扩军直接相关的因子是：
$$
f_{\text{FA}}(N)=1+0.05\cdot \max(0,\ N-2),
$$
$$
f_{\text{bid}}(B)=1-0.02\cdot \max(0,\ B-5).
$$
解释：
\begin{itemize}
    \item 当扩军导致 $N_{\text{Star,FA}}$ 从 3 降到 1 时，$f_{\text{FA}}$ 从 $1.05$ 变为 $1.00$，表示同样的 roster/salary 操作“更难买到”同等质量的补强；
    \item 当扩军导致竞价强度从 5 升到 7 时，$f_{\text{bid}}$ 从 $1.00$ 变为 $0.96$，表示竞价溢价使补强效率下降；
    \item 二者合并会使“同一动作带来的 $\Delta \mathbf{Q}$”相对基准下降约 $0.96/1.05\approx 0.914$（约 8.6\% 的边际效率损失）。
\end{itemize}

进一步，$\mathbf{Q}_t$ 会进入赛季内胜率更新（`project/mdp/transitions_comp.py: game_sim_update`）：
$$
P(\text{win})=\sigma\big(\eta_0+\eta_1\Delta ELO-\eta_{SOS}(SOS-1)+\eta_2 Syn+\eta_3\tanh(\overline{Q})\big),
$$
因此扩军对 $\Delta\mathbf{Q}$ 的压缩会传导为更低的赢球概率与更慢的 ELO 进化。

### 3) “市场/竞争/媒体 bonus”到底影响了哪些变量（财务层传播链）

#### (a) 门票收入：$\mu_{\text{size}}$ 与 $compete_{local}$ 的作用位置

财务转移在 `project/mdp/transitions_fin.py` 中。门票收入（gate revenue）的核心结构为：
$$
Rev_{\text{gate}}=\text{BaseGate}\cdot f_{\text{perf}}(Win\%)\cdot f_{\text{star}}(\overline{Q})
\cdot f_{\text{macro}}(\text{Macro})\cdot \underbrace{\mu_{\text{size}}\cdot f_{\text{compete}}(compete_{local})}_{\text{扩军选址影响入口}}
\cdot f_{\text{ticket}}(a_{\text{ticket}})\cdot f_{\text{mkt}}(a_{\text{marketing}}).
$$
其中本地竞争因子在代码中定义为：
$$
f_{\text{compete}}=\text{clip}\big(1-\beta_{\text{compete}}\cdot compete_{local}\big),\qquad \beta_{\text{compete}}=0.04,
$$
解释：新增本地竞争者会削弱 Indiana 的门票需求/注意力份额；该削弱通过 $Rev_{\text{gate}}$ 进入经营现金流 $CF_t$。

#### (b) 薪资成本：竞价强度如何抬升成本

扩军会提高竞价强度 $\text{Bidding}$，从而抬升给定薪资档位下的真实支付成本（代码同样在 `transitions_fin.py`）：
$$
\text{Payroll}=\text{Cap}\cdot M_{\text{salary}}(a_{\text{salary}})\cdot f_{\text{roster}}(a_{\text{roster}})
\cdot\Big(1+\beta_{\text{bid}}\cdot\max(0,\text{Bidding}-B_0)\Big),
$$
其中 $\beta_{\text{bid}}=0.03$，$B_0=5$ 为基准竞价水平。解释：当 $\text{Bidding}$ 从 5 升到 7 时，Payroll 额外增加约 $1+0.03\cdot 2=1.06$（约 +6\%），使“高薪抢星”在扩军年更难转化为正的经营利润。

#### (c) 媒体 bonus：收入与估值增长的双通道注入

扩军年媒体 bonus 在财务转移中通过两条渠道进入：
\begin{itemize}
    \item 媒体收入提升：若 $I_{\text{Expansion}}=1$（或媒体周期事件发生），则
    $$
    Rev_{\text{media}}=\text{BaseMedia}\cdot f_{\text{macro}}\cdot (1+media\_bonus).
    $$
    \item 估值增长提升：估值增长率 $V_t$ 在扩军年额外增加一项
    $$
    V_t\leftarrow V_t+0.5\cdot media\_bonus,
    $$
    表示扩军带来的品牌关注度提升（但幅度保守，低于一次完整媒体周期的 spike）。
\end{itemize}

### 4) 扩军年 Indiana 应如何从初始策略调整（必须“拿数据说话”的定量对比）

为避免“只说结论不说依据”，我们在同样初始化设置下比较四种策略，并固定扩军年份为 2026（只触发一次扩军），在 3 个赛季（12 个 phase step）后比较关键指标：
\begin{itemize}
    \item Baseline：保持默认固定策略（mask-aware + nearest-valid）。
    \item Aggressive：高薪抢星 + 激进补强 + 加杠杆（动作目标为 $(6,5,6,3,4,0)$；若触发风险约束则自动投影到最近合法动作）。
    \item Defensive：现金流稳定 + 阵容稳定 + 适度去杠杆（动作目标为 $(3,1,2,3,1,0)$）。
    \item PPO：在相同 Action Mask 与风险约束下训练得到的自适应策略。
\end{itemize}

代码与输出位置：
\begin{itemize}
    \item 实验脚本：`project/experiments/q3_expansion_policy_comparison.py`
    \item 输出：`project/experiments/output/q3_policy_comparison_summary.csv`（赛季末均值/方差）
    \item 关键动作统计：`project/experiments/output/q3_policy_comparison_detail.csv`（含扩军年 Offseason 的动作均值）
\end{itemize}

以“第 3 赛季末（Season 3）”为例，各选址下 PPO 相对 Baseline 的提升（均为 6 个 seed 平均）：
\begin{itemize}
    \item Denver：$\Delta CF_{\text{cum}}\approx +28.0$，$\Delta Terminal\approx +100.1$，并将债务从约 67 降至 0（去杠杆），胜率变化约 $-0.016$；
    \item Nashville：$\Delta CF_{\text{cum}}\approx +20.1$，$\Delta Terminal\approx +74.0$，债务由 67 降至 0，胜率变化约 $-0.040$；
    \item SanDiego：$\Delta CF_{\text{cum}}\approx +27.8$，$\Delta Terminal\approx +100.3$，债务由 67 降至 0，胜率变化约 $-0.009$；
    \item Toronto：$\Delta CF_{\text{cum}}\approx +26.4$，$\Delta Terminal\approx +101.8$，债务由 67 降至 0，胜率变化约 $-0.005$。
\end{itemize}

对比“高杠杆 + 高薪抢星”的 Aggressive 路径，其在所有选址下都显著恶化终值（例如 Denver 下终值约从 431 降至 296，且债务升至约 179），并伴随更低的累计现金流与更差的竞技结果。这一结果的机制解释是：扩军导致的 $N_{\text{Star,FA}}\downarrow$ 与 $\text{Bidding}\uparrow$ 使“高薪补强”的边际效率下降，同时 Payroll 上升与利息/风险惩罚压缩利润，最终出现“花更多钱但买不到同比例胜率提升”的现象。

#### (a) “降低杠杆冲动”在代码中如何量化？

我们并非凭空给出“扩军年应降低杠杆冲动”的口号，而是通过扩军冲击改变“同一杠杆动作的净收益”，并观察到 PPO 在扩军年 Offseason 的债务动作均值明显低于基线：
\begin{itemize}
    \item Baseline 在扩军年（2026 Offseason）维持 $a_{\text{debt}}\approx 2$（Maintain）；
    \item PPO 在扩军年 Offseason 的均值约为 $a_{\text{debt}}\approx 1.0\sim 1.2$（Deleverage），同时薪资档位保持在中等水平（$a_{\text{salary}}\approx 2.3\sim 3.0$）而非极端压缩；
    \item 该统计由 `q3_policy_comparison_detail.csv` 中 `marker=offseason_decision` 的记录直接给出，可复现。
\end{itemize}

### 5) 新球队选址如何影响模型与最终策略（敏感性结论）

选址通过 $(\Delta_{market},\Delta_{compete},media\_bonus)$ 进入环境与财务转移，导致：
\begin{itemize}
    \item 直接影响门票收入上限（$\mu_{\text{size}}$）与竞争稀释（$compete_{local}$）；
    \item 间接影响最优 roster/salary 操作强度（因为补强边际效率和 Payroll 结构改变）；
    \item 媒体 bonus 同时抬升当期媒体收入与估值增长，可能抵消部分门票稀释。
\end{itemize}
在我们的情景集里，Toronto（$\Delta_{market}>0$ 且 $\Delta_{compete}=0$）对 Indiana 的长期终值最有利；Nashville（$\Delta_{market}<0$ 且 $\Delta_{compete}>0$）更不利。

### 6) 哪些球队所有者受益/受损（联盟层代理指标与可解释性）

题目还要求指出“扩军选址下，哪些球队受益/受损”。由于我们缺乏每支球队的完整财务报表与交易细节，本节采用一个\textbf{可解释的联盟层代理评分}（实现：`project/experiments/q3_expansion_site_sensitivity.py:_league_impact`）：

市场规模指数（attendance proxy）：
$$
\mu_i = \frac{AvgAttendance_i}{LeagueAvg},
$$
解释：$\mu_i>1$ 表示该队拥有更大的球迷基础，因而更能放大媒体红利。

强度指数（player-based ELO proxy）：
$$
s_i = \frac{ELO_i - 1500}{100},
$$
解释：$s_i$ 越高表示“强队/明星多”，在稀缺度上升时更不易被抽空。

综合影响分数（越高越受益）：
$$
Impact_i = 0.6\,media\_bonus\,\mu_i
+ \Delta_{market}\mu_i
- 0.05\Delta_{compete}(1-\mu_i)
- LocalPenalty
- 0.02\,Scarcity(1-s_i),
$$
其中：
\begin{itemize}
    \item $LocalPenalty$：若扩军城市与某队地理/市场重叠，则该队被额外扣分（例如 Nashville 的本地队列包含 IND/ATL/WAS）；
    \item $Scarcity=|\Delta_{\text{Star,FA}}|$：扩军导致自由市场稀缺度上升的强度；
    \item 最后一项表示“弱队在稀缺冲击下更难通过自由市场补强”，因此更不利。
\end{itemize}

该代理指标可以直接输出每个选址的 Top3/Bottom3 受益/受损球队，并且每个项都能由 attendance 与 player-based strength 解释，避免“只给结论”的问题。

---

## Q4 额外业务决策（票价 / 股权）

### 1) 单维策略控制（严格 Action Mask）

为保证“只讨论票价或股权”，定义单维动作集：
$$
\mathcal{A}_{mode}(s)=\{a\mid a_j=K_j,\ j\neq target\}
$$
- **ticket 模式**：仅 $a_{ticket}$ 可变
- **equity 模式**：仅 $a_{equity}$ 可变，并强制股权上限

### 2) 策略学习与评估

- **Baseline**：保持当前合法动作（nearest valid）。
- **Learned**：PPO（线性策略）在单维动作空间内训练。

PPO 参数（与代码一致）：
- steps\_per\_update = 128
- epochs = 2
- 训练轮数 = 12
- 评估 episode = 8

### 3) 输出与作用

实现脚本：`project/experiments/q4_dynamic_ticket_or_equity.py`  
输出文件：
- `project/experiments/output/q4_dynamic_policy_summary_ticket.md`
- `project/experiments/output/q4_dynamic_policy_summary_equity.md`

该结果用于回答“票价最优策略”或“股权激励合理水平”。

---

## 复现命令（附录）
```
python project/experiments/q2_recruitment_strategy.py
python project/experiments/q3_expansion_site_sensitivity.py
python project/experiments/q3_expansion_policy_comparison.py
python project/experiments/q4_dynamic_ticket_or_equity.py --mode ticket
python project/experiments/q4_dynamic_ticket_or_equity.py --mode equity
```
