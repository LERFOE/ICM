import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT_MD = Path("project/experiments/output/q5_letter.md")
Q1_PATH = Path("project/experiments/output/q1_leverage_policy_map.csv")
Q2_PATH = Path("project/experiments/output/q2_recruitment_strategy.csv")
Q3_PATH = Path("project/experiments/output/q3_expansion_sensitivity.csv")
Q4_PATH = Path("project/experiments/output/q4_dynamic_policy_summary.md")


def _load_csv(path: Path):
    if not path.exists():
        return []
    with path.open("r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def summarize_q1(rows):
    if not rows:
        return "杠杆策略：暂无结果（未生成 Q1 输出）。"
    by_macro = {"0": [], "1": [], "2": []}
    for r in rows:
        by_macro[str(r["macro"])].append(int(r["a_debt"]))
    summary = []
    labels = {0: "衰退", 1: "正常", 2: "繁荣"}
    for macro, acts in by_macro.items():
        if not acts:
            continue
        avg = sum(acts) / len(acts)
        if avg < 0.7:
            rec = "偏向去杠杆"
        elif avg < 1.3:
            rec = "保持杠杆"
        else:
            rec = "适度加杠杆"
        summary.append(f"宏观{labels[int(macro)]}期：{rec}（债务动作均值≈{avg:.2f}）")
    return "；".join(summary)


def summarize_q2(rows):
    if not rows:
        return "招募策略：暂无结果（未生成 Q2 输出）。"
    # rows are sorted in file; take top 3
    top = rows[:3]
    parts = []
    for r in top:
        parts.append(
            f"{r['roster_label']} + {r['salary_label']}（终值≈{float(r['mean_terminal']):.2f}）"
        )
    return "优先组合：" + "；".join(parts)


def summarize_q3(rows):
    if not rows:
        return "扩军敏感性：暂无结果（未生成 Q3 输出）。"
    worst = rows[-1]
    best = rows[0]
    return (
        f"最有利选址：{best['site']}（Δ终值≈{float(best['delta_terminal']):.2f}）；"
        f"最不利选址：{worst['site']}（Δ终值≈{float(worst['delta_terminal']):.2f}）。"
    )


def summarize_q4():
    if not Q4_PATH.exists():
        return "票价/股权策略：暂无结果（未生成 Q4 输出）。"
    # Use file as-is
    return "票价/股权策略已生成（见 Q4 输出）。"


def main():
    q1 = summarize_q1(_load_csv(Q1_PATH))
    q2 = summarize_q2(_load_csv(Q2_PATH))
    q3 = summarize_q3(_load_csv(Q3_PATH))
    q4 = summarize_q4()

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_MD.open("w") as f:
        f.write("# 致球队老板与总经理的建议信\n\n")
        f.write("尊敬的老板与总经理：\n\n")
        f.write("基于我们构建的阶段约束 MDP 模型（竞技-财务-环境联合转移 + 动作冻结机制），我们形成了以下可执行建议，目标是在下一赛季及以后同时优化竞技表现与财务健康。\n\n")
        f.write("## 1. 杠杆与资本结构建议（Q1）\n")
        f.write(f"{q1}\n\n")
        f.write("要点：在经济衰退期避免过度杠杆，优先维持现金流与偿债能力；在繁荣期可适度融资支持设施或核心球员续约，但需保持杠杆率低于风险阈值。\n\n")
        f.write("## 2. 下赛季招募与阵容构建（Q2）\n")
        f.write(f"{q2}\n\n")
        f.write("要点：招募策略需同时考虑战绩提升与合同刚性风险。高薪梭哈可带来短期胜率与收入提升，但会放大奢侈税与协同惩罚；稳健补强可在维持现金流的同时改善战绩。\n\n")
        f.write("## 3. 联盟扩军情景与选址影响（Q3）\n")
        f.write(f"{q3}\n\n")
        f.write("要点：扩军会提升联盟整体曝光但稀释自由市场供给与本地市场份额。若扩军选址接近我队市场或为高度竞争城市，应提前降低薪资刚性并提高现金储备，以应对收入波动与竞价强度上升。\n\n")
        f.write("## 4. 额外业务决策（Q4）\n")
        f.write(f"{q4}\n\n")
        f.write("要点：票价和股权政策应被视为跨季节的长期决策。高票价提升短期门票收入，但可能削弱球迷基础；股权激励可减轻现金压力，但所有者稀释不可逆。\n\n")
        f.write("## 5. 关键风险与应对（含伤病冲击）\n")
        f.write("- 伤病冲击：模型已将核心球员伤病作为竞技状态的随机冲击处理。建议在常规赛保持一定薪资与阵容弹性，避免过度锁定长期合同，以便在伤病发生时快速调整。\n")
        f.write("- 财务风险：保持杠杆率低于阈值，避免因短期融资推高破产概率。\n")
        f.write("- 稀释风险：股权激励仅适用于对球队胜率与品牌有重大提升的球员群体，且需设定年度上限。\n\n")
        f.write("## 6. 行动清单（落地到 6 维动作向量）\n")
        f.write("- 休赛期：重点调整 a_roster / a_salary / a_debt / a_equity，完成资产重构并设定赛季财务边界。\n")
        f.write("- 常规赛：保持阵容冻结，侧重 a_marketing 与运营优化，谨慎处理股权激励。\n")
        f.write("- 交易期：视战绩与财务窗口进行有限度补强或变现。\n\n")
        f.write("我们建议每个赛季滚动运行该模型，并在宏观经济、联盟规则或扩军环境发生变化时重新校准参数。该方案兼顾竞技目标与资产增值，并为老板与总经理提供可执行的决策依据。\n\n")
        f.write("此致\n\n模型团队\n")

    print(f"Saved letter to {OUTPUT_MD}")


if __name__ == "__main__":
    main()
