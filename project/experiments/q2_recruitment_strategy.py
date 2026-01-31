import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.data.player_kmeans import build_player_model, competitive_state_from_lineup
from project.experiments.utils import build_env


OUTPUT_CSV = Path("project/experiments/output/q2_recruitment_strategy.csv")
OUTPUT_MD = Path("project/experiments/output/q2_recruitment_summary.md")


def _position_needs(roster: pd.DataFrame) -> List[str]:
    pos_order = ["PG", "SG", "SF", "PF", "C"]
    needs = []
    for pos in pos_order:
        subset = roster[roster["Position"] == pos]
        if subset.empty:
            needs.append((pos, -1e9))
        else:
            needs.append((pos, float(subset["skill_score"].mean())))
    needs.sort(key=lambda x: x[1])
    return [p for p, _ in needs]


def _replace_player(roster: pd.DataFrame, candidate: pd.Series) -> pd.DataFrame:
    roster = roster.copy()
    pos = candidate.get("Position", None)
    if pos in roster["Position"].values:
        idx = roster[roster["Position"] == pos]["skill_score"].idxmin()
    else:
        idx = roster["skill_score"].idxmin()
    roster.loc[idx] = candidate
    return roster


def _evaluate_candidate(
    roster: pd.DataFrame, candidate: pd.Series, star_thresh: float, cost_mult: float
) -> Dict[str, float]:
    Q_old, _, _ = competitive_state_from_lineup(roster)
    roster_new = _replace_player(roster, candidate)
    Q_new, _, _ = competitive_state_from_lineup(roster_new)
    delta_q = float(np.mean(Q_new) - np.mean(Q_old))
    win_gain = 0.08 * np.tanh(delta_q)
    is_star = 1.0 if float(candidate["skill_score"]) >= star_thresh else 0.0
    cost = cost_mult * float(candidate["skill_score"])
    owner_value = 1.2 * delta_q + 0.6 * is_star - 0.8 * cost
    return {
        "delta_q": delta_q,
        "win_gain": win_gain,
        "star": is_star,
        "cost": cost,
        "owner_value": owner_value,
    }


def _pool_select(df: pd.DataFrame, pos_needs: List[str], quant_low: float, quant_high: float) -> pd.DataFrame:
    q_low = df["skill_score"].quantile(quant_low)
    q_high = df["skill_score"].quantile(quant_high)
    pool = df[(df["skill_score"] >= q_low) & (df["skill_score"] <= q_high)].copy()
    # prioritize needed positions
    pool["need_priority"] = pool["Position"].apply(lambda p: pos_needs.index(p) if p in pos_needs else 99)
    pool = pool.sort_values(["need_priority", "skill_score"], ascending=[True, False])
    return pool


def _write_summary(md_path: Path, sections: Dict[str, List[Dict]]):
    with md_path.open("w") as f:
        f.write("# Q2 Recruitment Strategy (Draft / FA / Trade)\n\n")
        f.write("基于2024赛季真实球员数据（`allplayers.csv`），构建Indiana Fever的阵容缺口与招募优先级。\n\n")
        for title, rows in sections.items():
            f.write(f"## {title}\n")
            if not rows:
                f.write("无可用候选。\n\n")
                continue
            for i, r in enumerate(rows[:5], 1):
                f.write(
                    f"{i}. {r['Player']} ({r['Team']}, {r['Position']}) | "
                    f"value={r['owner_value']:.3f} | ΔQ={r['delta_q']:.3f} | "
                    f"win_gain≈{r['win_gain']:.3f}\n"
                )
            f.write("\n")
        f.write("## 业务利弊\n")
        f.write("- Draft：成本低、风险分散，但短期胜率提升有限。\n")
        f.write("- 自由市场：短期提升明显，但工资与税线压力大。\n")
        f.write("- 交易：可精准补强，但资产消耗与阵容协同风险较高。\n")


def main():
    env = build_env(use_data=True, seed=42)
    cfg = env.config
    model = build_player_model(roster_size=cfg.roster_size)

    team_code = cfg.team_code
    roster = model.build_roster(team=team_code, roster_size=cfg.roster_size)
    if roster.empty:
        raise RuntimeError("Failed to build Indiana roster from player data.")

    # Attach Position if missing
    if "Position" not in roster.columns:
        roster = roster.copy()
        roster["Position"] = roster["cluster"].map(model.cluster_to_position).fillna("F")

    pos_needs = _position_needs(roster)
    star_thresh = float(model.df["skill_score"].quantile(0.85))

    # Candidate pools (heuristic)
    players = model.df.copy()
    players = players[~players["Team"].astype(str).str.contains(team_code, na=False)]
    players["Position"] = players["cluster"].map(model.cluster_to_position).fillna("F")

    pools = {
        "Draft（低成本/长期潜力）": _pool_select(players, pos_needs, 0.00, 0.45),
        "Free Agency（即战力）": _pool_select(players, pos_needs, 0.75, 1.00),
        "Trade（中高强度补强）": _pool_select(players, pos_needs, 0.60, 0.90),
    }
    cost_mult = {
        "Draft（低成本/长期潜力）": 0.4,
        "Free Agency（即战力）": 1.2,
        "Trade（中高强度补强）": 0.9,
    }

    rows = []
    sections: Dict[str, List[Dict]] = {}
    for name, pool in pools.items():
        scored: List[Dict] = []
        for _, cand in pool.head(40).iterrows():
            metrics = _evaluate_candidate(roster, cand, star_thresh, cost_mult[name])
            row = {
                "pool": name,
                "Player": cand["Player"],
                "Team": cand["Team"],
                "Position": cand["Position"],
                "skill_score": float(cand["skill_score"]),
                **metrics,
            }
            scored.append(row)
            rows.append(row)
        scored.sort(key=lambda r: r["owner_value"], reverse=True)
        sections[name] = scored

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "pool",
                "Player",
                "Team",
                "Position",
                "skill_score",
                "delta_q",
                "win_gain",
                "star",
                "cost",
                "owner_value",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r["pool"],
                    r["Player"],
                    r["Team"],
                    r["Position"],
                    round(r["skill_score"], 3),
                    round(r["delta_q"], 3),
                    round(r["win_gain"], 3),
                    int(r["star"]),
                    round(r["cost"], 3),
                    round(r["owner_value"], 3),
                ]
            )

    _write_summary(OUTPUT_MD, sections)

    print(f"Saved recruitment results to {OUTPUT_CSV} and {OUTPUT_MD}")


if __name__ == "__main__":
    main()
