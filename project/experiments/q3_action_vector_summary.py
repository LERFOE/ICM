import csv
from pathlib import Path
import pandas as pd


INPUT = Path("project/experiments/output/q3_policy_comparison_detail.csv")
OUTPUT = Path("project/experiments/output/q3_action_vector_summary.csv")


def _mode(series: pd.Series) -> int:
    if series.empty:
        return 0
    counts = series.value_counts()
    # tie-breaker: smaller action index
    top = counts[counts == counts.max()].index.tolist()
    return int(sorted(top)[0])


def main():
    if not INPUT.exists():
        raise SystemExit(f"Missing input: {INPUT}")
    df = pd.read_csv(INPUT)
    df = df[(df["marker"] == "offseason_decision") & (df["year"] == 2026)].copy()
    if df.empty:
        raise SystemExit("No expansion-year offseason records found.")

    actions = ["a_roster", "a_salary", "a_ticket", "a_marketing", "a_debt", "a_equity"]
    rows = []
    for (site, policy), sub in df.groupby(["site", "policy"]):
        row = {"site": site, "policy": policy, "n": len(sub)}
        means = []
        for a in actions:
            mean_val = float(sub[a].mean())
            mode_val = _mode(sub[a])
            row[f"{a}_mean"] = round(mean_val, 3)
            row[f"{a}_mode"] = int(mode_val)
            means.append(mean_val)
        row["action_vector_mean"] = "(" + ", ".join(f"{m:.2f}" for m in means) + ")"
        rows.append(row)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved action vector summary to {OUTPUT}")


if __name__ == "__main__":
    main()
