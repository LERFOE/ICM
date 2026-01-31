import json
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT
OUT_DIR = Path(__file__).resolve().parent / "clean"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _to_float(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "na", "none"}:
        return np.nan
    s = s.replace(",", "")
    s = s.replace("+", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def _parse_salary_cell(cell: str) -> Tuple[float, float, str]:
    """Return salary_m (millions), pct (0-1), status."""
    if pd.isna(cell):
        return np.nan, np.nan, ""
    s = str(cell).strip()
    if s == "":
        return np.nan, np.nan, ""

    status = ""
    if s.upper() in {"UFA", "RFA"}:
        return np.nan, np.nan, s.upper()

    # Extract percent
    pct_match = re.search(r"([0-9]+(?:\.[0-9]+)?)%", s)
    pct = float(pct_match.group(1)) / 100.0 if pct_match else np.nan

    # Extract dollar amount
    money_match = re.search(r"\$?([0-9,]+(?:\.[0-9]+)?)", s)
    if money_match:
        salary = float(money_match.group(1).replace(",", "")) / 1e6
    else:
        salary = np.nan

    return salary, pct, status


def clean_spotrac():
    path = DATA_DIR / "spotrac_ind_yearly_raw.csv"
    df = pd.read_csv(path)

    df = df.rename(columns={"Player (16)": "Player"})
    df["Age"] = df["Age"].apply(_to_float)

    # Parse year columns
    year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c))]
    for col in year_cols:
        salaries = []
        pcts = []
        statuses = []
        for val in df[col].values:
            salary, pct, status = _parse_salary_cell(val)
            salaries.append(salary)
            pcts.append(pct)
            statuses.append(status)
        df[f"{col}_salary_m"] = salaries
        df[f"{col}_pct"] = pcts
        df[f"{col}_status"] = statuses

    # Drop raw year columns to avoid confusion
    df = df.drop(columns=year_cols)

    df.to_csv(OUT_DIR / "spotrac_ind_yearly_clean.csv", index=False)
    return df


def clean_ind_ind():
    path = DATA_DIR / "IND_IND.csv"
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

    numeric_cols = ["W", "L", "W/L%", "Finish", "SRS", "Pace", "ORtg", "DRtg"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(_to_float)
    if "Year" in df.columns:
        df["Year"] = df["Year"].apply(_to_float).astype("Int64")

    df.to_csv(OUT_DIR / "ind_team_season_stats.csv", index=False)
    return df


def clean_master_wide():
    path = DATA_DIR / "IND_master_wide.csv"
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains(r"^IND__Unnamed")]
    if "Season" in df.columns:
        df["Season"] = df["Season"].apply(_to_float).astype("Int64")

    # Convert numeric columns with IND__ prefix
    for col in df.columns:
        if col.startswith("IND__"):
            if any(k in col for k in ["W", "L", "W/L%", "Finish", "SRS", "Pace", "ORtg", "DRtg"]):
                df[col] = df[col].apply(_to_float)

    df.to_csv(OUT_DIR / "ind_master_wide_clean.csv", index=False)
    return df


def clean_attendance():
    path = DATA_DIR / "wnba_attendance.csv"
    df = pd.read_csv(path)
    if "Season" in df.columns:
        df["Season"] = df["Season"].apply(_to_float).astype("Int64")
    for col in ["GP", "Total_Attendance", "Avg_Attendance"]:
        if col in df.columns:
            df[col] = df[col].apply(_to_float)
    df.to_csv(OUT_DIR / "wnba_attendance_clean.csv", index=False)
    return df


def clean_advanced_stats():
    path = DATA_DIR / "wnba_advanced_stats.csv"
    df = pd.read_csv(path)
    if "Season" in df.columns:
        df["Season"] = df["Season"].apply(_to_float).astype("Int64")
    for col in ["W", "L", "Win%", "ORtg", "DRtg", "NetRtg", "Pace", "SRS"]:
        if col in df.columns:
            df[col] = df[col].apply(_to_float)
    df.to_csv(OUT_DIR / "wnba_advanced_stats_clean.csv", index=False)
    return df


def clean_elo_game():
    path = DATA_DIR / "IND_ELO_O_SOS_game_level.csv"
    df = pd.read_csv(path)
    if "Season" in df.columns:
        df["Season"] = df["Season"].apply(_to_float).astype("Int64")
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    numeric_cols = [
        "home_pts",
        "away_pts",
        "home_win",
        "ELO_t",
        "opp_pre_elo_t",
        "O_mean",
        "O_var",
        "O_p90",
        "SOS_t",
        "elo_avg_pre",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(_to_float)

    if "O_t_list_json" in df.columns:
        def _parse_list(val):
            if pd.isna(val):
                return []
            try:
                data = json.loads(val)
                return data if isinstance(data, list) else []
            except Exception:
                return []
        o_lists = df["O_t_list_json"].apply(_parse_list)
        df["O_len"] = o_lists.apply(len)
        df["O_mean_calc"] = o_lists.apply(lambda x: float(np.mean(x)) if len(x) else np.nan)
        df["O_var_calc"] = o_lists.apply(lambda x: float(np.var(x)) if len(x) else np.nan)
        df["O_p90_calc"] = o_lists.apply(lambda x: float(np.percentile(x, 90)) if len(x) else np.nan)

    df.to_csv(OUT_DIR / "ind_elo_game_clean.csv", index=False)
    return df


def clean_elo_season():
    path = DATA_DIR / "IND_ELO_O_SOS_season_level.csv"
    df = pd.read_csv(path)
    if "Season" in df.columns:
        df["Season"] = df["Season"].apply(_to_float).astype("Int64")
    for col in ["ELO_last_pregame", "SOS_mean", "O_mean_mean", "games"]:
        if col in df.columns:
            df[col] = df[col].apply(_to_float)
    df.to_csv(OUT_DIR / "ind_elo_season_clean.csv", index=False)
    return df


def clean_valuations():
    path = DATA_DIR / "wnba_valuations.csv"
    df = pd.read_csv(path)
    if "Year" in df.columns:
        df["Year"] = df["Year"].apply(_to_float).astype("Int64")
    for col in ["Valuation_M", "Revenue_M"]:
        if col in df.columns:
            df[col] = df[col].apply(_to_float)
    df.to_csv(OUT_DIR / "wnba_valuations_clean.csv", index=False)
    return df


def audit(df: pd.DataFrame, name: str) -> str:
    lines = []
    lines.append(f"## {name}")
    lines.append(f"shape: {df.shape}")
    lines.append("columns:")
    lines.append(", ".join(df.columns))

    missing = df.isna().mean().sort_values(ascending=False)
    top_missing = missing[missing > 0]
    if len(top_missing):
        lines.append("missing_ratio:")
        for col, ratio in top_missing.head(10).items():
            lines.append(f"- {col}: {ratio:.2f}")
    else:
        lines.append("missing_ratio: none")

    return "\n".join(lines)


def main():
    datasets: Dict[str, pd.DataFrame] = {}
    datasets["spotrac_ind_yearly_clean.csv"] = clean_spotrac()
    datasets["ind_team_season_stats.csv"] = clean_ind_ind()
    datasets["ind_master_wide_clean.csv"] = clean_master_wide()
    datasets["wnba_attendance_clean.csv"] = clean_attendance()
    datasets["wnba_advanced_stats_clean.csv"] = clean_advanced_stats()
    datasets["ind_elo_game_clean.csv"] = clean_elo_game()
    datasets["ind_elo_season_clean.csv"] = clean_elo_season()
    datasets["wnba_valuations_clean.csv"] = clean_valuations()

    audit_lines = ["# Data Audit (Cleaned)"]
    for name, df in datasets.items():
        audit_lines.append(audit(df, name))
        audit_lines.append("")

    audit_path = OUT_DIR / "data_audit.md"
    audit_path.write_text("\n".join(audit_lines))
    print(f"Wrote cleaned files to {OUT_DIR}")
    print(f"Wrote audit to {audit_path}")


if __name__ == "__main__":
    main()
