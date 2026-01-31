from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path("allplayers.csv")
OUTPUT = Path("project/data/wnba_2023_skill_vector.csv")

REQUIRED = ["Player", "Team", "WS/40", "TS%", "USG%", "AST%", "TRB%", "DWS", "MP"]


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT}")

    df = pd.read_csv(INPUT)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Derive DWS_40 from DWS and MP (defensive win shares per 40 minutes)
    mp = df["MP"].replace(0, np.nan).astype(float)
    df["DWS_40"] = df["DWS"].astype(float) / mp * 40.0

    out_cols = ["Player", "Team", "WS/40", "TS%", "USG%", "AST%", "TRB%", "DWS_40"]
    out = df[out_cols].dropna().copy()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT, index=False)
    print(f"Wrote {OUTPUT} ({len(out)} rows)")


if __name__ == "__main__":
    main()
