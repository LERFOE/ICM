from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = Path(__file__).resolve().parent / "clean"


def load_clean_csv(name: str) -> pd.DataFrame:
    path = CLEAN_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Cleaned file not found: {path}")
    return pd.read_csv(path)


def load_spotrac() -> pd.DataFrame:
    return load_clean_csv("spotrac_ind_yearly_clean.csv")


def load_ind_team_stats() -> pd.DataFrame:
    return load_clean_csv("ind_team_season_stats.csv")


def load_ind_master_wide() -> pd.DataFrame:
    return load_clean_csv("ind_master_wide_clean.csv")


def load_attendance() -> pd.DataFrame:
    return load_clean_csv("wnba_attendance_clean.csv")


def load_advanced_stats() -> pd.DataFrame:
    return load_clean_csv("wnba_advanced_stats_clean.csv")


def load_elo_game() -> pd.DataFrame:
    return load_clean_csv("ind_elo_game_clean.csv")


def load_elo_season() -> pd.DataFrame:
    return load_clean_csv("ind_elo_season_clean.csv")


def load_valuations() -> pd.DataFrame:
    return load_clean_csv("wnba_valuations_clean.csv")
