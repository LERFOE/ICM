from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import svd


DEFAULT_SKILL_FILE_CANDIDATES = [
    # Prefer the raw allplayers table because it includes `Pos`/`MP` (needed for
    # position fallback and roster/contract logic). We z-score internally.
    Path("allplayers.csv"),
    Path("wnba_2023_skill_vector.csv"),
    Path("project/data/wnba_2023_skill_vector.csv"),
    Path("project/data/clean/wnba_2023_skill_vector.csv"),
]

FEATURES = ["WS/40", "TS%", "USG%", "AST%", "TRB%", "DWS_40"]

# Prefer explicit 5-position labels when available. This file is authored in the
# workspace and provides a `pos`/`Pos5` column that maps each player into
# {PG,SG,SF,PF,C}.
POSITION_LABEL_FILE_CANDIDATES = [
    Path("players_with_position_scores.xlsx"),
    Path("players_with_position_scores.csv"),
    Path("副本players_with_position_scores.xlsx"),
    Path("project/data/players_with_position_scores.xlsx"),
    Path("project/data/players_with_position_scores.csv"),
]

# Position-specific skill coefficients (global z-scores of the metrics).
# Columns: TS%, USG%, AST%, TRB%, DWS_40 + intercept.
POS_SKILL_FEATURES = ["TS%", "USG%", "AST%", "TRB%", "DWS_40"]
POS_SKILL_COEFS = {
    "PG": {
        "TS%": 0.9008671242509894,
        "USG%": 0.40756460796412836,
        "AST%": 0.24717441934183307,
        "TRB%": -0.09648838687418784,
        "DWS_40": 0.6689871162882965,
        "intercept": -0.23055264421285782,
    },
    "SG": {
        "TS%": 0.6688038013398157,
        "USG%": 0.40077434060153266,
        "AST%": 0.22978475669950094,
        "TRB%": 0.5710248435273881,
        "DWS_40": 0.6391548425167576,
        "intercept": 0.10998784379143325,
    },
    "SF": {
        "TS%": 0.5616816855093115,
        "USG%": 0.13946503868177368,
        "AST%": 0.20592666321068887,
        "TRB%": 0.5215194331071136,
        "DWS_40": 0.6355697175003144,
        "intercept": -0.0017023490973049752,
    },
    "PF": {
        "TS%": 0.862690858962707,
        "USG%": 0.32980035691846504,
        "AST%": 0.21651299797958964,
        "TRB%": 0.0052714881470029545,
        "DWS_40": 0.9148345116037043,
        "intercept": -0.00578164029869202,
    },
    "C": {
        "TS%": 0.5935323393302042,
        "USG%": 0.04857135754404065,
        "AST%": 0.4739198176983939,
        "TRB%": 0.559347058479791,
        "DWS_40": 0.6267665466322905,
        "intercept": -0.13958131617902808,
    },
}
DEFAULT_SKILL_WEIGHTS = {
    "WS/40": 0.35,
    "TS%": 0.20,
    "USG%": 0.10,
    "AST%": 0.15,
    "TRB%": 0.10,
    "DWS_40": 0.10,
}
WEIGHT_FILE_CANDIDATES = [
    Path("project/data/skill_weights.json"),
    Path("project/data/clean/skill_weights.json"),
]


@dataclass
class PlayerModel:
    df: pd.DataFrame
    scaler: Dict[str, np.ndarray]
    kmeans: Dict[str, np.ndarray]
    features: List[str]
    cluster_to_position: Dict[int, str]
    cluster_profiles: pd.DataFrame
    cluster_q_targets: Dict[int, np.ndarray]
    age_map: Dict[str, float]
    contract_map: Dict[str, int]
    team_rosters: Dict[str, pd.DataFrame] = field(default_factory=dict)
    team_strengths: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def lineup_for_team(self, team: Optional[str] = None) -> pd.DataFrame:
        df = self.df
        if team:
            team_df = df[df["Team"].astype(str).str.contains(team, na=False)]
            if not team_df.empty:
                df = team_df

        lineup_rows = []
        used = set()
        for cluster, pos in self.cluster_to_position.items():
            cluster_df = df[df["cluster"] == cluster]
            if cluster_df.empty:
                # fallback to team-only pool (still real players)
                cluster_df = df
            pick = cluster_df[~cluster_df["pid"].isin(used)].sort_values("skill_score", ascending=False).head(1)
            if pick.empty:
                continue
            row = pick.iloc[0].to_dict()
            used.add(int(row["pid"]))
            row["Position"] = pos
            lineup_rows.append(row)

        # If any positions missing, fill with best remaining team players
        if team and len(lineup_rows) < 5:
            remaining = df[~df["pid"].isin(used)].sort_values("skill_score", ascending=False)
            for _, row in remaining.iterrows():
                if len(lineup_rows) >= 5:
                    break
                r = row.to_dict()
                r["Position"] = self.cluster_to_position.get(int(r.get("cluster", -1)), "F")
                lineup_rows.append(r)
        return pd.DataFrame(lineup_rows)

    def build_roster(self, team: Optional[str] = None, roster_size: int = 12, rng=None) -> pd.DataFrame:
        rng = rng or np.random.default_rng(42)
        starters = self.lineup_for_team(team)
        used = set(starters["pid"].tolist()) if not starters.empty else set()

        pool = self.df
        if team:
            team_df = self.df[self.df["Team"].astype(str).str.contains(team, na=False)]
            if len(team_df) >= roster_size:
                pool = team_df

        # Fill bench with highest skill_score not used
        bench_needed = max(0, roster_size - len(starters))
        bench = pool[~pool["pid"].isin(used)].sort_values("skill_score", ascending=False).head(bench_needed)
        if not bench.empty and "Position" not in bench.columns:
            bench = bench.copy()
            bench["Position"] = bench["cluster"].map(self.cluster_to_position).fillna("F")
        roster = pd.concat([starters, bench], ignore_index=True)
        return roster

    def build_league_rosters(self, roster_size: int = 12) -> None:
        teams = sorted(self.df["Team"].dropna().astype(str).unique().tolist())
        team_rosters: Dict[str, pd.DataFrame] = {}
        for team in teams:
            roster = self.build_roster(team=team, roster_size=roster_size)
            team_rosters[team] = roster
        self.team_rosters = team_rosters
        self.team_strengths = _compute_team_strengths(self, team_rosters)

    def sample_opponent(self, team_code: Optional[str], rng=None) -> Dict[str, float]:
        rng = rng or np.random.default_rng(42)
        if not self.team_strengths:
            self.build_league_rosters()
        teams = list(self.team_strengths.keys())
        if team_code:
            teams = [t for t in teams if t != team_code]
        if not teams:
            teams = list(self.team_strengths.keys())
        opp = rng.choice(teams)
        out = dict(self.team_strengths[opp])
        out["team"] = opp
        return out

    def compute_state_from_roster(self, roster: pd.DataFrame, base_state) -> object:
        Q, C, P = competitive_state_from_lineup(roster)
        base_state.Q = Q
        # pad C to expected length
        if len(C) < len(base_state.C):
            C_pad = np.zeros_like(base_state.C)
            C_pad[: len(C)] = C
            base_state.C = C_pad
        else:
            base_state.C = C[: len(base_state.C)]
        base_state.P = P

        # Age profile from spotrac map if available
        ages = []
        for name in roster["Player"].astype(str).values:
            if name in self.age_map:
                ages.append(self.age_map[name])
        if ages:
            base_state.A = np.array([float(np.mean(ages)), float(np.var(ages)), float(np.sum(np.array(ages) >= 28))])

        # Contract maturity from spotrac map if available
        if self.contract_map:
            maturities = []
            for name in roster["Player"].astype(str).values:
                if name in self.contract_map:
                    maturities.append(self.contract_map[name])
            if maturities:
                l0 = sum(1 for m in maturities if m <= 1)
                l1 = sum(1 for m in maturities if m == 2)
                l2 = sum(1 for m in maturities if m >= 3)
                base_state.L = np.array([float(l0), float(l1), float(l2)])

        # Synergy based on usage dispersion
        if "USG%" in roster.columns:
            usg = roster["USG%"].astype(float)
            base_state.Syn = -0.5 * float(np.std(usg)) + 0.1 * (0.20 - float(np.mean(usg)))

        base_state.roster_ids = roster["pid"].astype(int).tolist()
        return base_state

    def roster_update(self, R, action, rng, config):
        rng = rng or np.random.default_rng(42)
        R_next = R.copy()

        # Current roster
        if R.roster_ids:
            current = self.df[self.df["pid"].isin(R.roster_ids)]
        else:
            current = self.build_roster(roster_size=config.roster_size, rng=rng)

        # Action intensities
        action_q = [0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 0.90]
        salary_adj = [-0.15, -0.05, 0.0, 0.05, 0.10, 0.15]
        q = action_q[min(action.a_roster, len(action_q) - 1)]
        q += salary_adj[min(action.a_salary, len(salary_adj) - 1)]
        q = float(np.clip(q, 0.10, 0.95))

        replace_rate = [0.6, 0.45, 0.25, 0.0, 0.2, 0.4, 0.6][min(action.a_roster, 6)]

        # Build new starters by position cluster
        new_players = []
        used = set()
        for cluster, pos in self.cluster_to_position.items():
            cluster_pool = self.df[self.df["cluster"] == cluster].sort_values("skill_score")
            if cluster_pool.empty:
                continue
            idx = int(q * (len(cluster_pool) - 1))
            pick = cluster_pool.iloc[idx]
            used.add(int(pick["pid"]))
            row = pick.to_dict()
            row["Position"] = pos
            new_players.append(row)

        starters = pd.DataFrame(new_players)

        # Merge with current roster and adjust by action
        roster = current.copy()
        if not starters.empty:
            # Replace top/bottom players depending on action
            roster = roster.sort_values("skill_score", ascending=True)
            replace_n = int(replace_rate * config.roster_size)
            if action.a_roster >= 4:
                # replace weakest with stronger
                roster = roster.iloc[replace_n:]
            elif action.a_roster <= 2:
                # replace strongest to shed talent
                roster = roster.iloc[: max(1, len(roster) - replace_n)]

            roster = pd.concat([roster, starters], ignore_index=True)
            roster = roster.drop_duplicates(subset=["pid"])

        # Fill to roster size with nearest skill quantile
        if len(roster) < config.roster_size:
            pool = self.df[~self.df["pid"].isin(roster["pid"])]
            pool = pool.sort_values("skill_score")
            idx = int(q * (len(pool) - 1))
            bench = pool.iloc[: idx + 1].tail(config.roster_size - len(roster))
            roster = pd.concat([roster, bench], ignore_index=True)
        if "Position" not in roster.columns:
            roster = roster.copy()
            roster["Position"] = roster["cluster"].map(self.cluster_to_position).fillna("F")
        roster = roster.sort_values("skill_score", ascending=False).head(config.roster_size)

        # Update state from roster
        R_next = self.compute_state_from_roster(roster, R_next)

        # Synergy penalty if aggressive changes
        change_intensity = abs(config.roster_delta[min(action.a_roster, len(config.roster_delta) - 1)])
        R_next.Syn -= config.syn_penalty * change_intensity
        return R_next



def find_player_file() -> Optional[Path]:
    for path in DEFAULT_SKILL_FILE_CANDIDATES:
        if path.exists():
            return path
    return None


def load_player_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize column names
    if "Player" not in df.columns:
        raise ValueError("Player column not found in player dataset")
    # Ensure DWS_40 exists (derive from DWS and MP if needed)
    if "DWS_40" not in df.columns:
        if "DWS" in df.columns and "MP" in df.columns:
            df["DWS_40"] = df["DWS"].astype(float) / df["MP"].replace(0, np.nan).astype(float) * 40.0
        else:
            raise ValueError("Missing DWS_40; please provide DWS and MP to derive it.")
    # Ensure required features exist
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    # Remove rows with NaNs in feature set
    df = df.dropna(subset=FEATURES).copy()
    return df


def load_skill_weights() -> Dict[str, float]:
    for path in WEIGHT_FILE_CANDIDATES:
        if path.exists():
            try:
                import json

                data = json.loads(path.read_text())
                weights = data.get("weights", data)
                if all(k in weights for k in FEATURES):
                    return {k: float(weights[k]) for k in FEATURES}
            except Exception:
                continue
    return DEFAULT_SKILL_WEIGHTS.copy()


def _load_position_labels() -> pd.DataFrame:
    """
    Load explicit 5-position labels (PG/SG/SF/PF/C) if available.

    Expected columns:
      - Player, Team
      - pos OR Pos5 (preferred)
    """
    for path in POSITION_LABEL_FILE_CANDIDATES:
        if not path.exists():
            continue
        try:
            if path.suffix.lower() in (".xlsx", ".xls"):
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path)
        except Exception:
            continue

        pos_col = None
        for c in ("pos", "Pos5", "pos5", "POS5"):
            if c in df.columns:
                pos_col = c
                break
        if pos_col is None:
            continue
        if "Player" not in df.columns or "Team" not in df.columns:
            continue

        out = df[["Player", "Team", pos_col]].copy()
        out = out.rename(columns={pos_col: "Pos5"})
        out["Player"] = out["Player"].astype(str).str.strip()
        out["Team"] = out["Team"].astype(str).str.strip()
        out["Pos5"] = out["Pos5"].astype(str).str.strip().str.upper()
        out = out[out["Pos5"].isin(["PG", "SG", "SF", "PF", "C"])]
        if not out.empty:
            return out

    return pd.DataFrame(columns=["Player", "Team", "Pos5"])


def _infer_pos5_from_raw(pos_raw: str, row_z: pd.Series) -> str:
    """
    Fallback mapper for players missing Pos5 in the label file.

    Uses the original allplayers `Pos` (often G/F/C or hybrids) and standardized
    stats to split into 5 positions. This is only used for a small tail of
    players not covered by the position-score file.
    """
    s = str(pos_raw).upper()
    if not s or s in {"NAN", "NONE"}:
        # No categorical hint: infer from z-feature profile only.
        try:
            ast = float(row_z.get("AST%", 0.0))
            trb = float(row_z.get("TRB%", 0.0))
            dws = float(row_z.get("DWS_40", 0.0))
            usg = float(row_z.get("USG%", 0.0))
        except Exception:
            return "SF"
        frontcourt = 0.6 * trb + 0.4 * dws
        if frontcourt >= 1.0:
            return "C" if trb >= 1.2 else "PF"
        if ast >= 0.6:
            return "PG"
        if usg >= 0.4:
            return "SG"
        return "SF"
    if "C" in s:
        return "C"
    if "G" in s:
        # playmaking proxy: higher AST% -> PG; otherwise SG
        try:
            return "PG" if float(row_z.get("AST%", 0.0)) >= 0.0 else "SG"
        except Exception:
            return "SG"
    if "F" in s:
        # frontcourt proxy: higher reb/def -> PF; otherwise SF
        try:
            score = 0.5 * float(row_z.get("TRB%", 0.0)) + 0.5 * float(row_z.get("DWS_40", 0.0))
            return "PF" if score >= 0.0 else "SF"
        except Exception:
            return "SF"
    # default wing
    return "SF"


def _pos_skill_score(df_z: pd.DataFrame, position_col: str = "Position") -> pd.Series:
    """
    Position-conditioned skill score using POS_SKILL_COEFS on global z-scores.

    Notes:
      - df_z must contain z-scored columns for POS_SKILL_FEATURES.
      - if a player's position is missing/unknown, fall back to SF weights.
    """
    scores = []
    for _, row in df_z.iterrows():
        pos = str(row.get(position_col, "")).upper()
        w = POS_SKILL_COEFS.get(pos) or POS_SKILL_COEFS["SF"]
        val = float(w["intercept"])
        for f in POS_SKILL_FEATURES:
            val += float(w[f]) * float(row.get(f, 0.0))
        scores.append(val)
    return pd.Series(scores, index=df_z.index, dtype=float)


def standardize(X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    std = np.where(std == 0, 1.0, std)
    Xs = (X - mean) / std
    return Xs, {"mean": mean, "std": std}


def simple_kmeans(X: np.ndarray, n_clusters: int = 5, seed: int = 42, iters: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n < n_clusters:
        raise ValueError("Not enough samples for kmeans")
    centroids = X[rng.choice(n, size=n_clusters, replace=False)]
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        # assign
        dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        # update
        for k in range(n_clusters):
            if np.any(labels == k):
                centroids[k] = X[labels == k].mean(axis=0)
    return labels, centroids


def pca_2d(X: np.ndarray) -> np.ndarray:
    # X should be standardized
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


def _assign_positions(cluster_profiles: pd.DataFrame) -> Dict[int, str]:
    # Heuristic mapping of clusters to positions based on feature dominance
    clusters = cluster_profiles.index.tolist()
    profile = cluster_profiles

    # Highest AST -> PG
    pg = profile["AST%"].idxmax()
    remaining = [c for c in clusters if c != pg]

    # Highest USG among remaining -> SG
    sg = profile.loc[remaining, "USG%"].idxmax()
    remaining = [c for c in remaining if c != sg]

    # Highest TRB -> C
    c_pos = profile.loc[remaining, "TRB%"].idxmax()
    remaining = [c for c in remaining if c != c_pos]

    # Highest defensive win shares (per 40) -> PF
    pf = profile.loc[remaining, "DWS_40"].idxmax()
    remaining = [c for c in remaining if c != pf]

    # Remaining -> SF
    sf = remaining[0]

    return {pg: "PG", sg: "SG", sf: "SF", pf: "PF", c_pos: "C"}


def build_player_model(
    path: Optional[Path] = None,
    n_clusters: int = 5,
    seed: int = 42,
    roster_size: int = 12,
    use_position_labels: bool = True,
) -> PlayerModel:
    if path is None:
        path = find_player_file()
    if path is None:
        raise FileNotFoundError(
            "Player skill dataset not found. Please place 'wnba_2023_skill_vector.csv' in project/data/ or repo root."
        )

    df = load_player_data(path)
    df = df.copy()
    df["pid"] = np.arange(len(df))

    # Prefer explicit 5-position labels if available (Pos5/pos from xlsx).
    # This replaces the previous KMeans->position heuristic for lineup building.
    pos_labels = _load_position_labels() if use_position_labels else pd.DataFrame()
    if not pos_labels.empty:
        df["Player"] = df["Player"].astype(str).str.strip()
        df["Team"] = df["Team"].astype(str).str.strip()
        df = df.merge(pos_labels, on=["Player", "Team"], how="left")
        df["Position"] = df["Pos5"].astype(str).str.upper()
    else:
        df["Position"] = np.nan

    # Raw feature matrix (before z-scoring). Note: must be captured *after* merge
    # so indices stay aligned with Pos5 coverage mask.
    X = df[FEATURES].copy()

    # Standardize using the same player subset that was used to fit positional coefficients
    # (i.e., the position-score file coverage), to keep coefficient interpretation stable.
    calib_mask = df.get("Pos5").notna() if "Pos5" in df.columns else pd.Series(False, index=df.index)
    X_calib = X[calib_mask] if calib_mask.any() else X
    Xs_calib, scaler = standardize(X_calib.values)
    mean = scaler["mean"]
    std = scaler["std"]
    Xs_all = (X.values - mean) / std
    z = pd.DataFrame(Xs_all, columns=FEATURES)
    df = df.copy()
    # Store standardized features in the dataframe (used by Q-vector computation)
    df[FEATURES] = z[FEATURES].values

    # Fallback for players not covered by the position label file
    missing_pos = df["Position"].isna() | df["Position"].astype(str).str.lower().isin(["", "nan", "none"])
    if missing_pos.any():
        df.loc[missing_pos, "Position"] = df.loc[missing_pos].apply(
            lambda r: _infer_pos5_from_raw(r.get("Pos", ""), r), axis=1
        )
    df["Position"] = df["Position"].astype(str).str.strip().str.upper()

    pos_order = ["PG", "SG", "SF", "PF", "C"]
    pos_to_cluster = {p: i for i, p in enumerate(pos_order)}
    df["cluster"] = df["Position"].map(pos_to_cluster).fillna(pos_to_cluster["SF"]).astype(int)
    cluster_to_position = {i: p for p, i in pos_to_cluster.items()}

    # Skill score: position-conditioned linear model on global z-scores.
    df["skill_score"] = _pos_skill_score(df, position_col="Position")

    # Keep the legacy global weighted-z score for debugging/compatibility.
    weights = load_skill_weights()
    df["skill_score_global"] = sum(float(weights[f]) * df[f].astype(float) for f in FEATURES)

    # Position "centroids" are not used for decision-making; keep a placeholder for logging/plots.
    centroids = np.zeros((len(pos_order), len(FEATURES)), dtype=float)

    cluster_profiles = df.groupby("cluster")[FEATURES].mean()

    age_map, contract_map = _load_spotrac_maps()

    # Precompute Q target quantiles from player distribution
    player_q = _player_q_vectors(df)
    q_targets = {}
    for q in [20, 35, 50, 65, 80, 88, 93]:
        q_targets[q] = np.nanpercentile(player_q, q, axis=0)

    model = PlayerModel(
        df=df,
        scaler=scaler,
        kmeans={"centroids": centroids},
        features=FEATURES,
        cluster_to_position=cluster_to_position,
        cluster_profiles=cluster_profiles,
        cluster_q_targets=q_targets,
        age_map=age_map,
        contract_map=contract_map,
    )
    model.build_league_rosters(roster_size=roster_size)
    return model


def _load_spotrac_maps() -> Tuple[Dict[str, float], Dict[str, int]]:
    age_map: Dict[str, float] = {}
    contract_map: Dict[str, int] = {}
    spot_path = Path("project/data/clean/spotrac_ind_yearly_clean.csv")
    if not spot_path.exists():
        return age_map, contract_map
    df = pd.read_csv(spot_path)
    if "Player" in df.columns and "Age" in df.columns:
        for _, row in df.iterrows():
            name = str(row["Player"]).strip()
            if name:
                try:
                    age_map[name] = float(row["Age"])
                except Exception:
                    pass
    # contract length from salary columns
    year_cols = [c for c in df.columns if c.endswith("_salary_m")]
    for _, row in df.iterrows():
        name = str(row["Player"]).strip()
        if not name:
            continue
        years = 0
        for col in year_cols:
            val = row.get(col, np.nan)
            if not pd.isna(val) and float(val) > 0:
                years += 1
        if years > 0:
            contract_map[name] = years
    return age_map, contract_map


def _player_q_vectors(df: pd.DataFrame) -> np.ndarray:
    z = df[FEATURES]
    # compute Q vector per player: Off, Def, Play, Reb
    off = z["WS/40"] + z["TS%"] + 0.5 * z["USG%"]
    ddef = z["DWS_40"] - 0.3 * z["USG%"]
    play = z["AST%"]
    reb = z["TRB%"]
    return np.vstack([off, ddef, play, reb]).T


def competitive_state_from_lineup(lineup: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = lineup[FEATURES]
    off = z["WS/40"] + z["TS%"] + 0.5 * z["USG%"]
    ddef = z["DWS_40"] - 0.3 * z["USG%"]
    play = z["AST%"]
    reb = z["TRB%"]
    Q = np.array([off.mean(), ddef.mean(), play.mean(), reb.mean()], dtype=float)

    # C vector from cluster counts (pad to length 6)
    counts = lineup["cluster"].value_counts().sort_index()
    c_vec = np.zeros(6, dtype=float)
    for i, (_, v) in enumerate(counts.items()):
        if i < 6:
            c_vec[i] = float(v)

    # Position counts
    pos_order = ["PG", "SG", "SF", "PF", "C"]
    p_vec = np.array([float((lineup["Position"] == p).sum()) for p in pos_order], dtype=float)
    return Q, c_vec, p_vec


def _compute_team_strengths(model: PlayerModel, team_rosters: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
    entries = []
    for team, roster in team_rosters.items():
        if roster.empty:
            continue
        lineup = model.lineup_for_team(team)
        if lineup.empty:
            lineup = roster.sort_values("skill_score", ascending=False).head(5)
        Q, _, _ = competitive_state_from_lineup(lineup)
        entries.append((team, Q, lineup))

    if not entries:
        return {}

    Q_mat = np.vstack([q for _, q, _ in entries])
    q_mean = Q_mat.mean(axis=0)
    q_std = np.where(Q_mat.std(axis=0) == 0, 1.0, Q_mat.std(axis=0))
    zQ = (Q_mat - q_mean) / q_std

    strengths: Dict[str, Dict[str, float]] = {}
    for idx, (team, Q, lineup) in enumerate(entries):
        z = zQ[idx]
        elo = 1500.0 + 45.0 * z[0] + 35.0 * z[1] + 25.0 * z[2] + 20.0 * z[3]
        elo = float(np.clip(elo, 1400.0, 1600.0))
        # Star count from top skill_score within lineup
        if "skill_score" in lineup.columns and len(lineup) > 0:
            star_thresh = float(model.df["skill_score"].quantile(0.85))
            stars = int((lineup["skill_score"] >= star_thresh).sum())
        else:
            stars = 0
        # Pace proxy from usage (bounded)
        if "USG%" in lineup.columns:
            pace = 78.0 + 15.0 * float(np.tanh(lineup["USG%"].astype(float).mean()))
        else:
            pace = 80.0
        strengths[team] = {"elo": elo, "pace": pace, "stars": float(stars)}
    return strengths


def save_cluster_figures(model: PlayerModel, out_dir: Path) -> Dict[str, Path]:
    import os
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
    except Exception:
        sns = None
    out_dir.mkdir(parents=True, exist_ok=True)
    df = model.df.copy()

    # PCA scatter
    Xs, _ = standardize(df[FEATURES].values)
    coords = pca_2d(Xs)
    df["pc1"] = coords[:, 0]
    df["pc2"] = coords[:, 1]

    plt.figure(figsize=(8, 6))
    if sns is not None:
        sns.scatterplot(data=df, x="pc1", y="pc2", hue="cluster", palette="tab10", alpha=0.7, s=40)
    else:
        for cluster in sorted(df["cluster"].unique()):
            subset = df[df["cluster"] == cluster]
            plt.scatter(subset["pc1"], subset["pc2"], label=f"{cluster}", alpha=0.7, s=40)
    for cluster, pos in model.cluster_to_position.items():
        subset = df[df["cluster"] == cluster]
        if not subset.empty:
            cx = subset["pc1"].mean()
            cy = subset["pc2"].mean()
            plt.text(cx, cy, pos, fontsize=10, weight="bold")
    plt.title("WNBA Player Clusters (KMeans, PCA view)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    pca_path = out_dir / "player_clusters_pca.png"
    plt.tight_layout()
    plt.savefig(pca_path, dpi=200)
    plt.close()

    # Cluster profile bar chart (z-scored)
    profile = model.cluster_profiles
    profile = (profile - profile.mean()) / profile.std(ddof=0)
    profile = profile.reset_index().melt(id_vars="cluster", var_name="metric", value_name="z")

    plt.figure(figsize=(9, 5))
    if sns is not None:
        sns.barplot(data=profile, x="metric", y="z", hue="cluster", palette="tab10")
    else:
        # fallback: simple grouped bars
        metrics = profile["metric"].unique()
        clusters = sorted(profile["cluster"].unique())
        x = np.arange(len(metrics))
        width = 0.8 / max(1, len(clusters))
        for i, cluster in enumerate(clusters):
            vals = profile[profile["cluster"] == cluster]["z"].values
            plt.bar(x + i * width, vals, width=width, label=str(cluster))
        plt.xticks(x + width * (len(clusters) - 1) / 2, metrics)
    plt.title("Cluster Skill Profiles (Z-score)")
    plt.xlabel("Metric")
    plt.ylabel("Z-score")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    profile_path = out_dir / "player_cluster_profiles.png"
    plt.tight_layout()
    plt.savefig(profile_path, dpi=200)
    plt.close()

    return {"pca": pca_path, "profiles": profile_path}
