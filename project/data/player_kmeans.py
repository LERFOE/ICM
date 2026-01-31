from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import svd


DEFAULT_SKILL_FILE_CANDIDATES = [
    Path("wnba_2023_skill_vector.csv"),
    Path("project/data/wnba_2023_skill_vector.csv"),
    Path("project/data/clean/wnba_2023_skill_vector.csv"),
    Path("allplayers.csv"),
]

FEATURES = ["WS/40", "TS%", "USG%", "AST%", "TRB%", "DWS_40"]


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
    X = df[FEATURES].copy()

    Xs, scaler = standardize(X.values)
    labels, centroids = simple_kmeans(Xs, n_clusters=n_clusters, seed=seed, iters=30)
    df = df.copy()
    df["cluster"] = labels

    # Composite skill score (weighted z)
    z = pd.DataFrame(Xs, columns=FEATURES)
    df["skill_score"] = (
        0.35 * z["WS/40"]
        + 0.20 * z["TS%"]
        + 0.10 * z["USG%"]
        + 0.15 * z["AST%"]
        + 0.10 * z["TRB%"]
        + 0.10 * z["DWS_40"]
    )

    cluster_profiles = df.groupby("cluster")[FEATURES].mean()
    cluster_to_position = _assign_positions(cluster_profiles)

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
