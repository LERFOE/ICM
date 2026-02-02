#!/usr/bin/env python3
"""
Q2: Robust Archetypes (Outliers + Soft Clustering) + Entropy-Regularized Inverse Optimization Skill Scoring
--------------------------------------------------------------------------------
This script builds player archetypes and a scientifically-stable skill score under small-team-sample constraints.

Key ideas (paper-aligned):
  1) Outlier-aware preprocessing (robust Mahalanobis + downweighting)
  2) Soft archetypes via Gaussian Mixture responsibilities (soft KMeans alternative)
  3) Role-adjusted (cluster-adjusted) standardization using responsibilities
  4) Skill scoring via entropy-regularized inverse optimization on the simplex:
        min_w  MSE(Xw + Rb, y*) + rho * sum(w log w) + lambda_b * ||b||^2
        s.t.  w in simplex (implemented with softmax)
     where y* is a latent performance target built from multiple proxies (PER, WS/40, OWS_40, DWS_40)

Outputs:
  - players_with_archetype_and_skill.csv
  - A folder of publication-style figures (English labels, colorful palettes)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

# -----------------------------
# Utility
# -----------------------------
def set_global_plot_style():
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 240,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

def robust_scale(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median/IQR scaling (robust)."""
    med = np.nanmedian(X, axis=0)
    q1 = np.nanpercentile(X, 25, axis=0)
    q3 = np.nanpercentile(X, 75, axis=0)
    iqr = np.where((q3 - q1) < 1e-9, 1.0, (q3 - q1))
    Xs = (X - med) / iqr
    return Xs, med, iqr

def softmax(u: np.ndarray) -> np.ndarray:
    u = u - np.max(u)
    e = np.exp(u)
    return e / (np.sum(e) + 1e-12)

def entropy_simplex_penalty(w: np.ndarray) -> float:
    """sum w log w (<=0)."""
    w_safe = np.clip(w, 1e-12, 1.0)
    return float(np.sum(w_safe * np.log(w_safe)))

def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2) + 1e-12)
    return 1.0 - ss_res / ss_tot

# -----------------------------
# Model: entropy-regularized simplex regression
# -----------------------------
@dataclass
class FitResult:
    w: np.ndarray              # (P,)
    b: np.ndarray              # (K,)
    yhat: np.ndarray           # (N,)
    rho: float
    lambda_b: float

def fit_entropy_simplex_regression(
    X: np.ndarray,             # (N, P)
    R: np.ndarray,             # (N, K) soft responsibilities
    y: np.ndarray,             # (N,)
    sample_weight: np.ndarray, # (N,)
    rho: float,
    lambda_b: float,
    max_iter: int = 5000,
    lr: float = 0.05,
    seed: int = 42
) -> FitResult:
    """
    Gradient-based optimization on unconstrained u (for simplex weights w = softmax(u))
    and b (cluster offsets). Uses weighted MSE + entropy regularization + L2 on b.
    """
    rng = np.random.default_rng(seed)
    N, P = X.shape
    K = R.shape[1]

    # init
    u = rng.normal(scale=0.1, size=P)
    b = np.zeros(K, dtype=float)

    sw = sample_weight / (np.mean(sample_weight) + 1e-12)

    # Precompute for speed
    Xt = X
    Rt = R

    def loss_and_grads(u: np.ndarray, b: np.ndarray):
        w = softmax(u)
        yhat = Xt @ w + Rt @ b
        resid = (yhat - y)
        # weighted mse
        mse = float(np.mean(sw * (resid ** 2)))
        # entropy penalty (note: w log w <= 0, so rho * sum w log w encourages spread)
        ent = entropy_simplex_penalty(w)
        regb = float(lambda_b * np.sum(b ** 2))

        loss = mse + rho * ent + regb

        # gradients
        # grad wrt yhat: d/dyhat mse = 2/N * sw * resid
        g = (2.0 / N) * (sw * resid)  # (N,)
        # grad wrt b: R^T g + 2 lambda_b b
        grad_b = Rt.T @ g + 2.0 * lambda_b * b  # (K,)

        # grad wrt w: X^T g + rho*(log w + 1)
        grad_w = Xt.T @ g + rho * (np.log(np.clip(w, 1e-12, 1.0)) + 1.0)  # (P,)

        # convert grad_w to grad_u via Jacobian of softmax: J = diag(w) - w w^T
        # grad_u = J^T grad_w = J grad_w
        grad_u = (w * grad_w) - w * np.sum(w * grad_w)

        return loss, grad_u, grad_b, yhat, w

    best = None
    best_loss = np.inf

    for it in range(max_iter):
        loss, grad_u, grad_b, yhat, w = loss_and_grads(u, b)

        # simple Adam-like step (lightweight)
        u = u - lr * grad_u
        b = b - lr * grad_b

        if loss < best_loss:
            best_loss = loss
            best = (u.copy(), b.copy(), yhat.copy(), w.copy())

        if it % 500 == 0 and it > 0:
            # mild learning-rate decay
            lr = max(lr * 0.85, 0.005)

    u_best, b_best, yhat_best, w_best = best
    return FitResult(w=w_best, b=b_best, yhat=yhat_best, rho=rho, lambda_b=lambda_b)

# -----------------------------
# Main pipeline
# -----------------------------
def build_latent_target(df: pd.DataFrame) -> np.ndarray:
    """
    Latent performance target y* from multiple proxies (player-level; avoids 12-team small sample):
      - PER
      - WS/40
      - OWS_40
      - DWS_40
    Combine via PCA(1) on standardized proxies.
    """
    proxies = ["PER", "WS/40", "OWS_40", "DWS_40"]
    Y = df[proxies].to_numpy(dtype=float)
    # standardize
    Y = (Y - np.nanmean(Y, axis=0)) / (np.nanstd(Y, axis=0) + 1e-12)

    pca = PCA(n_components=1, random_state=42)
    y_star = pca.fit_transform(Y).ravel()

    # orient so "higher is better" (positive correlation with WS/40)
    ws40 = df["WS/40"].to_numpy(dtype=float)
    if np.corrcoef(y_star, ws40)[0, 1] < 0:
        y_star = -y_star

    # standardize final target
    y_star = (y_star - y_star.mean()) / (y_star.std() + 1e-12)
    return y_star

def robust_outlier_weights(Xs: np.ndarray, contamination: float = 0.08) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust Mahalanobis distance via MinCovDet.
    Returns:
      - outlier_mask (bool)
      - sample_weight (float in (0,1])
    """
    mcd = MinCovDet(random_state=42, support_fraction=None).fit(Xs)
    # squared Mahalanobis distances
    d2 = mcd.mahalanobis(Xs)
    # automatic threshold by empirical quantile
    thr = np.quantile(d2, 1.0 - contamination)
    outlier = d2 >= thr

    # smooth downweight (no manual removal): w = 1 / (1 + d2 / median_d2)
    med = np.median(d2) + 1e-12
    w = 1.0 / (1.0 + (d2 / med))
    # ensure outliers get extra downweight
    w[outlier] *= 0.35
    return outlier, w

def weighted_cluster_stats(X: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster-wise mean and std using soft responsibilities.
    X: (N,P) (already robust-scaled)
    R: (N,K), rows sum to 1
    """
    N, P = X.shape
    K = R.shape[1]
    mu = np.zeros((K, P))
    sd = np.zeros((K, P))
    for k in range(K):
        w = R[:, k]
        wsum = np.sum(w) + 1e-12
        mu[k] = (w[:, None] * X).sum(axis=0) / wsum
        var = (w[:, None] * (X - mu[k])**2).sum(axis=0) / wsum
        sd[k] = np.sqrt(np.maximum(var, 1e-12))
    return mu, sd

def role_adjusted_standardize(Xs: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Role-adjusted standardization inspired by the paper:
      z_i = (x_i - mu_role(i)) / sigma_role(i)
    With soft responsibilities, use expected mu and sigma:
      mu_i = sum_k r_ik mu_k
      sd_i = sum_k r_ik sd_k
    """
    mu_k, sd_k = weighted_cluster_stats(Xs, R)
    mu_i = R @ mu_k
    sd_i = R @ sd_k
    Z = (Xs - mu_i) / (sd_i + 1e-12)
    return Z

def cv_select_hyperparams(X: np.ndarray, R: np.ndarray, y: np.ndarray, sw: np.ndarray,
                          rho_grid: List[float], lb_grid: List[float], kfold: int = 5) -> Tuple[float, float, Dict]:
    """
    Choose (rho, lambda_b) via K-fold CV on players (stable N~150+).
    """
    kf = KFold(n_splits=kfold, shuffle=True, random_state=42)
    results = []
    for rho in rho_grid:
        for lb in lb_grid:
            r2s = []
            rmses = []
            for tr, te in kf.split(X):
                fit = fit_entropy_simplex_regression(
                    X[tr], R[tr], y[tr], sw[tr],
                    rho=rho, lambda_b=lb,
                    max_iter=1400, lr=0.07, seed=42
                )
                yhat = (X[te] @ fit.w) + (R[te] @ fit.b)
                resid = yhat - y[te]
                rmse = float(np.sqrt(np.mean(resid**2)))
                r2s.append(r2_score(y[te], yhat))
                rmses.append(rmse)
            results.append((rho, lb, float(np.mean(r2s)), float(np.mean(rmses))))
    # prioritize higher R^2, then lower RMSE
    results.sort(key=lambda x: (-x[2], x[3]))
    best_rho, best_lb, best_r2, best_rmse = results[0]
    meta = {
        "grid_results": results,
        "best": {"rho": best_rho, "lambda_b": best_lb, "cv_r2": best_r2, "cv_rmse": best_rmse}
    }
    return best_rho, best_lb, meta

def bootstrap_weights(X: np.ndarray, R: np.ndarray, y: np.ndarray, sw: np.ndarray,
                      rho: float, lb: float, B: int = 250, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bootstrap CIs for simplex weights and cluster offsets.
    Returns:
      w_boot: (B,P)
      b_boot: (B,K)
    """
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    P = X.shape[1]
    K = R.shape[1]
    w_boot = np.zeros((B, P))
    b_boot = np.zeros((B, K))
    for b in range(B):
        idx = rng.integers(0, N, size=N)
        fit = fit_entropy_simplex_regression(X[idx], R[idx], y[idx], sw[idx], rho=rho, lambda_b=lb,
                                             max_iter=900, lr=0.07, seed=seed + b)
        w_boot[b] = fit.w
        b_boot[b] = fit.b
    return w_boot, b_boot

# -----------------------------
# Visualizations
# -----------------------------
def plot_pca_scatter(Z: np.ndarray, labels: np.ndarray, mp: np.ndarray, outlier: np.ndarray,
                     team: np.ndarray, out_path: Path):
    pca = PCA(n_components=2, random_state=42)
    U = pca.fit_transform(Z)
    cmap = plt.get_cmap("tab10")
    K = int(labels.max()) + 1

    plt.figure(figsize=(9.5, 6.2))
    for k in range(K):
        m = labels == k
        plt.scatter(U[m, 0], U[m, 1],
                    s=20 + 80 * (mp[m] / (np.percentile(mp, 95) + 1e-12)),
                    c=[cmap(k)], alpha=0.85, edgecolors="none", label=f"Archetype {k+1}")

    # outliers highlighted
    if outlier.any():
        plt.scatter(U[outlier, 0], U[outlier, 1], s=90,
                    facecolors="none", edgecolors="black", linewidths=1.2,
                    label="Outlier (downweighted)")

    plt.title("Soft Archetypes in PCA Space (Size ~ Minutes Played)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    plt.legend(ncol=2, frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def plot_cluster_feature_heatmap(mu_k: np.ndarray, feat_names: List[str], out_path: Path):
    # mu_k: (K,P)
    plt.figure(figsize=(10, 5.5))
    im = plt.imshow(mu_k, aspect="auto", interpolation="nearest", cmap="coolwarm")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Role-adjusted mean (z)")
    plt.yticks(range(mu_k.shape[0]), [f"Archetype {i+1}" for i in range(mu_k.shape[0])])
    plt.xticks(range(mu_k.shape[1]), feat_names, rotation=30, ha="right")
    plt.title("Archetype Profiles (Role-adjusted Feature Means)")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def plot_weights_ci(w_mean: np.ndarray, w_ci: Tuple[np.ndarray, np.ndarray], feat_names: List[str], out_path: Path):
    lo, hi = w_ci
    x = np.arange(len(feat_names))
    plt.figure(figsize=(10, 4.8))
    plt.bar(x, w_mean, color=plt.get_cmap("Set2")(x % 8), edgecolor="black", linewidth=0.6)
    plt.errorbar(x, w_mean, yerr=[w_mean - lo, hi - w_mean], fmt="none", ecolor="black", capsize=4, linewidth=1.0)
    plt.xticks(x, feat_names, rotation=30, ha="right")
    plt.ylabel("Weight (simplex)")
    plt.title("Entropy-Regularized Simplex Weights (Bootstrap 95% CI)")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def plot_bias_ci(b_mean: np.ndarray, b_ci: Tuple[np.ndarray, np.ndarray], out_path: Path):
    lo, hi = b_ci
    x = np.arange(len(b_mean))
    plt.figure(figsize=(9.5, 4.6))
    plt.bar(x, b_mean, color=plt.get_cmap("tab10")(x), edgecolor="black", linewidth=0.6)
    plt.errorbar(x, b_mean, yerr=[b_mean - lo, hi - b_mean], fmt="none", ecolor="black", capsize=4, linewidth=1.0)
    plt.xticks(x, [f"Archetype {i+1}" for i in x])
    plt.ylabel("Offset (latent units)")
    plt.title("Archetype Offsets (Bootstrap 95% CI)")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def plot_pred_scatter(y: np.ndarray, yhat: np.ndarray, labels: np.ndarray, out_path: Path, title: str):
    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(6.8, 6.0))
    for k in range(int(labels.max()) + 1):
        m = labels == k
        plt.scatter(y[m], yhat[m], c=[cmap(k)], alpha=0.8, label=f"Archetype {k+1}")
    lo = min(y.min(), yhat.min())
    hi = max(y.max(), yhat.max())
    plt.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.2)
    plt.xlabel("Observed latent performance (standardized)")
    plt.ylabel("Predicted latent performance (standardized)")
    plt.title(title)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def plot_residuals(yhat: np.ndarray, resid: np.ndarray, out_path: Path):
    plt.figure(figsize=(7.4, 5.0))
    plt.scatter(yhat, resid, color=plt.get_cmap("viridis")(0.35), alpha=0.85)
    plt.axhline(0, linestyle="--", color="black", linewidth=1.0)
    plt.xlabel("Fitted value")
    plt.ylabel("Residual")
    plt.title("Residual Diagnostics")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def plot_skill_distributions(skill: np.ndarray, labels: np.ndarray, out_path: Path):
    plt.figure(figsize=(10, 5.4))
    cmap = plt.get_cmap("tab10")
    bins = np.linspace(0, 100, 26)
    for k in range(int(labels.max()) + 1):
        vals = skill[labels == k]
        plt.hist(vals, bins=bins, alpha=0.55, color=cmap(k), label=f"Archetype {k+1}", density=True)
    plt.xlabel("Skill score (0-100)")
    plt.ylabel("Density")
    plt.title("Skill Score Distributions by Archetype")
    plt.legend(ncol=2, frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def plot_team_ranking(df: pd.DataFrame, out_path: Path):
    """Minutes-weighted team average skill score (deprecation-safe)."""
    tmp = df.copy()
    tmp["w_skill"] = tmp["MP"].to_numpy(float) * tmp["skill_score"].to_numpy(float)
    g = (tmp.groupby("Team", as_index=False)
            .agg(w_skill_sum=("w_skill", "sum"), mp_sum=("MP", "sum")))
    g["team_skill"] = g["w_skill_sum"] / (g["mp_sum"] + 1e-12)
    g = g.sort_values("team_skill", ascending=False).reset_index(drop=True)

    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(g)))

    plt.figure(figsize=(10, 5.4))
    plt.barh(g["Team"][::-1], g["team_skill"][::-1], color=colors[::-1], edgecolor="black", linewidth=0.4)
    plt.xlabel("Minutes-weighted team skill (0-100)")
    plt.title("Team Skill Ranking (Aggregated from Player Scores)")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def plot_proxy_correlations(df: pd.DataFrame, out_path: Path):
    # correlate skill_score with proxies
    cols = ["skill_score", "PER", "WS/40", "OWS_40", "DWS_40", "TS%", "USG%", "AST%", "TRB%"]
    cols = [c for c in cols if c in df.columns]
    X = df[cols].to_numpy(dtype=float)
    C = np.corrcoef(X, rowvar=False)

    plt.figure(figsize=(8.2, 7.0))
    im = plt.imshow(C, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Correlation")
    plt.xticks(range(len(cols)), cols, rotation=35, ha="right")
    plt.yticks(range(len(cols)), cols)
    plt.title("Skill Score vs. Statistical Proxies (Correlation Matrix)")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

# -----------------------------
# Run
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", type=str, required=True, help="Path to player advanced stats CSV (2025 WNBA).")
    parser.add_argument("--out", type=str, default="q2_outputs", help="Output directory.")
    parser.add_argument("--k", type=int, default=5, help="Number of archetypes.")
    parser.add_argument("--min_mp", type=float, default=None, help="Minimum minutes threshold (auto if None).")
    parser.add_argument("--fast", action="store_true", help="Fast mode: smaller CV grid & fewer bootstraps (for quick runs).")
    args = parser.parse_args()

    set_global_plot_style()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.players)

    # Required columns
    required = ["Player", "Team", "MP", "PER", "TS%", "USG%", "AST%", "TRB%", "DWS", "OWS", "WS/40"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

    # Derived per-40 proxies
    df["DWS_40"] = (df["DWS"] / df["MP"].replace(0, np.nan)) * 40.0
    df["OWS_40"] = (df["OWS"] / df["MP"].replace(0, np.nan)) * 40.0

    # Basic cleaning
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["MP", "PER", "WS/40", "TS%", "USG%", "AST%", "TRB%", "DWS_40", "OWS_40"])
    df["MP"] = df["MP"].astype(float)

    # Automatic minutes threshold: keep top ~80% minutes players (stabilizes small-minute noise)
    if args.min_mp is None:
        min_mp = float(np.quantile(df["MP"].values, 0.20))
    else:
        min_mp = float(args.min_mp)

    df = df[df["MP"] >= min_mp].copy()
    df = df.reset_index(drop=True)

    # Features (exclude WS/40 to avoid leakage; use ability indicators)
    feat_cols = ["TS%", "USG%", "AST%", "TRB%", "DWS_40"]
    X = df[feat_cols].to_numpy(dtype=float)

    # Robust scaling
    Xs, med, iqr = robust_scale(X)

    # Outlier handling (downweight rather than delete)
    outlier, sw = robust_outlier_weights(Xs, contamination=0.08)

    # Soft archetypes (GMM) on downweighted non-outliers
    K = int(args.k)
    X_fit = Xs[~outlier]
    gmm = GaussianMixture(n_components=K, covariance_type="full", random_state=42, n_init=20, reg_covar=1e-5)
    gmm.fit(X_fit)
    R = gmm.predict_proba(Xs)  # (N,K) responsibilities

    # Hard label for visualization
    labels = np.argmax(R, axis=1)

    # Role-adjusted standardization (paper-aligned)
    Z = role_adjusted_standardize(Xs, R)  # (N,P)

    # Latent target (avoid 12-team small sample)
    y = build_latent_target(df)

    # CV select regularization strengths (automatic)
        # Speed/quality toggle
    fast = getattr(args, 'fast', False)
    if fast:
        rho_grid = [1e-4, 1e-3, 1e-2, 3e-2]
        lb_grid  = [1e-4, 1e-3, 1e-2]
        kfold = 3
    else:
        rho_grid = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
        lb_grid  = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
        kfold = 5
    best_rho, best_lb, meta = cv_select_hyperparams(Z, R, y, sw, rho_grid, lb_grid, kfold=kfold)

    # Final fit
    fit = fit_entropy_simplex_regression(Z, R, y, sw, rho=best_rho, lambda_b=best_lb, max_iter=2600, lr=0.07, seed=42)
    yhat = fit.yhat
    resid = yhat - y

    # Bootstrap uncertainty for weights and offsets
    B = 80 if getattr(args, 'fast', False) else 250
    w_boot, b_boot = bootstrap_weights(Z, R, y, sw, rho=best_rho, lb=best_lb, B=B, seed=42)
    w_mean = w_boot.mean(axis=0)
    w_lo = np.quantile(w_boot, 0.025, axis=0)
    w_hi = np.quantile(w_boot, 0.975, axis=0)

    b_mean = b_boot.mean(axis=0)
    b_lo = np.quantile(b_boot, 0.025, axis=0)
    b_hi = np.quantile(b_boot, 0.975, axis=0)

    # Skill score (0-100) robust scaling
    score_raw = yhat.copy()
    q05, q95 = np.quantile(score_raw, [0.05, 0.95])
    skill = np.clip((score_raw - q05) / (q95 - q05 + 1e-12), 0, 1) * 100.0

    # Score CI (propagate from bootstrap prediction)
    # Predict bootstrap skill quickly
    yhat_boot = (Z @ w_boot.T) + (R @ b_boot.T)  # (N,B)
    score_lo = np.quantile(yhat_boot, 0.025, axis=1)
    score_hi = np.quantile(yhat_boot, 0.975, axis=1)
    skill_lo = np.clip((score_lo - q05) / (q95 - q05 + 1e-12), 0, 1) * 100.0
    skill_hi = np.clip((score_hi - q05) / (q95 - q05 + 1e-12), 0, 1) * 100.0

    df["archetype_soft"] = labels + 1  # 1..K
    for k in range(K):
        df[f"resp_{k+1}"] = R[:, k]
    df["outlier_flag"] = outlier.astype(int)
    df["latent_y_star"] = y
    df["latent_y_pred"] = yhat
    df["skill_score"] = skill
    df["skill_ci_low"] = skill_lo
    df["skill_ci_high"] = skill_hi

    # Save main output
    out_csv = out_dir / "players_with_archetype_and_skill.csv"
    df.to_csv(out_csv, index=False)

    # Archetype profile means (role-adjusted Z means)
    mu_k, sd_k = weighted_cluster_stats(Z, R)
    prof = pd.DataFrame(mu_k, columns=feat_cols)
    prof.insert(0, "Archetype", [f"{i+1}" for i in range(K)])
    prof.to_csv(out_dir / "archetype_profiles_role_adjusted.csv", index=False)

    # Save model summary
    summary = {
        "n_players_used": int(df.shape[0]),
        "features": feat_cols,
        "k_archetypes": K,
        "minutes_threshold": float(min_mp),
        "best_hyperparams": meta["best"],
        "in_sample_r2": r2_score(y, yhat),
        "in_sample_rmse": float(np.sqrt(np.mean((yhat - y)**2))),
        "simplex_weights_mean": {feat_cols[i]: float(w_mean[i]) for i in range(len(feat_cols))},
        "archetype_offsets_mean": {f"Archetype {i+1}": float(b_mean[i]) for i in range(K)},
    }
    (out_dir / "model_summary.json").write_text(json.dumps(summary, indent=2))

    # -----------------------------
    # Figures
    # -----------------------------
    # PCA scatter
    plot_pca_scatter(Z, labels, df["MP"].to_numpy(float), outlier, df["Team"].to_numpy(str),
                     out_dir / "fig1_soft_archetypes_pca.png")

    # Archetype heatmap
    plot_cluster_feature_heatmap(mu_k, feat_cols, out_dir / "fig2_archetype_profiles_heatmap.png")

    # Weight CI
    plot_weights_ci(w_mean, (w_lo, w_hi), feat_cols, out_dir / "fig3_simplex_weights_ci.png")

    # Offset CI
    plot_bias_ci(b_mean, (b_lo, b_hi), out_dir / "fig4_archetype_offsets_ci.png")

    # Fit scatter
    plot_pred_scatter(y, yhat, labels, out_dir / "fig5_latent_fit_scatter.png",
                      title=f"Latent Performance: Observed vs Predicted (R² = {r2_score(y, yhat):.3f})")

    # Residuals
    plot_residuals(yhat, resid, out_dir / "fig6_residual_diagnostics.png")

    # Skill distributions
    plot_skill_distributions(skill, labels, out_dir / "fig7_skill_distributions.png")

    # Team ranking (derived from player scores; useful if you lack team Win% table)
    plot_team_ranking(df, out_dir / "fig8_team_skill_ranking.png")

    # Correlations with proxies
    plot_proxy_correlations(df, out_dir / "fig9_skill_proxy_correlations.png")

    # Top players table figure (academic-style)
    top = df.sort_values("skill_score", ascending=False).head(15).copy()
    top["CI"] = top.apply(lambda r: f"[{r['skill_ci_low']:.1f}, {r['skill_ci_high']:.1f}]", axis=1)
    show_cols = ["Player", "Team", "Pos", "MP", "skill_score", "CI", "PER", "WS/40"]
    show_cols = [c for c in show_cols if c in top.columns]
    top_tbl = top[show_cols]

    plt.figure(figsize=(11.5, 4.0))
    plt.axis("off")
    tbl = plt.table(cellText=top_tbl.values, colLabels=top_tbl.columns, loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.3)
    plt.title("Top-15 Players by Skill Score (with 95% Bootstrap CI)", pad=12)
    plt.tight_layout()
    plt.savefig(out_dir / "fig10_top_players_table.png", bbox_inches="tight")
    plt.close()

    # Write README
    readme = f"""# Q2 Outputs

This folder contains the full Q2 results:
- `players_with_archetype_and_skill.csv` : player-level archetypes (soft), outlier flags, latent target & predictions, and final 0-100 skill score with 95% CI.
- `archetype_profiles_role_adjusted.csv` : role-adjusted archetype feature means.
- `model_summary.json` : hyperparameters, goodness-of-fit, and learned weights.

## Figures (paper-ready, English)
1. `fig1_soft_archetypes_pca.png` — Soft archetypes in PCA space (size ~ minutes).
2. `fig2_archetype_profiles_heatmap.png` — Archetype profiles (role-adjusted means).
3. `fig3_simplex_weights_ci.png` — Learned simplex weights with bootstrap 95% CI.
4. `fig4_archetype_offsets_ci.png` — Archetype offsets with bootstrap 95% CI.
5. `fig5_latent_fit_scatter.png` — Observed vs predicted latent performance.
6. `fig6_residual_diagnostics.png` — Residual plot.
7. `fig7_skill_distributions.png` — Skill distributions by archetype.
8. `fig8_team_skill_ranking.png` — Team ranking aggregated from player scores.
9. `fig9_skill_proxy_correlations.png` — Correlation matrix (skill vs proxies).
10. `fig10_top_players_table.png` — Top-15 skill table.

## Method summary
- Outliers: robust Mahalanobis (MinCovDet), downweighted (not removed).
- Archetypes: Gaussian Mixture Model responsibilities (soft clustering).
- Standardization: role-adjusted standardization using responsibilities.
- Scoring: entropy-regularized inverse optimization on simplex weights, plus archetype offsets.
- Hyperparameters: automatically selected via K-fold CV (player-level, stable sample size).
"""
    (out_dir / "README.md").write_text(readme)

    print("Done.")
    print("Output directory:", out_dir.resolve())
    print("Main CSV:", out_csv.resolve())
    print("Best hyperparams:", meta["best"])
    print("In-sample R2:", r2_score(y, yhat))

if __name__ == "__main__":
    import json
    main()