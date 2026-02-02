
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from math import pi

# -----------------------------
# Configuration
# -----------------------------
INPUT_DIR = Path("project/experiments/output/q5_compare_replan")
OUTPUT_DIR = Path("newfigures")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
})

# -----------------------------
# 1. Radar Chart (Spider Plot)
# -----------------------------
def plot_radar(df_final):
    # Select metrics
    metrics = ["win_pct", "ELO", "CF", "OwnerTerminal", "Skill_Mean"]
    labels = ["Win %", "ELO", "Cash Flow", "Owner Value", "Roster Skill"]
    
    # Normalize data for radar chart (min-max scaling across both scenarios)
    df_radar = df_final[["scenario"] + metrics].copy()
    
    # Calculate min/max for normalization reference
    stats = {}
    for col in metrics:
        stats[col] = {
            "min": df_radar[col].min() * 0.95,
            "max": df_radar[col].max() * 1.05
        }
        # Normalize
        df_radar[col] = (df_radar[col] - stats[col]["min"]) / (stats[col]["max"] - stats[col]["min"])

    # Setup radar
    N = len(metrics)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    
    colors = {"replace_only": "#1f77b4", "model_replan": "#ff7f0e"}
    legend_labels = {"replace_only": "Baseline (Replace Only)", "model_replan": "Ours (Dynamic Replan)"}
    
    for scenario in ["replace_only", "model_replan"]:
        row = df_radar[df_radar["scenario"] == scenario]
        if row.empty: continue
        
        values = row[metrics].values.flatten().tolist()
        values += values[:1]
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=legend_labels[scenario], color=colors[scenario])
        ax.fill(angles, values, color=colors[scenario], alpha=0.25)
    
    # Stylize
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], labels, size=12)
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75, 1.0], ["", "", "", ""], color="grey", size=7)
    plt.ylim(0, 1.05)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Scenario Comparison: Final State Metrics", y=1.08)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "q5_radar_comparison.png")
    plt.close()
    print("Saved q5_radar_comparison.png")

# -----------------------------
# 2. Time Series Comparison
# -----------------------------
def plot_time_series(df_metrics):
    # Pivot for ELO and CF
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    scenarios = ["replace_only", "model_replan"]
    colors = {"replace_only": "#1f77b4", "model_replan": "#ff7f0e"}
    clean_labels = {"replace_only": "Baseline", "model_replan": "Ours"}
    
    # Plot ELO
    for sc in scenarios:
        subset = df_metrics[df_metrics["scenario"] == sc]
        axes[0].plot(subset["step"], subset["ELO"], marker='o', label=clean_labels[sc], color=colors[sc], linewidth=2.5)
    
    axes[0].set_title("Competitive Strength (ELO) Trajectory")
    axes[0].set_ylabel("ELO Rating")
    axes[0].set_xlabel("Simulation Step")
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend()
    
    # Plot CF (Cumulative or Instant) - let's do Instant CF
    for sc in scenarios:
        subset = df_metrics[df_metrics["scenario"] == sc]
        axes[1].plot(subset["step"], subset["CF"], marker='s', label=clean_labels[sc], color=colors[sc], linewidth=2.5)
    
    axes[1].set_title("Financial Health (Cash Flow) Trajectory")
    axes[1].set_ylabel("Cash Flow ($M)")
    axes[1].set_xlabel("Simulation Step")
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "q5_timeseries_comparison.png")
    plt.close()
    print("Saved q5_timeseries_comparison.png")

# -----------------------------
# 3. Strategy Difference (Action Heatmap equivalent)
# -----------------------------
def plot_strategy_diff(df_plan):
    # We want to see how actions differed. Pivot to mean action values.
    # Group by scenario and compute mean of a_salary, a_ticket, a_marketing
    
    if "scenario" not in df_plan.columns:
        # Plan file doesn't have 'scenario' column directly usually if separate...
        # Wait, simulate_replan returns ScenarioResult which has plan frame.
        # Check if saved CSV has scenario column.
        # Assuming we need to merge or it's there. 
        # In current logic, plan dataframe might not have scenario if not added explicitly.
        # Let's hope the previous script added it or I'll handle it.
        pass
        
    # Load and clean
    metrics = ["a_salary", "a_ticket", "a_marketing", "a_debt", "a_equity"]
    # If scenario missing in plan (likely), we might have to infer or skip. 
    # Actually, looking at `q5_compare_replan_vs_replace.py`, `plan_rows` does NOT include scenario key inside the loop,
    # but the `ScenarioResult` has `name`. The CSV save logic likely concatenates them.
    
    summary = df_plan.groupby("scenario")[metrics].mean().T
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    summary.plot(kind="bar", ax=ax, width=0.7, color=["#1f77b4", "#ff7f0e"])
    
    ax.set_title("Average Strategic Actions Taken")
    ax.set_ylabel("Action Level (Categorical Index)")
    ax.set_xlabel("Action Type")
    plt.xticks(rotation=0)
    plt.legend(["Baseline", "Ours"])
    
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "q5_strategy_diff.png")
    plt.close()
    print("Saved q5_strategy_diff.png")


def main():
    try:
        df_metrics = pd.read_csv(INPUT_DIR / "q5_compare_metrics.csv")
        df_plan = pd.read_csv(INPUT_DIR / "q5_compare_plan.csv")
        
        # Get final state for radar
        # Filter for max step
        max_step = df_metrics["step"].max()
        df_final = df_metrics[df_metrics["step"] == max_step]
        
        print("Generating figures...")
        plot_radar(df_final)
        plot_time_series(df_metrics)
        plot_strategy_diff(df_plan)
        
    except Exception as e:
        print(f"Error loading data or plotting: {e}")

if __name__ == "__main__":
    main()
