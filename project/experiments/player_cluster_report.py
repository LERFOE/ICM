import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.data.player_kmeans import build_player_model, save_cluster_figures

OUTPUT_DIR = Path("figures")
OUTPUT_MD = Path("project/experiments/output/player_cluster_report.md")


def main():
    model = build_player_model()
    figs = save_cluster_figures(model, OUTPUT_DIR)

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_MD.open("w") as f:
        f.write("# Player Cluster Report\n\n")
        f.write("KMeans clustering on player skill vectors (k=5).\n\n")
        
        f.write("## Cluster Interpretation (Raw Profiles)\n")
        f.write("Below are the average **RAW statistics** for each cluster (not Z-scores):\n\n")
        
        # Format the raw profile table nicely
        profile_raw = model.cluster_profiles.copy()
        # Add position label for clarity
        profile_raw["Assigned_Pos"] = profile_raw.index.map(model.cluster_to_position)
        
        # Write markdown table
        f.write(profile_raw.round(4).to_markdown())
        f.write("\n\n")

        f.write("## Cluster → Position Mapping\n")
        for cluster, pos in model.cluster_to_position.items():
            f.write(f"- Cluster {cluster}: {pos}\n")
        
        f.write("\n## Figures\n")
        f.write("Note: The figures below using Z-scores (standardized values) to allow comparison across different metrics.\n\n")
        f.write(f"- PCA plot: {figs['pca']}\n")
        f.write(f"- Skill profile (Normalized): {figs['profiles']}\n")

    print(f"Saved figures to {OUTPUT_DIR}")
    print(f"Saved report to {OUTPUT_MD}")
    
    print("\n--- Raw Cluster Profiles (Preview) ---")
    print(profile_raw.round(4).to_string())


if __name__ == "__main__":
    main()
