# Player Cluster Report

KMeans clustering on player skill vectors (k=5).

## Cluster Interpretation (Raw Profiles)
Below are the average **RAW statistics** for each cluster (not Z-scores):

|   cluster |   WS/40 |    TS% |    USG% |    AST% |    TRB% |   DWS_40 | Assigned_Pos   |
|----------:|--------:|-------:|--------:|--------:|--------:|---------:|:---------------|
|         0 |  0.1326 | 0.548  | 19.6576 | 11.6636 | 16.6818 |   0.0782 | SG             |
|         1 |  0.0812 | 0.5429 | 16.7348 | 11.55   |  8.45   |   0.0474 | PF             |
|         2 | -0.1322 | 0.2332 | 15.1615 |  6.1462 | 10.7154 |   0.0191 | C              |
|         3 |  0.1753 | 0.5713 | 25.0625 | 26.7708 |  9.5083 |   0.0622 | PG             |
|         4 |  0.0199 | 0.4985 | 17.9308 | 18.1615 |  5.9077 |   0.0093 | SF             |

## Cluster → Position Mapping
- Cluster 3: PG
- Cluster 0: SG
- Cluster 4: SF
- Cluster 1: PF
- Cluster 2: C

## Figures
Note: The figures below using Z-scores (standardized values) to allow comparison across different metrics.

- PCA plot: figures/player_clusters_pca.png
- Skill profile (Normalized): figures/player_cluster_profiles.png
