# Skill Weight Calibration Report

## Data sources
- allplayers.csv (2024)
- wnba_advanced_stats.csv (season=2023)
- IND_ELO_O_SOS_season_level.csv (IND historical)

## Target construction
Targets: Win%, NetRtg, ELO_proxy (calibrated from IND historical ELO).

## Regression
- Chosen model: ridge
- Ridge CV MSE: 0.970409
- Lasso CV MSE: 1.000000

## Weights (L1-normalized)
- WS/40: -0.0431
- TS%: +0.0177
- USG%: +0.0204
- AST%: +0.0851
- TRB%: -0.3491
- DWS_40: -0.4847

## ELO proxy (IND calibrated)
ELO = 1526.89 + 5.541*NetRtg + 2.237*Win%

## Sensitivity analysis (200 perturbations)
- Spearman mean: 0.976 (std 0.020)
- Top5 overlap Draft: mean 0.878 (std 0.134)
- Top5 overlap FA: mean 0.700 (std 0.148)
- Top5 overlap Trade: mean 0.474 (std 0.181)
