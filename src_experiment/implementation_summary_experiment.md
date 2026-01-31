# implementation_summary_experiment.md

## Overview
This directory (`src_experiment`) implements the "Factor-based Game Engine" as described in the thesis (`lunwen.tex`), replacing the simplified "Win Shares" model.

## Logic Implementation

### 1. Factor Game Engine (`game_engine_factor.py`)
Instead of a single "Rating" number, the team performance is now driven by a 4-Factor Model ($\mathbf{Q}_t$):
- **Scoring**: $0.6 \times Z_{TS\%} + 0.4 \times Z_{USG\%}$
- **Playmaking**: $0.8 \times Z_{AST\%} - 0.3 \times Z_{TOV\%}$
- **Defense**: $0.6 \times Z_{DBPM} + 0.4 \times Z_{TRB\%}$
- **Gravity**: $0.9 \times Z_{USG\%} + 0.1 \times Z_{TS\%}$

### 2. Synergy Calculation (`Syn_t`)
The model explicitly calculates the "Synergy" term:
- **Positive Synergy**: Rewards teams with high `Playmaking` scores (Ball movement).
- **Negative Synergy**: Penalizes teams where total `Gravity` (Ball Dominance) exceeds a threshold (diminishing returns of too many stars).

### 3. Prediction Pipeline
1. **Normalize**: Convert raw player stats ($TS, USG, AST...$) to Z-Scores based on League Norms.
2. **Minutes Allocation**: Assign minutes based on `Score = Scoring + Defense`.
3. **Aggregation**: Compute weighted average Team Factors.
4. **Win%**: `Logistic(Alpha*Offense + Beta*Defense + Gamma*Synergy)`.

## Verification Results (`test_factor_engine.py`)
- **Superteam Dominance**: A team of Superstars achieved ~82% Win Rate vs ~53% for a Scrub team.
- **Ballhog Penalty**: A team of 5 "Ball Dominant" players yielded **Negative Synergy (-0.051)** due to conflict.
- **Playmaking Bonus**: Increasing Team AST% from 5% to 30% yielded **Positive Synergy (+0.225)**.

## How to use
This engine can replace the logic in `src/mdp_simulation.py` to upgrade the entire simulation to the "Thesis-Compliant" version.
