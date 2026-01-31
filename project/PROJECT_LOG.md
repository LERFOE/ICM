# Project Log

## What I did (step-by-step)
1. **Data discovery**: inventoried all CSV files in the workspace and summarized their schemas.
2. **Data cleaning**: created a reproducible cleaning pipeline that normalizes numeric fields, parses Spotrac salary strings, removes unnamed columns, and adds derived opponent stats.
3. **Data audit**: generated a markdown audit with shape, columns, and missing ratios for each cleaned dataset.
4. **Data integration**: built loaders + feature constructors to turn cleaned data into initial MDP states and to calibrate model parameters.
5. **Model calibration (based on available data)**:
   - Attendance vs Win% regression → `rev_win_beta`
   - Residual variability → ticket price elasticity and marketing attendance lift
   - ELO vs Win% logistic regression → `win_eta0`, `win_eta1`, `win_eta_sos`
   - Market size μ derived from Indiana attendance vs league average
   - Revenue split inferred from `wnba_valuations.csv` (gate/media/sponsor ratios)
6. **MDP updates**: wired data-driven initialization into the environment (`reset(use_data=True)`) and applied price elasticity & marketing attendance effects in financial transitions.
7. **Experiments updated**: default to data-driven config + state.
8. **Improvement pass**: added stricter leverage penalties, terminal value weighting, action caps for debt/equity, longer MCTS horizon, and stronger performance→revenue feedback; re-ran Q1–Q4 outputs.
9. **Player-based competitive module**: added KMeans clustering over real player skill vectors (k=5), lineup construction by position clusters, and PCA/profile plots for paper figures. Connected player-based Q/C/P into the competitive state builder, and enabled roster updates driven by real player pool.
10. **Action space refinement**: expanded action discretization (roster/salary/ticket/marketing/debt/equity) and updated mappings/normalization accordingly.
11. **Solver upgrade**: added heuristic value function and greedy rollout candidates in MCTS to reduce variance and improve planning quality.
12. **Q2/Q3/Q4 fixes**: added player-level recruitment strategy outputs (Draft/FA/Trade), expansion winners/losers using market+strength proxies, and strict single-dimension policy control for ticket/equity PPO.
13. **Skill weight calibration**: added ridge/lasso weight fitting from team outcomes (Win%/NetRtg/ELO proxy) with sensitivity analysis; stored in `project/data/skill_weights.json`.

---

## Repository overview

### High-level architecture
This project models team operations as a **phase-aware MDP** with **frozen actions** and **configuration state** `K_t`, matching the paper’s definition. The MDP state is:
`S_t = { R_t, F_t, E_t, Θ_t, K_t }`
where competitive, financial, and environmental dynamics are jointly simulated.

**Core flow**
1. Phase-aware action mask → enforce frozen decisions
2. Competitive transition → roster updates + game simulation
3. Financial transition → revenue/cost/valuation update
4. Environment transition → macro & expansion shocks
5. Reward → CF + valuation growth − risk

---

## Data pipeline (new)

### `project/data/clean_and_audit.py`
- **What it does**: Cleans raw CSVs and writes standardized files to `project/data/clean/`.
- **Key steps**:
  - Parses Spotrac salary strings into `*_salary_m`, `*_pct`, `*_status`
  - Removes `Unnamed` columns
  - Casts numeric columns to floats
  - Parses JSON lists in `O_t_list_json`

### `project/data/clean/data_audit.md`
- **What it is**: Markdown report of cleaned dataset schemas and missingness.

### `project/data/loaders.py`
- **What it does**: Simple file loaders for cleaned datasets (single source of truth).

### `project/data/integration.py`
- **What it does**: Bridges cleaned data into the MDP.
- **Key outputs**:
  - `calibrate_config_from_data()` → sets MDPConfig parameters from data
  - `build_initial_state_from_data()` → constructs competitive + financial + env states
- **Calibration details**:
- `rev_win_beta`: from attendance vs win% regression
  - Lasso regularization optional to reduce overfit on sparse data
  - `ticket_elasticity`: derived from attendance residual variance
  - `marketing_attendance_beta`: derived from same residuals
  - `win_eta0 / win_eta1 / win_eta_sos`: logistic fit of IND ELO diff → win prob
  - `market_size μ`: Indiana attendance / league average
  - Base revenues & `FV`: from valuations file (fallback to attendance)

### `project/data/run_calibration.py`
- **What it does**: Writes `project/data/clean/calibration_report.md` for traceability.

### `project/data/player_kmeans.py`
- **What it does**: Loads real player skill vectors, runs KMeans (k=5), maps clusters to positions, computes Q/C/P from lineup, and saves PCA/profile figures.
- **Defense signal**: Uses `DWS_40` (derived from `DWS/MP`) as the defensive feature instead of DBPM.
- **League context**: Builds team rosters for all teams and derives opponent ELO/pace/star proxies from real-player lineups.

### `project/data/prepare_player_data.py`
- **What it does**: Builds a clean player skill file from `allplayers.csv` and derives `DWS_40` from `DWS` and `MP`.

### `project/data/calibrate_skill_weights.py`
- **What it does**: Fits ridge/lasso weights for `skill_score` from team outcomes (Win%/NetRtg/ELO proxy), writes `project/data/skill_weights.json` and `project/data/skill_weights_report.md`, and runs sensitivity analysis on candidate stability.

### `project/experiments/player_cluster_report.py`
- **What it does**: Generates figures and a short markdown report for the cluster analysis (for direct use in the paper).

---

## MDP core (existing + updated)

### `project/mdp/config.py`
- Global parameters for the MDP.
- **New fields**:
  - `ticket_elasticity`
  - `marketing_attendance_beta`

### `project/mdp/state.py`
- Typed dataclasses for competitive/financial/environment states.
- Supports conversion to vector for RL.

### `project/mdp/mask.py`
- Implements phase-aware action mask and `K_t` roll-forward:
  - Frozen actions must equal `K_t`.
  - Mutable actions update `K_{t+1}`.

### `project/mdp/action.py`
- Action ranges and labels for 6-D discrete vector (refined granularity).

### `project/mdp/transitions_comp.py`
- Competitive dynamics:
  - `roster_update()` with statistical changes to Q/C/P/L/A/Syn or player-model based updates
  - `game_sim_update()` using logistic win prob from ELO/SOS/Syn/skill

### `project/mdp/transitions_fin.py`
- Financial dynamics:
  - Revenue, costs, leverage updates
  - **Updated**: ticket price elasticity and marketing attendance lift

### `project/mdp/reward.py`
- Reward = CF + valuation growth − risk penalties
- Terminal value = `s_T * (FV_T − D_T)`

### `project/mdp/env.py`
- Step function ties together mask → transitions → reward.
- **Updated**: `reset(use_data=True)` to initialize from cleaned data.

---

## Solvers

### `project/solvers/mcts.py`
- Monte Carlo Tree Search for planning (Q1/Q2) with heuristic value and candidate rollouts.

### `project/solvers/heuristic_value.py`
- Heuristic value model used by MCTS for leaf evaluation.

### `project/solvers/rl_ppo.py`
- Lightweight PPO (tabular-linear policy) for adaptive strategies (Q3/Q4).

### `project/solvers/eval.py`
- Rollout evaluation utilities for comparing policies.

---

## Experiments (data-driven by default)

### `project/experiments/utils.py`
- Builds `MDPEnv` using calibrated config + data-driven state.

### `project/experiments/q1_leverage_policy_map.py`
- Produces leverage policy map under macro & CF buckets.

### `project/experiments/q2_recruitment_strategy.py`
- Builds Draft/FA/Trade candidate pools from real players and ranks by owner-value score (ΔQ, win gain, star premium, cost proxy).

### `project/experiments/q3_expansion_site_sensitivity.py`
- Tests expansion site scenarios; compares Indiana baseline vs PPO, and outputs league winners/losers based on market-size (attendance proxy) + team-strength (player-based ELO) + local competition.

### `project/experiments/q4_dynamic_ticket_or_equity.py`
- Dynamic ticket/equity policy analysis with PPO using strict action masking (only target dimension can move). Outputs `q4_dynamic_policy_summary_ticket.md` and `..._equity.md`.

### `project/experiments/q5_letter_generator.py`
- Generates the owner/GM recommendation letter using Q1–Q4 outputs.

---

## Known data limits & assumptions
- Ticket price history and marketing spend are not available; elasticity and marketing lift are estimated from attendance residual variance.
- Valuations dataset has limited revenue history; base revenue for 2025 uses 2024 revenue when available.
- Competitive state `Q_t` is player-based when `allplayers.csv` is available; falls back to team-level ORtg/DRtg/NetRtg/SRS only if player data is missing.

---

## Action discretization (refined)
We expanded the 6-D action vector to increase control precision:
- **roster**: 7 levels (seller_aggressive → buyer_aggressive)
- **salary**: 6 levels (floor → max_apron)
- **ticket**: 7 levels (0.85x → 1.30x)
- **marketing**: 4 levels (0%, 3%, 6%, 10%)
- **debt**: 5 levels (−8%, −4%, 0, +4%, +8% of FV)
- **equity**: 5 levels (0%, 0.5%, 1%, 2%, 3%)

These refinements required updates to:
- `project/mdp/action.py` (ranges + labels)
- `project/mdp/config.py` (mapping arrays)
- `project/mdp/state.py` (K normalization)
- `project/mdp/transitions_comp.py` / `project/mdp/transitions_fin.py` (threshold logic)

---

## How to re-run the pipeline
1. **Clean & audit**
   - `python project/data/clean_and_audit.py`
2. **Calibrate from data**
   - `python project/data/run_calibration.py`
3. **Run experiments**
   - `python project/experiments/q1_leverage_policy_map.py`
   - `python project/experiments/q2_recruitment_strategy.py`
   - `python project/experiments/q3_expansion_site_sensitivity.py`
   - `python project/experiments/q4_dynamic_ticket_or_equity.py --mode ticket`
   - `python project/experiments/q4_dynamic_ticket_or_equity.py --mode equity`
   - `python project/experiments/q5_letter_generator.py`

---

## Current outputs
- `project/data/clean/data_audit.md`
- `project/data/clean/calibration_report.md`
- `project/experiments/output/*.csv` and `*.md`
