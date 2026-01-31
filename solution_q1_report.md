# Solution Report: Question 1 Dynamic Decision Model

## 1. Problem Statement
Design a dynamic decision model to adjust leverage based on team performance and economic conditions (Recession/Boom) to maximize Profit and Franchise Value.

## 2. Model Implementation
We implemented a **Markov Decision Process (MDP)** framework with the following components:

### A. Factor-Based Engine (Team Operations)
- **Engine**: `src_experiment/game_engine_factor.py`
- **Logic**: Uses a 4-Factor Model (Scoring, Playmaking, Defense, Gravity) to simulate realistic basketball outcomes.
- **Connection**: Better team performance (Win%) leads to higher Revenue and Valuation Growth.

### B. Financial Dynamics (Business Operations)
- **State**: Tracks Cash flow, Debt stock, and Franchise Value.
- **Leverage Logic**: 
    - **Interest Rate**: $r = r_{free} + \beta \times \lambda_t$. High leverage increases cost of capital.
    - **Actions**: Paydown Debt (Deleverage), Maintain, Borrow (Leverage Up).

### C. Economic Environment
- **Macro Cycle**: Simulates `Recession` (-20% Revenue), `Normal`, and `Boom` (+20% Revenue) cycles.
- **Impact**: Macro state directly affects the "Return on Investment" (ROI) of signing expensive players.

## 3. Experiment Results Analysis

We ran a 10-year simulation forcing a specific economic pattern:
- **Years 1-3 (Normal)**: The agent started with 10% leverage. It prioritized **Deleveraging** (Debt:0) and Selling overvalued assets (Rost:2) to build a cash fortress.
- **Years 4-6 (Recession)**: 
    - Revenue dropped significantly.
    - **Agent Decision**: The solver correctly identified the risk. It chose **Debt:0 (Paydown)** and **Rost:0 (Hold)** or Selling.
    - result: Despite the recession, the team remained profitable ($2.0M) and Value stabilized.
- **Years 7-10 (Boom)**:
    - Revenue surged.
    - **Agent Decision**: The solver shifted strategy. In Year 7-9, it often chose **Debt:2 (Borrow)** or Hold to fund operations and maximize growth, riding the valuation wave from 600M to 862M.

## 4. Conclusion
The model successfully demonstrates "Dynamic Leverage Adjustment":
1.  **Defensive Mode**: In Recession, it reduces leverage to minimize interest expense and preserve equity value.
2.  **Offensive Mode**: In Boom, it utilizes capital more aggressively (or holds debt) to capture valuation growth.
3.  **Holistic Management**: The decisions integrate both Team Ops (Roster) and Business Ops (Debt) into a single objective function.
