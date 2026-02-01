# Q3 Expansion Policy Comparison (Baseline vs PPO vs Heuristics)

Metrics are reported at the end of each season (1–3).

Policies:
- baseline: keep DEFAULT_ACTION (mask-aware + nearest-valid)
- aggressive: high salary + buyer_aggressive + leverage_high
- defensive: roster stability + marketing + deleverage_high
- ppo: learned adaptive policy under the same constraints

Seeds=2, PPO train episodes=8.

See `project/experiments/output/q3_policy_comparison_summary.csv` for season-end metrics.

## Expansion-Year Offseason Action Summary (Year=2026)

- Columbus/aggressive: a_debt=2.00±0.00, a_salary=5.00±0.00, a_roster=6.00±0.00
- Columbus/baseline: a_debt=2.00±0.00, a_salary=2.00±0.00, a_roster=3.00±0.00
- Columbus/defensive: a_debt=1.00±0.00, a_salary=1.00±0.00, a_roster=3.00±0.00
- Columbus/ppo: a_debt=1.00±0.00, a_salary=5.00±0.00, a_roster=4.00±0.00
- Denver/aggressive: a_debt=2.00±0.00, a_salary=5.00±0.00, a_roster=6.00±0.00
- Denver/baseline: a_debt=2.00±0.00, a_salary=2.00±0.00, a_roster=3.00±0.00
- Denver/defensive: a_debt=1.00±0.00, a_salary=1.00±0.00, a_roster=3.00±0.00
- Denver/ppo: a_debt=1.00±0.00, a_salary=5.00±0.00, a_roster=2.00±0.00
- Nashville/aggressive: a_debt=2.00±0.00, a_salary=5.00±0.00, a_roster=6.00±0.00
- Nashville/baseline: a_debt=2.00±0.00, a_salary=2.00±0.00, a_roster=3.00±0.00
- Nashville/defensive: a_debt=1.00±0.00, a_salary=1.00±0.00, a_roster=3.00±0.00
- Nashville/ppo: a_debt=1.50±0.50, a_salary=4.00±0.00, a_roster=6.00±0.00
- Portland/aggressive: a_debt=2.00±0.00, a_salary=5.00±0.00, a_roster=6.00±0.00
- Portland/baseline: a_debt=2.00±0.00, a_salary=2.00±0.00, a_roster=3.00±0.00
- Portland/defensive: a_debt=1.00±0.00, a_salary=1.00±0.00, a_roster=3.00±0.00
- Portland/ppo: a_debt=1.00±0.00, a_salary=5.00±0.00, a_roster=4.00±1.00
- StLouis/aggressive: a_debt=2.00±0.00, a_salary=5.00±0.00, a_roster=6.00±0.00
- StLouis/baseline: a_debt=2.00±0.00, a_salary=2.00±0.00, a_roster=3.00±0.00
- StLouis/defensive: a_debt=1.00±0.00, a_salary=1.00±0.00, a_roster=3.00±0.00
- StLouis/ppo: a_debt=0.50±0.50, a_salary=5.00±0.00, a_roster=2.00±0.00
- Toronto/aggressive: a_debt=2.00±0.00, a_salary=5.00±0.00, a_roster=6.00±0.00
- Toronto/baseline: a_debt=2.00±0.00, a_salary=2.00±0.00, a_roster=3.00±0.00
- Toronto/defensive: a_debt=1.00±0.00, a_salary=1.00±0.00, a_roster=3.00±0.00
- Toronto/ppo: a_debt=1.00±0.00, a_salary=3.50±1.50, a_roster=2.00±0.00
