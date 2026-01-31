# Data Audit (Cleaned)
## spotrac_ind_yearly_clean.csv
shape: (16, 16)
columns:
Player, Pos, Age, Protection, 2025_salary_m, 2025_pct, 2025_status, 2026_salary_m, 2026_pct, 2026_status, 2027_salary_m, 2027_pct, 2027_status, 2028_salary_m, 2028_pct, 2028_status
missing_ratio:
- 2028_pct: 1.00
- 2028_salary_m: 0.94
- 2027_salary_m: 0.88
- 2027_pct: 0.88
- 2026_salary_m: 0.81
- 2026_pct: 0.81
- Protection: 0.06

## ind_team_season_stats.csv
shape: (26, 13)
columns:
Year, Team, W, L, W/L%, Finish, SRS, Pace, ORtg, DRtg, Coaches, Playoffs Result, Top WS
missing_ratio:
- Playoffs Result: 0.42

## ind_master_wide_clean.csv
shape: (26, 13)
columns:
Season, IND__Team, IND__W, IND__L, IND__W/L%, IND__Finish, IND__SRS, IND__Pace, IND__ORtg, IND__DRtg, IND__Coaches, IND__Playoffs Result, IND__Top WS
missing_ratio:
- IND__Top WS: 1.00
- IND__Playoffs Result: 0.42

## wnba_attendance_clean.csv
shape: (107, 5)
columns:
Season, Team, GP, Total_Attendance, Avg_Attendance
missing_ratio: none

## wnba_advanced_stats_clean.csv
shape: (120, 10)
columns:
Season, Team, W, L, Win%, ORtg, DRtg, NetRtg, Pace, SRS
missing_ratio:
- W: 0.01
- L: 0.01
- Win%: 0.01
- SRS: 0.01

## ind_elo_game_clean.csv
shape: (244, 20)
columns:
Season, Date, home_abbr, away_abbr, home_pts, away_pts, home_win, opponent_abbr, ELO_t, opp_pre_elo_t, O_mean, O_var, O_p90, SOS_t, elo_avg_pre, O_t_list_json, O_len, O_mean_calc, O_var_calc, O_p90_calc
missing_ratio:
- O_mean: 0.10
- O_var: 0.10
- O_var_calc: 0.10
- O_mean_calc: 0.10
- SOS_t: 0.10
- O_p90: 0.10
- O_p90_calc: 0.10

## ind_elo_season_clean.csv
shape: (24, 5)
columns:
Season, ELO_last_pregame, SOS_mean, O_mean_mean, games
missing_ratio:
- SOS_mean: 0.04
- O_mean_mean: 0.04

## wnba_valuations_clean.csv
shape: (7, 4)
columns:
Year, Team, Valuation_M, Revenue_M
missing_ratio:
- Revenue_M: 0.71
