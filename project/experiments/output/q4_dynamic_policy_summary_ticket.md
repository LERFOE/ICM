# Q4 Dynamic Business Decision

Mode: ticket

Baseline vs Learned Policy (only target dimension changes; others fixed to K):
- Baseline avg_terminal=439.90, avg_cf=18.47
- Learned  avg_terminal=439.90, avg_cf=18.57

Most common learned action vectors:
- (3, 2, 4, 1, 1, 0): 172 steps
- (3, 2, 3, 1, 1, 0): 24 steps
- (3, 2, 6, 1, 1, 0): 4 steps

Interpretation:
- Ticket decisions trade off short-term gate revenue vs long-term brand growth.
- Equity decisions trade off cash relief vs permanent dilution.
