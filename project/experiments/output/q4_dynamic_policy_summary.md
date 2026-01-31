# Q4 Dynamic Business Decision

Mode: ticket

Baseline vs Learned Policy:
- Baseline avg_terminal=902.62, avg_cf=11.95
- Learned  avg_terminal=47.99, avg_cf=-3.51

Most common learned action vectors:
- (1, 3, 0, 1, 2, 3): 8 steps
- (0, 3, 2, 1, 2, 3): 6 steps
- (0, 3, 4, 1, 2, 3): 4 steps
- (0, 3, 0, 1, 2, 3): 4 steps
- (1, 3, 4, 1, 2, 3): 4 steps

Interpretation:
- Ticket decisions trade off short-term gate revenue vs long-term brand growth.
- Equity decisions trade off cash relief vs permanent dilution.
