class FinancialModel:
    def __init__(self):
        # Calibrated Linear Model for Attendance
        # Avg_Att = gamma * Win% + delta
        self.gamma_att = 5768.0
        self.delta_att = 4108.0
        
        # Revenue Per Attendee (derived from 2024 IND approx: 33.8M Rev / 340k Total Att)
        self.rev_per_attendee = 100.0 # USD
        
        # Costs (Millions)
        self.cost_charter_flights = 2.1
        self.cost_venue = 1.0
        self.cost_fixed_ops = 2.0 # General Admin
        
        # Salary Cap (Soft) - WNBA 2025 approx $1.5M
        # Used for Luxury Tax calc if needed
        self.salary_cap = 1.5 

    def calculate_economics(self, win_pct, payroll_millions, marketing_boost=1.0):
        """
        marketing_boost: Multiplier for 'Caitlin Clark Effect' or similar structural shifts.
        For normal years, 1.0. For 2024-like years, maybe 2.5.
        """
        
        # 1. Attendance
        base_avg_att = (self.gamma_att * win_pct) + self.delta_att
        avg_att = base_avg_att * marketing_boost
        
        # Cap at venue capacity ~18,000
        avg_att = min(18000, avg_att)
        
        total_att = avg_att * 20.0 # 20 Home Games
        
        # 2. Revenue
        revenue = (total_att * self.rev_per_attendee) / 1e6 # Convert to Millions
        
        # 3. Expenses
        # Payroll + Charter + Venue + Fixed
        total_expenses = payroll_millions + self.cost_charter_flights + self.cost_venue + self.cost_fixed_ops
        
        # 4. Cash Flow
        operating_income = revenue - total_expenses
        
        # 5. Valuation (Simple Multiple)
        # Using 5x Revenue conservative estimate for growth league
        valuation = revenue * 5.0
        
        return {
            'Avg_Att': int(avg_att),
            'Total_Att': int(total_att),
            'Revenue_M': round(revenue, 2),
            'Expenses_M': round(total_expenses, 2),
            'Operating_Income_M': round(operating_income, 2),
            'Valuation_M': round(valuation, 2)
        }
