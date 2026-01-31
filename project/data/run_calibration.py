import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.data.integration import calibrate_config_from_data
from project.mdp.config import MDPConfig

OUT = Path(__file__).resolve().parent / "clean" / "calibration_report.md"


def main():
    cfg, summary = calibrate_config_from_data(MDPConfig())
    lines = ["# Calibration Report", ""]
    lines.append("## Revenue/Attendance Calibration")
    lines.append(f"attendance_slope: {summary.attendance_slope:.2f}")
    lines.append(f"attendance_intercept: {summary.attendance_intercept:.2f}")
    lines.append(f"attendance_method: {summary.attendance_method}")
    lines.append(f"rev_win_beta: {cfg.rev_win_beta:.3f}")
    lines.append(f"rev_win_beta_scale: {cfg.rev_win_beta_scale:.2f}")
    lines.append(f"lasso_alpha: {cfg.lasso_alpha:.2f}")
    lines.append(f"ticket_elasticity: {cfg.ticket_elasticity:.3f}")
    lines.append(f"marketing_attendance_beta: {cfg.marketing_attendance_beta:.3f}")
    lines.append("")
    lines.append("## ELO Win% Logistic Calibration")
    lines.append(f"elo_b0: {summary.elo_b0:.3f}")
    lines.append(f"elo_b1 (ELO diff): {summary.elo_b1:.3f}")
    lines.append(f"elo_b2 (SOS): {summary.elo_b2:.3f}")
    lines.append(f"elo_samples: {summary.elo_samples}")
    lines.append("")
    lines.append("## Market & Revenue Base")
    lines.append(f"market_size(mu): {cfg.market_size:.3f}")
    lines.append(f"base_gate_revenue: {cfg.base_gate_revenue:.2f} M")
    lines.append(f"base_media_revenue: {cfg.base_media_revenue:.2f} M")
    lines.append(f"base_sponsor_revenue: {cfg.base_sponsor_revenue:.2f} M")
    lines.append(f"franchise_value: {cfg.base_franchise_value:.2f} M")
    lines.append(f"base_debt: {cfg.base_debt:.2f} M")
    if summary.revenue_total:
        lines.append(f"revenue_total (valuation file): {summary.revenue_total:.2f} M")
    lines.append("")
    lines.append("## Assumptions")
    lines.append(f"gate/media/sponsor ratios: {summary.gate_ratio:.2f}/{summary.media_ratio:.2f}/{summary.sponsor_ratio:.2f}")
    if summary.notes:
        lines.append(f"notes: {summary.notes}")

    OUT.write_text("\n".join(lines))
    print(f"Wrote calibration report to {OUT}")


if __name__ == "__main__":
    main()
