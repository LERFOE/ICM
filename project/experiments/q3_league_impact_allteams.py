import csv
import sys
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.experiments.q3_expansion_site_sensitivity import (
    TEAM_CODE_TO_NAME,
    _market_index,
    _strength_index,
)
from project.data.geo import TEAM_LOCATIONS, build_expansion_sites, haversine_km
from project.data.player_kmeans import build_player_model
from project.mdp.config import MDPConfig


OUTPUT_CSV = Path("project/experiments/output/q3_league_impact_allteams.csv")
OUTPUT_MD = Path("project/experiments/output/q3_league_impact_summary.md")


def _impact_components(site: Dict, market_idx: Dict[str, float], strength_idx: Dict[str, float], cfg: MDPConfig):
    rows = []
    scarcity = abs(cfg.expansion_star_fa_delta)
    for team in market_idx:
        m = market_idx[team]
        s = strength_idx.get(team, 0.0)
        loc_team = TEAM_LOCATIONS.get(team)
        if loc_team is None:
            dist_km = 1500.0
        else:
            dist_km = haversine_km(loc_team.lat, loc_team.lon, site["lat"], site["lon"])
        overlap = max(0.0, 1.0 - dist_km / 600.0)
        travel_term = -0.02 * (dist_km / 3000.0) + 0.01 * float(site["hub_score"])

        media_term = 0.6 * site["media_bonus"] * m
        market_term = (site["market_delta"] - 0.03 * overlap) * m
        compete_term = -0.05 * site["compete_delta"] * (1.0 - m)
        scarcity_term = -0.02 * scarcity * (1.0 - s)
        impact = media_term + market_term + compete_term + travel_term + scarcity_term
        rows.append(
            {
                "site": site["name"],
                "team": team,
                "team_name": TEAM_CODE_TO_NAME.get(team, team),
                "impact_score": impact,
                "dist_km": dist_km,
                "overlap": overlap,
                "market_idx": m,
                "strength_idx": s,
                "scarcity": scarcity,
                "media_term": media_term,
                "market_term": market_term,
                "compete_term": compete_term,
                "travel_term": travel_term,
                "scarcity_term": scarcity_term,
            }
        )
    return rows


def main():
    model = build_player_model()
    market_idx = _market_index()
    strength_idx = _strength_index(model)
    cfg = MDPConfig()

    sites = build_expansion_sites()
    rows = []
    for site in sites:
        rows.extend(_impact_components(site, market_idx, strength_idx, cfg))

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Summary: top3 and bottom3 per site
    with OUTPUT_MD.open("w") as f:
        f.write("# Q3 League Impact Summary (All Teams)\n\n")
        for site in sites:
            sub = [r for r in rows if r["site"] == site["name"]]
            sub_sorted = sorted(sub, key=lambda x: x["impact_score"], reverse=True)
            top3 = sub_sorted[:3]
            bot3 = sub_sorted[-3:]
            f.write(f"## {site['name']}\n")
            f.write("Top3 (most favorable):\n")
            for r in top3:
                f.write(f"- {r['team']} ({r['team_name']}): {r['impact_score']:.4f}\n")
            f.write("Bottom3 (most unfavorable):\n")
            for r in bot3:
                f.write(f"- {r['team']} ({r['team_name']}): {r['impact_score']:.4f}\n")
            f.write("\n")

    print(f"Saved league impact to {OUTPUT_CSV} and {OUTPUT_MD}")


if __name__ == "__main__":
    main()
