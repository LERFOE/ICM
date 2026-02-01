from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import math


@dataclass(frozen=True)
class Location:
    name: str
    lat: float
    lon: float


TEAM_LOCATIONS: Dict[str, Location] = {
    "ATL": Location("Atlanta", 33.7490, -84.3880),
    "CHI": Location("Chicago", 41.8781, -87.6298),
    "CON": Location("Uncasville", 41.4340, -72.1100),
    "DAL": Location("Dallas", 32.7767, -96.7970),
    "IND": Location("Indianapolis", 39.7684, -86.1581),
    "LVA": Location("Las Vegas", 36.1699, -115.1398),
    "LAS": Location("Los Angeles", 34.0522, -118.2437),
    "MIN": Location("Minneapolis", 44.9778, -93.2650),
    "NYL": Location("New York", 40.7128, -74.0060),
    "PHO": Location("Phoenix", 33.4484, -112.0740),
    "SEA": Location("Seattle", 47.6062, -122.3321),
    "WAS": Location("Washington", 38.9072, -77.0369),
    "GSV": Location("San Francisco", 37.7749, -122.4194),
}


EXPANSION_CANDIDATES = [
    {
        "name": "Columbus",
        "lat": 39.9612,
        "lon": -82.9988,
        "market_score": 0.55,
        "hub_score": 0.65,
        "note": "Near IND, mid market, high overlap",
    },
    {
        "name": "StLouis",
        "lat": 38.6270,
        "lon": -90.1994,
        "market_score": 0.58,
        "hub_score": 0.70,
        "note": "Midwest hub, moderate distance, schedule-efficient",
    },
    {
        "name": "Nashville",
        "lat": 36.1627,
        "lon": -86.7816,
        "market_score": 0.60,
        "hub_score": 0.55,
        "note": "Southern growth market, moderate distance",
    },
    {
        "name": "Denver",
        "lat": 39.7392,
        "lon": -104.9903,
        "market_score": 0.62,
        "hub_score": 0.70,
        "note": "Mountain hub, cross-region travel, network balance",
    },
    {
        "name": "Portland",
        "lat": 45.5152,
        "lon": -122.6784,
        "market_score": 0.60,
        "hub_score": 0.35,
        "note": "Far west coast, market potential, high travel cost",
    },
    {
        "name": "Toronto",
        "lat": 43.6532,
        "lon": -79.3832,
        "market_score": 0.85,
        "hub_score": 0.60,
        "note": "International large market, strong media bonus",
    },
]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def _distance_km(loc_a: Location, loc_b: Location) -> float:
    return haversine_km(loc_a.lat, loc_a.lon, loc_b.lat, loc_b.lon)


def build_expansion_sites(team_locations: Dict[str, Location] | None = None) -> List[Dict[str, float | int | str]]:
    team_locations = team_locations or TEAM_LOCATIONS
    ind_loc = team_locations["IND"]
    team_list = list(team_locations.values())
    centroid_lat = sum(t.lat for t in team_list) / len(team_list)
    centroid_lon = sum(t.lon for t in team_list) / len(team_list)

    # Max distance from centroid to normalize centrality
    max_centroid_dist = 1.0
    for t in team_list:
        max_centroid_dist = max(max_centroid_dist, haversine_km(t.lat, t.lon, centroid_lat, centroid_lon))

    sites: List[Dict[str, float | int | str]] = []
    for cand in EXPANSION_CANDIDATES:
        loc = Location(cand["name"], cand["lat"], cand["lon"])
        dist_to_ind = _distance_km(ind_loc, loc)
        dist_to_centroid = haversine_km(loc.lat, loc.lon, centroid_lat, centroid_lon)

        nearby = 0
        nearest = 1e9
        for t in team_list:
            d = _distance_km(t, loc)
            nearest = min(nearest, d)
            if d <= 600.0:
                nearby += 1
        ind_overlap = max(0.0, 1.0 - dist_to_ind / 600.0)
        local_overlap = max(0.0, 1.0 - nearest / 600.0)

        market_score = float(cand["market_score"])
        hub_score = float(cand["hub_score"])
        centrality = 1.0 - min(1.0, dist_to_centroid / max_centroid_dist)

        market_delta = 0.02 * market_score - 0.03 * ind_overlap - 0.01 * local_overlap
        compete_delta = 1 if ind_overlap > 0.30 else 0
        media_bonus = 0.02 + 0.03 * market_score + 0.01 * hub_score
        travel_fatigue = min(1.0, dist_to_ind / 3000.0) * (1.0 - 0.2 * hub_score)

        sites.append(
            {
                "name": cand["name"],
                "lat": loc.lat,
                "lon": loc.lon,
                "market_score": market_score,
                "hub_score": hub_score,
                "centrality": round(centrality, 4),
                "dist_to_ind_km": round(dist_to_ind, 1),
                "nearby_teams": int(nearby),
                "nearest_team_km": round(nearest, 1),
                "market_delta": round(market_delta, 4),
                "compete_delta": int(compete_delta),
                "media_bonus": round(media_bonus, 4),
                "travel_fatigue": round(travel_fatigue, 4),
                "note": cand.get("note", ""),
            }
        )

    return sites
