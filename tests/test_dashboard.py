import numpy as np
import pandas as pd
import pytest

from connectroute.dashboard import (
    find_hotspots,
    profile_metrics,
    recommend_tower_sites,
    score_routes,
)


def _profile() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "distance_along_route_m": [0, 500, 1000, 1500, 2000, 2500],
            "predicted_rsrp_dbm": [-80, -105, -112, -108, -78, -115],
            "estimated_lower_90_dbm": [-90] * 6,
            "estimated_upper_90_dbm": [-70] * 6,
            "nearest_tower_distance_m": [100, 1200, 1400, 1300, 200, 1800],
            "latitude": np.linspace(12.0, 12.1, 6),
            "longitude": np.linspace(76.5, 76.6, 6),
        }
    )


def test_profile_metrics_are_distance_aware():
    metrics = profile_metrics(_profile())
    assert metrics["route_length_m"] == 2500
    assert metrics["outage_fraction"] == pytest.approx(0.2)
    assert metrics["longest_weak_stretch_m"] == pytest.approx(1500)


def test_hotspots_and_tower_recommendations_are_ranked():
    hotspots = find_hotspots(_profile(), minimum_length_m=400)
    assert list(hotspots["Hotspot"]) == [1]
    recommendations = recommend_tower_sites(_profile(), minimum_length_m=400)
    assert len(recommendations) == 1
    assert recommendations.iloc[0]["Priority"] == 1


def test_route_scores_follow_priority():
    rows = pd.DataFrame(
        {
            "Route": [1, 2],
            "Mean estimated RSRP (dBm)": [-85, -92],
            "Estimated route below -110 dBm (%)": [20, 5],
            "Distance (km)": [140, 150],
            "Travel time (min)": [120, 110],
        }
    )
    scored = score_routes(rows, priority="Lowest outage risk")
    assert scored.iloc[0]["Route"] == 2
    single_route = score_routes(rows.iloc[[0]], priority="Balanced")
    assert single_route.iloc[0]["Route score"] == pytest.approx(100)
    with pytest.raises(ValueError):
        score_routes(rows, priority="unknown")
