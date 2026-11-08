import numpy as np
import pandas as pd
import pytest

from connectroute.towers import match_nearest_towers


def test_match_nearest_towers_returns_ranked_projected_features():
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
    tower_lon = np.array([13.0, 13.01, 13.02])
    tower_lat = np.array([47.0, 47.0, 47.0])
    tower_x, tower_y = transformer.transform(tower_lon, tower_lat)
    measurement_x, measurement_y = transformer.transform(13.009, 47.0)
    measurements = pd.DataFrame(
        {
            "easting_m": [measurement_x],
            "northing_m": [measurement_y],
            "timestamp": [pd.Timestamp("2019-01-01")],
        }
    )
    towers = pd.DataFrame(
        {
            "mcc": [232, 232, 232],
            "net": [1, 1, 1],
            "area": [11, 12, 13],
            "cell": [101, 102, 103],
            "unit": [-1, -1, -1],
            "latitude": tower_lat,
            "longitude": tower_lon,
            "range": [1000, 2000, 3000],
            "samples": [10, 1, 20],
            "created": pd.to_datetime(["2018-01-01", "2018-01-01", "2018-01-01"]),
            "updated": pd.to_datetime(["2018-12-01", "2018-12-01", "2018-12-01"]),
        }
    )

    matches, summary = match_nearest_towers(measurements, towers, k=2, minimum_samples=5)

    assert matches.loc[0, "tower_1_cell"] == 101
    assert matches.loc[0, "tower_2_cell"] == 103
    assert matches.loc[0, "tower_1_distance_m"] < matches.loc[0, "tower_2_distance_m"]
    assert matches.loc[0, "tower_1_samples"] == 10
    assert matches.loc[0, "tower_1_range"] == 1000
    assert matches.loc[0, "tower_1_created_age_at_measurement_days"] > 0
    assert summary["eligible_tower_rows"] == 2
    assert summary["k"] == 2


def test_match_nearest_towers_keeps_signed_negative_age_for_future_records():
    measurements = pd.DataFrame(
        {"easting_m": [370000.0], "northing_m": [5301000.0], "timestamp": [pd.Timestamp("2019-01-01")]}
    )
    towers = pd.DataFrame(
        {
            "mcc": [232],
            "net": [1],
            "area": [11],
            "cell": [101],
            "unit": [-1],
            "latitude": [47.85],
            "longitude": [13.2],
            "range": [1000],
            "samples": [10],
            "created": [pd.Timestamp("2020-01-01")],
            "updated": [pd.Timestamp("2021-01-01")],
        }
    )

    matches, summary = match_nearest_towers(measurements, towers, k=1)

    assert matches.loc[0, "tower_1_created_age_at_measurement_days"] < 0
    assert summary["negative_created_age_matches"] == 1


def test_match_nearest_towers_rejects_insufficient_eligible_towers():
    measurements = pd.DataFrame({"easting_m": [370000.0], "northing_m": [5301000.0]})
    towers = pd.DataFrame(
        {
            "mcc": [232],
            "net": [1],
            "area": [11],
            "cell": [101],
            "unit": [-1],
            "latitude": [47.85],
            "longitude": [13.2],
            "range": [1000],
            "samples": [1],
            "created": [pd.Timestamp("2020-01-01")],
            "updated": [pd.Timestamp("2021-01-01")],
        }
    )

    with pytest.raises(ValueError, match="No towers satisfy"):
        match_nearest_towers(measurements, towers, minimum_samples=5)
