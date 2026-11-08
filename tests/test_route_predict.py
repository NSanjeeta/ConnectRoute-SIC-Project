import pandas as pd
import pytest

from connectroute.route_predict import predict_route_signal


def _towers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "radio": ["LTE", "LTE", "GSM", "LTE"],
            "latitude": [12.0, 12.02, 12.0, 12.1],
            "longitude": [77.0, 77.001, 77.001, 77.0],
            "samples": [10, 20, 100, 2],
        }
    )


def test_route_prediction_uses_nearest_eligible_lte_and_marks_extrapolation():
    samples = [
        {
            "sample_index": 0,
            "distance_along_route_m": 0.0,
            "latitude": 12.0,
            "longitude": 77.0005,
        },
        {
            "sample_index": 1,
            "distance_along_route_m": 100.0,
            "latitude": 12.001,
            "longitude": 77.0005,
        },
    ]
    model = {
        "intercept_dbm_at_1m": -30.0,
        "slope_db_per_decade": -20.0,
        "path_loss_exponent": 2.0,
        "training_rows": 100,
    }
    residuals = {
        "residual_p05_db": -10.0,
        "residual_p95_db": 10.0,
        "residual_p10_db": -7.0,
        "residual_p90_db": 7.0,
        "oof_residual_rows": 100,
        "training_distance_min_m": 50.0,
        "training_distance_max_m": 30.0,
    }

    profile, summary = predict_route_signal(samples, _towers(), model, residuals)

    assert len(profile) == 2
    assert profile["predicted_rsrp_dbm"].notna().all()
    assert profile["nearest_tower_distance_m"].iloc[0] < 60
    assert profile["nearest_tower_samples"].iloc[0] in (10, 20)
    assert profile["estimated_upper_90_dbm"].iloc[0] - profile[
        "estimated_lower_90_dbm"
    ].iloc[0] == pytest.approx(20)
    assert summary["eligible_lte_towers"] == 2
    assert summary["outside_salzburg_distance_range_fraction"] == 1


def test_route_prediction_rejects_missing_eligible_towers():
    with pytest.raises(ValueError, match="No LTE towers"):
        predict_route_signal(
            [
                {
                    "distance_along_route_m": 0.0,
                    "latitude": 12.0,
                    "longitude": 77.0,
                }
            ],
            _towers(),
            {"intercept_dbm_at_1m": -30, "slope_db_per_decade": -20},
            {
                "residual_p05_db": -5,
                "residual_p95_db": 5,
                "training_distance_min_m": 10,
                "training_distance_max_m": 1000,
            },
            minimum_tower_samples=1000,
        )


def test_route_prediction_rejects_route_outside_local_tower_extent():
    with pytest.raises(ValueError, match="outside the geographic extent"):
        predict_route_signal(
            [
                {
                    "distance_along_route_m": 0.0,
                    "latitude": 53.48,
                    "longitude": -2.24,
                }
            ],
            _towers(),
            {"intercept_dbm_at_1m": -30, "slope_db_per_decade": -20},
            {
                "residual_p05_db": -5,
                "residual_p95_db": 5,
                "training_distance_min_m": 10,
                "training_distance_max_m": 1000,
            },
        )


def test_route_projection_model_uses_salzburg_oof_residual_ranges(tmp_path):
    from connectroute.route_predict import fit_route_projection_model

    features_path = tmp_path / "features.parquet"
    predictions_path = tmp_path / "predictions.parquet"
    pd.DataFrame(
        {
            "tower_1_distance_m": [10.0, 100.0, 1000.0],
            "rsrp_dbm": [-50.0, -70.0, -90.0],
        }
    ).to_parquet(features_path)
    pd.DataFrame(
        {
            "rsrp_dbm": [-52.0, -68.0, -91.0, -70.0],
            "predicted_fitted_log_distance_rsrp_dbm": [-50.0, -70.0, -90.0, -72.0],
        }
    ).to_parquet(predictions_path)

    model, residuals = fit_route_projection_model(features_path, predictions_path)

    assert model["intercept_dbm_at_1m"] == pytest.approx(-30)
    assert model["slope_db_per_decade"] == pytest.approx(-20)
    assert residuals["residual_p05_db"] <= residuals["residual_p95_db"]
    assert residuals["oof_residual_rows"] == 4
