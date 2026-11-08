import numpy as np
import pandas as pd
import pytest

from connectroute.models import (
    _calibration_training_split,
    _conformal_adjustment,
    _fold_physics_baseline,
    _interval_metrics,
    _quantile_outage_probability,
    build_model_features,
)


def _feature_frame() -> pd.DataFrame:
    data: dict[str, list[float]] = {
        "tower_1_receiver_elevation_m": [400.0, 410.0],
        "best_server_3gpp_rma_los_path_loss_db": [95.0, 100.0],
        "best_server_3gpp_rma_los_in_domain": [1.0, 1.0],
        "tower_1_3gpp_rma_los_path_loss_db": [95.0, 100.0],
        "tower_1_3gpp_rma_los_in_domain": [1.0, 1.0],
        "rsrp_dbm": [-80.0, -90.0],
        "rsrq": [-10.0, -12.0],
        "sinr": [5.0, 3.0],
        "signal": [-80.0, -90.0],
        "rssi": [-70.0, -80.0],
        "datarate": [10.0, 20.0],
        "cell_id": [123.0, 456.0],
        "latitude": [47.0, 47.1],
        "longitude": [13.0, 13.1],
        "chainage_m": [0.0, 100.0],
        "tower_1_created_age_at_measurement_days": [-10.0, 20.0],
    }
    for rank in range(1, 4):
        data[f"tower_{rank}_distance_m"] = [100.0 * rank, 110.0 * rank]
        data[f"tower_{rank}_bearing_deg"] = (
            [359.0, 1.0] if rank == 1 else [1.0, 359.0]
        )
        data[f"tower_{rank}_samples"] = [10.0 + rank, 20.0 + rank]
        data[f"tower_{rank}_range"] = [100.0 + rank, 200.0 + rank]
        data[f"tower_{rank}_fresnel_clearance_fraction"] = [1.0, 0.5]
        data[f"tower_{rank}_knife_edge_diffraction_loss_db"] = [0.0, 3.0]
        data[f"tower_{rank}_terrain_roughness_m"] = [2.0, 4.0]
        data[f"tower_{rank}_tower_ground_elevation_m"] = [420.0 + rank, 430.0 + rank]
    return pd.DataFrame(data)


def test_model_features_separate_pure_ml_from_physics_and_exclude_leakage():
    features, pure_columns, hybrid_columns = build_model_features(_feature_frame())

    assert "tower_1_distance_log10" in pure_columns
    assert "tower_1_2_bearing_separation_deg" in pure_columns
    assert "tower_1_elevation_difference_m" in pure_columns
    assert all("path_loss" not in column for column in pure_columns)
    assert "best_server_3gpp_rma_los_path_loss_db" in hybrid_columns
    assert "tower_1_3gpp_rma_los_in_domain" in hybrid_columns
    assert not any(
        token in column
        for column in pure_columns + hybrid_columns
        for token in ("rsrp", "rsrq", "sinr", "signal", "rssi", "datarate", "cell_id", "latitude", "longitude", "chainage", "created_age")
    )
    assert features["tower_1_2_bearing_separation_deg"].tolist() == pytest.approx([2.0, 2.0])


def test_calibration_split_holds_out_distinct_spatial_folds_and_buffers():
    chainage = np.arange(0.0, 10_000.0, 100.0)
    folds = (chainage // 2000).astype(np.int64)

    training, calibration, test = _calibration_training_split(
        chainage,
        folds,
        test_fold=2,
        calibration_fold=1,
        buffer_m=500.0,
    )

    assert set(folds[calibration]) == {1}
    assert set(folds[test]) == {2}
    assert not set(training).intersection(calibration)
    assert not set(training).intersection(test)
    assert np.all((chainage[training] < 1500) | (chainage[training] > 6400))
    assert len(calibration) > 0
    assert len(test) > 0


def test_conformal_adjustment_uses_finite_sample_order_statistic():
    scores = np.array([4.0, 1.0, 3.0, 2.0])

    assert _conformal_adjustment(scores, alpha=0.4) == 3.0
    assert _conformal_adjustment(scores, alpha=0.2) == 4.0
    with pytest.raises(ValueError, match="between zero and one"):
        _conformal_adjustment(scores, alpha=1.0)


def test_quantile_outage_probability_interpolates_and_clips():
    levels = np.array([0.05, 0.10, 0.50, 0.90, 0.95])
    predictions = np.array(
        [
            [-120.0, -110.0, -80.0, -70.0, -60.0],
            [-120.0, -110.0, -80.0, -70.0, -60.0],
            [-120.0, -110.0, -80.0, -70.0, -60.0],
        ]
    )

    result = _quantile_outage_probability(
        predictions,
        levels,
        threshold_dbm=-100.0,
    )

    assert result.tolist() == pytest.approx([0.10 + (10 / 30) * 0.4] * 3)
    assert _quantile_outage_probability(predictions, levels, -130.0).tolist() == [0, 0, 0]
    assert _quantile_outage_probability(predictions, levels, -50.0).tolist() == [1, 1, 1]


def test_interval_metrics_report_coverage_width_and_score():
    metrics = _interval_metrics(
        np.array([0.0, 2.0, 5.0]),
        np.array([-1.0, 1.0, 0.0]),
        np.array([1.0, 3.0, 4.0]),
        alpha=0.2,
    )

    assert metrics["nominal_coverage"] == pytest.approx(0.8)
    assert metrics["empirical_coverage"] == pytest.approx(2 / 3)
    assert metrics["mean_width_db"] == pytest.approx(8 / 3)
    assert metrics["mean_interval_score_db"] == pytest.approx((2 + 2 + 14) / 3)


def test_fitted_log_distance_baseline_uses_training_targets_only():
    distance = np.array([10.0, 100.0, 1000.0, 50.0])
    target = np.array([-50.0, -70.0, -90.0, 300.0])

    prediction, fitted = _fold_physics_baseline(
        target,
        distance,
        np.array([0, 1, 2]),
        np.array([3]),
    )
    changed = target.copy()
    changed[3] = -5000.0
    changed_prediction, changed_fit = _fold_physics_baseline(
        changed,
        distance,
        np.array([0, 1, 2]),
        np.array([3]),
    )

    assert fitted == changed_fit
    assert prediction.tolist() == pytest.approx(changed_prediction.tolist())
