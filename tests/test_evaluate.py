import numpy as np
import pandas as pd
import pytest

from connectroute.evaluate import (
    _metrics,
    _outage_report,
    _paired_block_bootstrap,
    _reliability_bins,
    build_benchmark_tables,
    evaluate_results,
)


def test_regression_metrics_ignore_nonfinite_predictions():
    result = _metrics(np.array([1.0, 3.0, 5.0]), np.array([1.0, 5.0, np.nan]))

    assert result["n"] == 2
    assert result["rmse_db"] == pytest.approx(np.sqrt(2))
    assert result["mae_db"] == pytest.approx(1.0)
    assert result["bias_db"] == pytest.approx(1.0)


def test_paired_block_bootstrap_is_reproducible_and_paired():
    actual = np.array([0.0, 2.0, 1.0, 3.0, -1.0, 2.0])
    baseline = actual + np.array([2.0, 2.0, 1.0, 1.0, 3.0, 3.0])
    candidate = actual + 0.5
    blocks = np.array([0, 0, 1, 1, 2, 2])

    first = _paired_block_bootstrap(
        actual,
        baseline,
        candidate,
        blocks,
        replicates=100,
        seed=42,
    )
    second = _paired_block_bootstrap(
        actual,
        baseline,
        candidate,
        blocks,
        replicates=100,
        seed=42,
    )

    assert first == second
    assert first["rmse_delta_candidate_minus_baseline_db"] < 0
    assert first["ci_95_upper_db"] < 0


def test_benchmark_tables_report_metrics_and_applicability():
    actual = np.array([-80.0, -100.0])
    propagation = pd.DataFrame(
        {
            "fold": [0, 1],
            "rsrp_dbm": actual,
            "predicted_free_space_rsrp_dbm": actual + 1,
            "predicted_cost231_hata_rsrp_dbm": actual + 3,
            "predicted_hata_rural_rsrp_dbm": actual + 4,
            "predicted_3gpp_rma_los_rsrp_dbm": actual + 2,
            "predicted_3gpp_rma_nlos_rsrp_dbm": actual + 5,
            "predicted_3gpp_uma_los_rsrp_dbm": actual + 2,
            "predicted_3gpp_uma_nlos_rsrp_dbm": actual + 6,
            "predicted_fitted_log_distance_rsrp_dbm": actual + 0.5,
        }
    )
    ml = pd.DataFrame(
        {
            "fold": [0, 1],
            "rsrp_dbm": actual,
            "pure_lightgbm_rsrp_dbm": actual + 2,
            "hybrid_lightgbm_rsrp_dbm": actual + 3,
            "hybrid_xgboost_rsrp_dbm": actual + 4,
            "quantile_50_rsrp_dbm": actual + 5,
        }
    )
    features = pd.DataFrame(
        {
            "rsrp_dbm": actual,
            "best_server_cost231_hata_in_domain": [False, False],
            "best_server_hata_rural_in_domain": [False, False],
            "best_server_3gpp_rma_los_in_domain": [True, False],
            "best_server_3gpp_rma_nlos_in_domain": [True, True],
            "best_server_3gpp_uma_los_in_domain": [True, True],
            "best_server_3gpp_uma_nlos_in_domain": [False, True],
        }
    )

    propagation_metrics, ml_metrics = build_benchmark_tables(
        propagation,
        ml,
        features,
    )

    fitted_distance = propagation_metrics[
        propagation_metrics["Model"] == "Fitted log distance"
    ].iloc[0]
    cost231 = propagation_metrics[propagation_metrics["Model"] == "COST-231 Hata"].iloc[0]
    assert fitted_distance["RMSE (dB)"] == pytest.approx(0.5)
    assert cost231["Best-server in domain (%)"] == 0
    assert propagation_metrics.iloc[0]["Model"] == "Fitted log distance"
    assert ml_metrics.iloc[0]["Model"] == "Pure LightGBM"


def test_outage_and_reliability_metrics_have_expected_prevalence():
    actual = np.array([-120.0, -105.0, -90.0, -115.0])
    probability = np.array([0.9, 0.7, 0.2, 0.1])

    result = _outage_report(actual, probability, threshold_dbm=-110)
    bins = _reliability_bins(actual, probability, threshold_dbm=-110)

    assert result["prevalence"] == pytest.approx(0.5)
    assert result["precision_at_0_5"] == pytest.approx(0.5)
    assert result["recall_at_0_5"] == pytest.approx(0.5)
    assert result["f1_at_0_5"] == pytest.approx(0.5)
    assert int(bins["count"].sum()) == 4


def test_evaluate_results_writes_spatial_tables_and_figures(tmp_path):
    features = pd.DataFrame(
        {
            "rsrp_dbm": [-80.0, -100.0, -85.0, -120.0],
            "chainage_m": [10.0, 60.0, 2100.0, 2150.0],
        }
    )
    predictions = pd.DataFrame(
        {
            "rsrp_dbm": features["rsrp_dbm"],
            "fold": [0, 0, 1, 1],
            "physics_baseline_rsrp_dbm": [-82.0, -98.0, -84.0, -118.0],
            "pure_lightgbm_rsrp_dbm": [-81.0, -99.0, -83.0, -119.0],
            "hybrid_lightgbm_rsrp_dbm": [-81.0, -99.0, -83.0, -119.0],
            "hybrid_xgboost_rsrp_dbm": [-81.0, -99.0, -83.0, -119.0],
            "quantile_50_rsrp_dbm": [-81.0, -99.0, -83.0, -119.0],
            "outage_probability_below_primary_threshold": [0.8, 0.2, 0.8, 0.2],
            "outage_probability_below_secondary_threshold": [0.9, 0.3, 0.9, 0.3],
        }
    )
    model_summary = {
        "seed": 42,
        "outage_thresholds_dbm": {"primary": -110.0, "secondary": -100.0},
        "uncertainty": {
            "intervals": {
                "80_percent": {"empirical_coverage": 0.8},
                "90_percent": {"empirical_coverage": 0.9},
            },
            "per_fold_coverage": {},
        },
    }

    summary = evaluate_results(
        features,
        predictions,
        model_summary,
        output_directory=tmp_path,
        bootstrap_replicates=20,
    )

    assert summary["rows"] == 4
    assert len(pd.read_csv(tmp_path / "point_metrics.csv")) == 5
    assert (tmp_path / "chainage_50m_bin_metrics.csv").exists()
    assert (tmp_path / "paired_block_bootstrap.csv").exists()
    assert (tmp_path / "outage_reliability.png").exists()
    assert (tmp_path / "spatial_model_rmse.png").exists()
    assert (tmp_path / "evaluation_report.md").exists()
