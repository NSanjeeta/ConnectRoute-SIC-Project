import numpy as np
import pandas as pd
import pytest

from connectroute.propagation import (
    add_propagation_features,
    cost231_hata_path_loss_db,
    evaluate_spatial_baselines,
    fit_global_offset,
    fit_log_distance_model,
    free_space_path_loss_db,
    hata_rural_path_loss_db,
    predict_log_distance_rsrp,
    predict_rsrp_from_path_loss,
    three_gpp_rma_path_loss_db,
    three_gpp_uma_path_loss_db,
)


def test_free_space_path_loss_known_reference_values():
    assert free_space_path_loss_db(1000, 900) == pytest.approx(91.53, abs=0.02)
    assert free_space_path_loss_db(1000, 1800) == pytest.approx(97.55, abs=0.02)
    assert free_space_path_loss_db(2000, 1800) - free_space_path_loss_db(
        1000, 1800
    ) == pytest.approx(20 * np.log10(2))


def test_cost231_hata_and_domain_mask():
    path_loss, valid = cost231_hata_path_loss_db(
        np.array([1000.0, 500.0]),
        frequency_mhz=1800,
        base_station_height_m=30,
        mobile_height_m=1.5,
    )

    assert path_loss[0] == pytest.approx(136.2, abs=0.1)
    assert valid.tolist() == [True, False]


def test_rural_hata_marks_default_1800mhz_out_of_domain():
    path_loss, valid = hata_rural_path_loss_db(
        np.array([1000.0, 1500.0]),
        frequency_mhz=900,
        base_station_height_m=50,
        mobile_height_m=1.5,
    )

    assert np.isfinite(path_loss).all()
    assert valid.tolist() == [True, True]
    _, default_valid = hata_rural_path_loss_db(1000, 1800, 50)
    assert not bool(default_valid)


def test_3gpp_models_are_finite_and_nlos_is_not_better_than_los():
    distance = np.array([100.0, 500.0, 2000.0])
    for model in (three_gpp_rma_path_loss_db, three_gpp_uma_path_loss_db):
        los, nlos, valid = model(
            distance,
            frequency_ghz=1.8,
            base_station_height_m=30,
            mobile_height_m=1.5,
        )
        assert np.isfinite(los).all()
        assert np.isfinite(nlos).all()
        assert np.all(nlos >= los)
        assert valid.tolist() == [True, True, True]
    rma_los, rma_nlos, _ = three_gpp_rma_path_loss_db(100, 1.8, 30, 1.5)
    uma_los, uma_nlos, _ = three_gpp_uma_path_loss_db(100, 1.8, 30, 1.5)
    assert rma_los == pytest.approx(78.2947, abs=0.01)
    assert rma_nlos == pytest.approx(88.1234, abs=0.01)
    assert uma_los == pytest.approx(77.4785, abs=0.01)
    assert uma_nlos == pytest.approx(97.4682, abs=0.01)


def test_3gpp_uma_marks_short_links_outside_formal_domain():
    _, _, valid = three_gpp_uma_path_loss_db(
        np.array([5.0, 10.0, 5001.0]),
        frequency_ghz=1.8,
        base_station_height_m=30,
        mobile_height_m=1.5,
    )

    assert valid.tolist() == [False, True, False]


def test_fold_safe_global_offset_uses_only_training_rows():
    measured = np.array([-80.0, -100.0, -30.0])
    path_loss = np.array([120.0, 140.0, 5.0])

    offset = fit_global_offset(measured, path_loss, np.array([0, 1]))
    prediction = predict_rsrp_from_path_loss(path_loss, offset)

    assert offset == pytest.approx(40.0)
    assert prediction[:2].tolist() == pytest.approx([-80, -100])
    assert prediction[2] == pytest.approx(35)


def test_log_distance_model_fits_training_rows_only():
    distance = np.array([10.0, 100.0, 1000.0, 10.0])
    measured = np.array([-50.0, -70.0, -90.0, 1000.0])
    fitted = fit_log_distance_model(measured, distance, np.array([0, 1, 2]))

    assert fitted["intercept_dbm_at_1m"] == pytest.approx(-30)
    assert fitted["slope_db_per_decade"] == pytest.approx(-20)
    assert fitted["path_loss_exponent"] == pytest.approx(2)
    assert predict_log_distance_rsrp(
        distance[:3],
        fitted["intercept_dbm_at_1m"],
        fitted["slope_db_per_decade"],
    ).tolist() == pytest.approx([-50, -70, -90])


def test_propagation_features_report_domain_coverage_without_dropping_rows():
    features = pd.DataFrame(
        {
            f"tower_{rank}_distance_m": [100.0, 150.0]
            for rank in range(1, 4)
        }
    )
    output, summary = add_propagation_features(features)

    assert len(output) == len(features)
    assert "best_server_3gpp_uma_nlos_path_loss_db" in output
    assert "best_server_cost231_hata_in_domain" in output
    assert summary["model_domain_coverage"]["cost231_hata"]["any_candidate_fraction"] == 0
    assert summary["model_domain_coverage"]["3gpp_uma_nlos"]["best_server_fraction"] == 1


def test_propagation_rejects_nonpositive_link_distance():
    with pytest.raises(ValueError, match="positive finite"):
        free_space_path_loss_db(0, 1800)


def test_spatial_baselines_fit_on_buffered_train_rows_only():
    chainage = np.arange(0, 10_000, 200, dtype=float)
    blocks = np.floor(chainage / 2000).astype(int)
    folds = blocks // 2
    distance = 50 + chainage / 20
    measured = -40 - 20 * np.log10(distance) + 0.1 * np.sin(chainage)
    data = pd.DataFrame(
        {
            "rsrp_dbm": measured,
            "chainage_m": chainage,
            "spatial_fold": folds,
            "tower_1_distance_m": distance,
        }
    )
    for model in (
        "free_space",
        "cost231_hata",
        "hata_rural",
        "3gpp_rma_los",
        "3gpp_rma_nlos",
        "3gpp_uma_los",
        "3gpp_uma_nlos",
    ):
        data[f"best_server_{model}_path_loss_db"] = 130 + np.log10(distance) * 20
    for model in ("cost231_hata", "hata_rural", "3gpp_rma_nlos", "3gpp_uma_nlos"):
        data[f"best_server_{model}_in_domain"] = folds >= 0

    predictions, summary = evaluate_spatial_baselines(data, buffer_m=500)

    assert len(predictions) == len(data)
    assert predictions.filter(like="predicted_").notna().all().all()
    assert all(
        fold["evaluated_rows"] > 0
        for fold in summary["metrics"]["fitted_log_distance"]["fold_metrics"]
    )
    original_fold_zero = summary["metrics"]["free_space"]["fold_parameters"][0][
        "global_offset_db"
    ]

    changed = data.copy()
    changed.loc[folds == 0, "rsrp_dbm"] += 500
    _, changed_summary = evaluate_spatial_baselines(changed, buffer_m=500)
    changed_fold_zero = changed_summary["metrics"]["free_space"]["fold_parameters"][0][
        "global_offset_db"
    ]
    assert changed_fold_zero == pytest.approx(original_fold_zero)
