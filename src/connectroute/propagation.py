"""Path-loss formulas and fold-safe RSRP calibration helpers.

The COST-231 Hata and Hata formulas are empirical. The RMa and UMa formulas
follow 3GPP TR 38.901, clause 7.4.1. Formula values are returned outside their
published applicability domains only so that extrapolation can be audited;
every model also returns an explicit in-domain mask.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from connectroute.geometry import buffered_spatial_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"
SPEED_OF_LIGHT_M_S = 299_792_458.0
REQUIRED_FEATURES = {
    *(f"tower_{rank}_distance_m" for rank in range(1, 4)),
}


def _positive_finite(name: str, values: np.ndarray) -> None:
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError(f"{name} must contain only positive finite values.")


def cost231_hata_path_loss_db(
    distance_m: np.ndarray | float,
    frequency_mhz: float,
    base_station_height_m: np.ndarray | float,
    mobile_height_m: float = 1.5,
    metropolitan: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return COST-231 Hata median path loss and its applicability mask.

    The formal domain is 1500-2000 MHz, 1-20 km, 30-200 m base
    antenna height, and 1-10 m mobile antenna height.
    """
    distance, base_height = np.broadcast_arrays(
        np.asarray(distance_m, dtype=float),
        np.asarray(base_station_height_m, dtype=float),
    )
    _positive_finite("distance_m", distance)
    _positive_finite("frequency_mhz", np.asarray(frequency_mhz, dtype=float))
    _positive_finite("base_station_height_m", base_height)
    _positive_finite("mobile_height_m", np.asarray(mobile_height_m, dtype=float))

    log_frequency = np.log10(frequency_mhz)
    mobile_correction = (
        (1.1 * log_frequency - 0.7) * mobile_height_m
        - (1.56 * log_frequency - 0.8)
    )
    city_correction = 3.0 if metropolitan else 0.0
    path_loss = (
        46.3
        + 33.9 * log_frequency
        - 13.82 * np.log10(base_height)
        - mobile_correction
        + (44.9 - 6.55 * np.log10(base_height)) * np.log10(distance / 1000)
        + city_correction
    )
    in_domain = (
        (frequency_mhz >= 1500)
        & (frequency_mhz <= 2000)
        & (distance >= 1000)
        & (distance <= 20_000)
        & (base_height >= 30)
        & (base_height <= 200)
        & (mobile_height_m >= 1)
        & (mobile_height_m <= 10)
    )
    return np.asarray(path_loss), np.asarray(in_domain, dtype=bool)


def hata_rural_path_loss_db(
    distance_m: np.ndarray | float,
    frequency_mhz: float,
    base_station_height_m: np.ndarray | float,
    mobile_height_m: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return rural Hata open-area path loss and its applicability mask.

    The Hata urban base is corrected with the open-area correction. The formal
    domain is 150-1500 MHz, 1-20 km, 30-200 m base antenna height, and 1-10 m
    mobile antenna height.
    """
    distance, base_height = np.broadcast_arrays(
        np.asarray(distance_m, dtype=float),
        np.asarray(base_station_height_m, dtype=float),
    )
    _positive_finite("distance_m", distance)
    _positive_finite("frequency_mhz", np.asarray(frequency_mhz, dtype=float))
    _positive_finite("base_station_height_m", base_height)
    _positive_finite("mobile_height_m", np.asarray(mobile_height_m, dtype=float))

    log_frequency = np.log10(frequency_mhz)
    mobile_correction = (
        (1.1 * log_frequency - 0.7) * mobile_height_m
        - (1.56 * log_frequency - 0.8)
    )
    urban_path_loss = (
        69.55
        + 26.16 * log_frequency
        - 13.82 * np.log10(base_height)
        - mobile_correction
        + (44.9 - 6.55 * np.log10(base_height)) * np.log10(distance / 1000)
    )
    open_area_correction = (
        -4.78 * log_frequency**2 + 18.33 * log_frequency - 40.94
    )
    path_loss = urban_path_loss + open_area_correction
    in_domain = (
        (frequency_mhz >= 150)
        & (frequency_mhz <= 1500)
        & (distance >= 1000)
        & (distance <= 20_000)
        & (base_height >= 30)
        & (base_height <= 200)
        & (mobile_height_m >= 1)
        & (mobile_height_m <= 10)
    )
    return np.asarray(path_loss), np.asarray(in_domain, dtype=bool)


def free_space_path_loss_db(
    distance_m: np.ndarray | float,
    frequency_mhz: float,
) -> np.ndarray:
    """Return free-space path loss in dB using metres and MHz."""
    distance = np.asarray(distance_m, dtype=float)
    _positive_finite("distance_m", distance)
    _positive_finite("frequency_mhz", np.asarray(frequency_mhz, dtype=float))
    return (
        20 * np.log10(distance)
        + 20 * np.log10(frequency_mhz)
        - 27.55
    )


def _three_gpp_breakpoint_m(
    base_station_height_m: np.ndarray,
    mobile_height_m: float,
    frequency_ghz: float,
    model: str,
) -> np.ndarray:
    if model == "uma":
        effective_bs_height = base_station_height_m - 1.0
        effective_ut_height = mobile_height_m - 1.0
        return (
            4
            * effective_bs_height
            * effective_ut_height
            * frequency_ghz
            * 1_000_000_000
            / SPEED_OF_LIGHT_M_S
        )
    if model == "rma":
        return (
            2
            * np.pi
            * base_station_height_m
            * mobile_height_m
            * frequency_ghz
            * 1_000_000_000
            / SPEED_OF_LIGHT_M_S
        )
    raise ValueError(f"Unknown 3GPP scenario: {model}")


def three_gpp_uma_path_loss_db(
    distance_m: np.ndarray | float,
    frequency_ghz: float,
    base_station_height_m: np.ndarray | float = 30.0,
    mobile_height_m: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return 3GPP UMa LOS/NLOS path loss and the scenario-domain mask.

    Equations are from 3GPP TR 38.901, clause 7.4.1, Table 7.4.1-1.
    """
    distance, base_height = np.broadcast_arrays(
        np.asarray(distance_m, dtype=float),
        np.asarray(base_station_height_m, dtype=float),
    )
    _positive_finite("distance_m", distance)
    _positive_finite("frequency_ghz", np.asarray(frequency_ghz, dtype=float))
    _positive_finite("base_station_height_m", base_height)
    _positive_finite("mobile_height_m", np.asarray(mobile_height_m, dtype=float))
    distance_3d = np.hypot(distance, base_height - mobile_height_m)
    breakpoint = _three_gpp_breakpoint_m(
        base_height,
        mobile_height_m,
        frequency_ghz,
        "uma",
    )
    first_slope = 28 + 22 * np.log10(distance_3d) + 20 * np.log10(frequency_ghz)
    second_slope = (
        28
        + 40 * np.log10(distance_3d)
        + 20 * np.log10(frequency_ghz)
        - 9 * np.log10(breakpoint**2 + (base_height - mobile_height_m) ** 2)
    )
    los = np.where(distance <= breakpoint, first_slope, second_slope)
    nlos = np.maximum(
        los,
        13.54
        + 39.08 * np.log10(distance_3d)
        + 20 * np.log10(frequency_ghz)
        - 0.6 * (mobile_height_m - 1.5),
    )
    in_domain = (
        (frequency_ghz >= 0.5)
        & (frequency_ghz <= 100)
        & (distance >= 10)
        & (distance <= 5000)
        & (base_height >= 10)
        & (base_height <= 150)
        & (mobile_height_m >= 1.5)
        & (mobile_height_m <= 22.5)
    )
    return np.asarray(los), np.asarray(nlos), np.asarray(in_domain, dtype=bool)


def three_gpp_rma_path_loss_db(
    distance_m: np.ndarray | float,
    frequency_ghz: float,
    base_station_height_m: np.ndarray | float = 30.0,
    mobile_height_m: float = 1.5,
    average_building_height_m: float = 5.0,
    street_width_m: float = 20.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return 3GPP RMa LOS/NLOS path loss and its scenario-domain mask.

    Equations are from 3GPP TR 38.901, clause 7.4.1, Table 7.4.1-1.
    """
    distance, base_height = np.broadcast_arrays(
        np.asarray(distance_m, dtype=float),
        np.asarray(base_station_height_m, dtype=float),
    )
    _positive_finite("distance_m", distance)
    _positive_finite("frequency_ghz", np.asarray(frequency_ghz, dtype=float))
    _positive_finite("base_station_height_m", base_height)
    _positive_finite("mobile_height_m", np.asarray(mobile_height_m, dtype=float))
    _positive_finite(
        "average_building_height_m",
        np.asarray(average_building_height_m, dtype=float),
    )
    _positive_finite("street_width_m", np.asarray(street_width_m, dtype=float))

    distance_3d = np.hypot(distance, base_height - mobile_height_m)
    breakpoint = _three_gpp_breakpoint_m(
        base_height,
        mobile_height_m,
        frequency_ghz,
        "rma",
    )
    first_slope = (
        20 * np.log10(40 * np.pi * distance_3d * frequency_ghz / 3)
        + min(0.03 * average_building_height_m**1.72, 10)
        * np.log10(distance_3d)
        - min(0.044 * average_building_height_m**1.72, 14.77)
        + 0.002 * np.log10(average_building_height_m) * distance_3d
    )
    breakpoint_3d = np.hypot(breakpoint, base_height - mobile_height_m)
    breakpoint_loss = (
        20 * np.log10(40 * np.pi * breakpoint_3d * frequency_ghz / 3)
        + min(0.03 * average_building_height_m**1.72, 10)
        * np.log10(breakpoint_3d)
        - min(0.044 * average_building_height_m**1.72, 14.77)
        + 0.002 * np.log10(average_building_height_m) * breakpoint_3d
    )
    second_slope = breakpoint_loss + 40 * np.log10(distance_3d / breakpoint_3d)
    los = np.where(distance <= breakpoint, first_slope, second_slope)
    nlos_candidate = (
        161.04
        - 7.1 * np.log10(street_width_m)
        + 7.5 * np.log10(average_building_height_m)
        - (
            24.37
            - 3.7 * (average_building_height_m / base_height) ** 2
        )
        * np.log10(base_height)
        + (43.42 - 3.1 * np.log10(base_height))
        * (np.log10(distance_3d) - 3)
        + 20 * np.log10(frequency_ghz)
        - (
            3.2 * np.log10(11.75 * mobile_height_m) ** 2 - 4.97
        )
    )
    nlos = np.maximum(los, nlos_candidate)
    in_domain = (
        (frequency_ghz >= 0.5)
        & (frequency_ghz <= 30)
        & (distance >= 10)
        & (distance <= 10_000)
        & (base_height >= 10)
        & (base_height <= 150)
        & (mobile_height_m >= 1)
        & (mobile_height_m <= 10)
        & (average_building_height_m >= 5)
        & (average_building_height_m <= 50)
        & (street_width_m >= 5)
        & (street_width_m <= 50)
    )
    return np.asarray(los), np.asarray(nlos), np.asarray(in_domain, dtype=bool)


def fit_global_offset(
    measured_rsrp_dbm: np.ndarray,
    path_loss_db: np.ndarray,
    training_indices: np.ndarray | None = None,
) -> float:
    """Fit the least-squares common transmit/system offset on training rows."""
    measured = np.asarray(measured_rsrp_dbm, dtype=float)
    path_loss = np.asarray(path_loss_db, dtype=float)
    if measured.ndim != 1 or path_loss.ndim != 1 or measured.shape != path_loss.shape:
        raise ValueError("Measured RSRP and path loss must be equal-length 1D arrays.")
    if training_indices is None:
        selected = np.arange(len(measured))
    else:
        selected = np.asarray(training_indices, dtype=np.int64)
        if selected.ndim != 1 or len(selected) == 0:
            raise ValueError("training_indices must be a non-empty 1D array.")
        if np.any(selected < 0) or np.any(selected >= len(measured)):
            raise ValueError("training_indices contain positions outside the data.")
    valid = np.isfinite(measured[selected]) & np.isfinite(path_loss[selected])
    if not valid.any():
        raise ValueError("No finite training rows are available to fit the offset.")
    return float(np.mean(measured[selected][valid] + path_loss[selected][valid]))


def predict_rsrp_from_path_loss(
    path_loss_db: np.ndarray,
    global_offset_db: float,
) -> np.ndarray:
    """Convert path loss to RSRP prediction using a fitted global offset."""
    path_loss = np.asarray(path_loss_db, dtype=float)
    if not np.isfinite(global_offset_db):
        raise ValueError("global_offset_db must be finite.")
    return global_offset_db - path_loss


def fit_log_distance_model(
    measured_rsrp_dbm: np.ndarray,
    distance_m: np.ndarray,
    training_indices: np.ndarray | None = None,
) -> dict[str, float]:
    """Fit RSRP = intercept + slope*log10(distance in metres) by least squares."""
    measured = np.asarray(measured_rsrp_dbm, dtype=float)
    distance = np.asarray(distance_m, dtype=float)
    if measured.ndim != 1 or distance.ndim != 1 or measured.shape != distance.shape:
        raise ValueError("Measured RSRP and distance must be equal-length 1D arrays.")
    if np.any(np.isfinite(distance) & (distance <= 0)):
        raise ValueError("Finite distances must be positive.")
    if training_indices is None:
        selected = np.arange(len(measured))
    else:
        selected = np.asarray(training_indices, dtype=np.int64)
        if selected.ndim != 1 or len(selected) == 0:
            raise ValueError("training_indices must be a non-empty 1D array.")
        if np.any(selected < 0) or np.any(selected >= len(measured)):
            raise ValueError("training_indices contain positions outside the data.")
    valid = (
        np.isfinite(measured[selected])
        & np.isfinite(distance[selected])
        & (distance[selected] > 0)
    )
    if valid.sum() < 2:
        raise ValueError("At least two finite training rows are required.")
    x = np.log10(distance[selected][valid])
    y = measured[selected][valid]
    design = np.column_stack((np.ones(len(x)), x))
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    return {
        "intercept_dbm_at_1m": float(intercept),
        "slope_db_per_decade": float(slope),
        "path_loss_exponent": float(-slope / 10),
        "training_rows": int(len(x)),
    }


def predict_log_distance_rsrp(
    distance_m: np.ndarray,
    intercept_dbm_at_1m: float,
    slope_db_per_decade: float,
) -> np.ndarray:
    """Predict RSRP from a fitted log-distance regression."""
    distance = np.asarray(distance_m, dtype=float)
    _positive_finite("distance_m", distance)
    if not np.isfinite(intercept_dbm_at_1m) or not np.isfinite(slope_db_per_decade):
        raise ValueError("Fitted log-distance parameters must be finite.")
    return intercept_dbm_at_1m + slope_db_per_decade * np.log10(distance)


def add_propagation_features(
    terrain_features: pd.DataFrame,
    frequency_mhz: float = 1800.0,
    base_station_height_m: float = 30.0,
    mobile_height_m: float = 1.5,
    metropolitan: bool = False,
    average_building_height_m: float = 5.0,
    street_width_m: float = 20.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Calculate per-tower path loss, best-server features, and domain flags."""
    missing = REQUIRED_FEATURES.difference(terrain_features.columns)
    if missing:
        raise ValueError(
            "Terrain features are missing columns: " + ", ".join(sorted(missing))
        )
    _positive_finite("frequency_mhz", np.asarray(frequency_mhz, dtype=float))
    _positive_finite(
        "base_station_height_m",
        np.asarray(base_station_height_m, dtype=float),
    )
    _positive_finite("mobile_height_m", np.asarray(mobile_height_m, dtype=float))
    if not np.isfinite(average_building_height_m) or average_building_height_m < 0:
        raise ValueError("average_building_height_m must be finite and non-negative.")
    if not np.isfinite(street_width_m) or street_width_m < 0:
        raise ValueError("street_width_m must be finite and non-negative.")

    result = terrain_features.reset_index(drop=True).copy()
    model_columns = {
        "free_space": [],
        "cost231_hata": [],
        "hata_rural": [],
        "3gpp_rma_los": [],
        "3gpp_rma_nlos": [],
        "3gpp_uma_los": [],
        "3gpp_uma_nlos": [],
    }
    valid_columns: dict[str, list[str]] = {
        model: [] for model in model_columns if model != "free_space"
    }

    for rank in range(1, 4):
        distance = pd.to_numeric(
            result[f"tower_{rank}_distance_m"],
            errors="raise",
        ).to_numpy(dtype=float)
        _positive_finite(f"tower_{rank}_distance_m", distance)

        fspl = free_space_path_loss_db(distance, frequency_mhz)
        cost231, cost231_valid = cost231_hata_path_loss_db(
            distance,
            frequency_mhz,
            base_station_height_m,
            mobile_height_m,
            metropolitan,
        )
        rural_hata, rural_hata_valid = hata_rural_path_loss_db(
            distance,
            frequency_mhz,
            base_station_height_m,
            mobile_height_m,
        )
        rma_los, rma_nlos, rma_valid = three_gpp_rma_path_loss_db(
            distance,
            frequency_mhz / 1000,
            base_station_height_m,
            mobile_height_m,
            average_building_height_m,
            street_width_m,
        )
        uma_los, uma_nlos, uma_valid = three_gpp_uma_path_loss_db(
            distance,
            frequency_mhz / 1000,
            base_station_height_m,
            mobile_height_m,
        )

        values = {
            "free_space": fspl,
            "cost231_hata": cost231,
            "hata_rural": rural_hata,
            "3gpp_rma_los": rma_los,
            "3gpp_rma_nlos": rma_nlos,
            "3gpp_uma_los": uma_los,
            "3gpp_uma_nlos": uma_nlos,
        }
        domains = {
            "cost231_hata": cost231_valid,
            "hata_rural": rural_hata_valid,
            "3gpp_rma_los": rma_valid,
            "3gpp_rma_nlos": rma_valid,
            "3gpp_uma_los": uma_valid,
            "3gpp_uma_nlos": uma_valid,
        }
        for model, path_loss in values.items():
            column = f"tower_{rank}_{model}_path_loss_db"
            result[column] = path_loss
            model_columns[model].append(column)
            if model in domains:
                valid_column = f"tower_{rank}_{model}_in_domain"
                result[valid_column] = domains[model]
                valid_columns[model].append(valid_column)

    for model, columns in model_columns.items():
        values = result[columns].to_numpy(dtype=float)
        best_position = np.argmin(values, axis=1)
        result[f"best_server_{model}_path_loss_db"] = values[
            np.arange(len(result)),
            best_position,
        ]
        result[f"best_server_{model}_tower_rank"] = best_position + 1
        if model in valid_columns:
            domain_values = result[valid_columns[model]].to_numpy(dtype=bool)
            result[f"best_server_{model}_in_domain"] = domain_values[
                np.arange(len(result)),
                best_position,
            ]

    summary: dict[str, Any] = {
        "rows": int(len(result)),
        "frequency_mhz": float(frequency_mhz),
        "base_station_height_m": float(base_station_height_m),
        "mobile_height_m": float(mobile_height_m),
        "metropolitan_correction": bool(metropolitan),
        "average_building_height_m": float(average_building_height_m),
        "street_width_m": float(street_width_m),
        "nearest_tower_distance_m": {
            "minimum": float(result["tower_1_distance_m"].min()),
            "median": float(result["tower_1_distance_m"].median()),
            "maximum": float(result["tower_1_distance_m"].max()),
        },
        "model_domain_coverage": {},
        "models": sorted(model_columns),
    }
    for model, columns in valid_columns.items():
        coverage = result[columns].to_numpy(dtype=bool)
        summary["model_domain_coverage"][model] = {
            "per_tower_fraction": coverage.mean(axis=0).tolist(),
            "any_candidate_fraction": coverage.any(axis=1).mean().item(),
            "best_server_fraction": result[
                f"best_server_{model}_in_domain"
            ].mean().item(),
        }
    return result, summary


def evaluate_spatial_baselines(
    propagation_features: pd.DataFrame,
    target_column: str = "rsrp_dbm",
    chainage_column: str = "chainage_m",
    fold_column: str = "spatial_fold",
    buffer_m: float = 500.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit calibration on buffered training folds and predict held-out blocks."""
    required = {target_column, chainage_column, fold_column, "tower_1_distance_m"}
    required.update(
        f"best_server_{model}_path_loss_db"
        for model in (
            "free_space",
            "cost231_hata",
            "hata_rural",
            "3gpp_rma_los",
            "3gpp_rma_nlos",
            "3gpp_uma_los",
            "3gpp_uma_nlos",
        )
    )
    missing = required.difference(propagation_features.columns)
    if missing:
        raise ValueError("Propagation features are missing columns: " + ", ".join(sorted(missing)))
    measured = pd.to_numeric(propagation_features[target_column], errors="coerce").to_numpy(
        dtype=float
    )
    chainage = pd.to_numeric(
        propagation_features[chainage_column],
        errors="raise",
    ).to_numpy(dtype=float)
    folds = pd.to_numeric(
        propagation_features[fold_column],
        errors="raise",
    ).to_numpy(dtype=np.int64)
    if not np.isfinite(chainage).all():
        raise ValueError("chainage must contain only finite values.")
    unique_folds = np.unique(folds)
    if len(unique_folds) < 2:
        raise ValueError("At least two spatial folds are required for held-out evaluation.")

    model_names = [
        "free_space",
        "cost231_hata",
        "hata_rural",
        "3gpp_rma_los",
        "3gpp_rma_nlos",
        "3gpp_uma_los",
        "3gpp_uma_nlos",
        "fitted_log_distance",
    ]
    predictions: dict[str, np.ndarray] = {
        model: np.full(len(propagation_features), np.nan, dtype=float)
        for model in model_names
    }
    fold_details: list[dict[str, Any]] = []
    model_parameters: dict[str, list[dict[str, Any]]] = {
        model: [] for model in model_names
    }

    for fold_id in unique_folds:
        train_indices, test_indices = buffered_spatial_split(
            chainage,
            folds,
            int(fold_id),
            buffer_m=buffer_m,
        )
        for model in model_names[:-1]:
            loss_column = f"best_server_{model}_path_loss_db"
            path_loss = pd.to_numeric(
                propagation_features[loss_column],
                errors="raise",
            ).to_numpy(dtype=float)
            offset = fit_global_offset(measured, path_loss, train_indices)
            predictions[model][test_indices] = predict_rsrp_from_path_loss(
                path_loss[test_indices],
                offset,
            )
            model_parameters[model].append(
                {
                    "fold": int(fold_id),
                    "global_offset_db": offset,
                    "training_rows": int(len(train_indices)),
                }
            )

        distance = pd.to_numeric(
            propagation_features["tower_1_distance_m"],
            errors="raise",
        ).to_numpy(dtype=float)
        fitted = fit_log_distance_model(measured, distance, train_indices)
        predictions["fitted_log_distance"][test_indices] = predict_log_distance_rsrp(
            distance[test_indices],
            fitted["intercept_dbm_at_1m"],
            fitted["slope_db_per_decade"],
        )
        model_parameters["fitted_log_distance"].append(
            {"fold": int(fold_id), **fitted}
        )
        fold_details.append(
            {
                "fold": int(fold_id),
                "train_rows": int(len(train_indices)),
                "test_rows": int(len(test_indices)),
            }
        )

    prediction_table = pd.DataFrame(
        {
            "row_index": np.arange(len(propagation_features), dtype=np.int64),
            "fold": folds,
            "rsrp_dbm": measured,
            **{
                f"predicted_{model}_rsrp_dbm": values
                for model, values in predictions.items()
            },
        }
    )
    metrics: dict[str, Any] = {}
    for model, predicted in predictions.items():
        valid = np.isfinite(measured) & np.isfinite(predicted)
        residual = predicted[valid] - measured[valid]
        total_sum_squares = np.sum((measured[valid] - np.mean(measured[valid])) ** 2)
        metrics[model] = {
            "evaluated_rows": int(valid.sum()),
            "rmse_db": float(np.sqrt(np.mean(residual**2))),
            "mae_db": float(np.mean(np.abs(residual))),
            "bias_db": float(np.mean(residual)),
            "r2": (
                float(1 - np.sum(residual**2) / total_sum_squares)
                if total_sum_squares > 0
                else None
            ),
            "fold_metrics": [],
            "fold_parameters": model_parameters[model],
        }
        for fold_id in unique_folds:
            mask = valid & (folds == fold_id)
            fold_residual = predicted[mask] - measured[mask]
            metrics[model]["fold_metrics"].append(
                {
                    "fold": int(fold_id),
                    "evaluated_rows": int(mask.sum()),
                    "rmse_db": float(np.sqrt(np.mean(fold_residual**2))),
                    "mae_db": float(np.mean(np.abs(fold_residual))),
                    "bias_db": float(np.mean(fold_residual)),
                }
            )

    domain_coverage: dict[str, float] = {}
    for model in ("cost231_hata", "hata_rural", "3gpp_rma_nlos", "3gpp_uma_nlos"):
        column = f"best_server_{model}_in_domain"
        if column in propagation_features:
            domain_coverage[model] = float(
                propagation_features.loc[np.isfinite(measured), column].astype(bool).mean()
            )

    summary: dict[str, Any] = {
        "protocol": "5-fold contiguous spatial blocks with chainage buffer",
        "buffer_m": float(buffer_m),
        "target": target_column,
        "evaluated_rows": int(np.isfinite(measured).sum()),
        "folds": fold_details,
        "model_domain_coverage": domain_coverage,
        "metrics": metrics,
        "interpretation": (
            "Path-loss formula values outside a model's published domain are "
            "extrapolations. Global offsets and log-distance parameters are "
            "fitted independently on each buffered training fold."
        ),
    }
    return prediction_table, summary


def _read_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Project config must contain a YAML mapping: {config_path}")
    return config


def process_propagation(
    terrain_features_path: Path,
    output_path: Path,
    summary_path: Path,
    predictions_path: Path | None = None,
    frequency_mhz: float = 1800.0,
    base_station_height_m: float = 30.0,
    mobile_height_m: float = 1.5,
    metropolitan: bool = False,
    average_building_height_m: float = 5.0,
    street_width_m: float = 20.0,
) -> dict[str, Any]:
    """Read terrain features, calculate baseline path losses, and save outputs."""
    if not terrain_features_path.is_file():
        raise FileNotFoundError(f"Terrain feature table not found: {terrain_features_path}")
    terrain_features = pd.read_parquet(terrain_features_path)
    propagation, summary = add_propagation_features(
        terrain_features,
        frequency_mhz=frequency_mhz,
        base_station_height_m=base_station_height_m,
        mobile_height_m=mobile_height_m,
        metropolitan=metropolitan,
        average_building_height_m=average_building_height_m,
        street_width_m=street_width_m,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    propagation.to_parquet(output_path, index=False)
    summary["terrain_features_path"] = str(terrain_features_path)
    summary["output_path"] = str(output_path)
    if predictions_path is not None:
        predictions, evaluation = evaluate_spatial_baselines(propagation)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(predictions_path, index=False)
        summary["predictions_path"] = str(predictions_path)
        summary["spatial_baseline_evaluation"] = evaluation
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    config = _read_config(args.config)
    prediction = config.get("prediction", {})
    summary = process_propagation(
        PROJECT_ROOT / "data" / "processed" / "salzburg_terrain_features.parquet",
        PROJECT_ROOT / "data" / "processed" / "salzburg_propagation_features.parquet",
        PROJECT_ROOT / "results" / "propagation_summary.json",
        PROJECT_ROOT / "data" / "processed" / "spatial_baseline_predictions.parquet",
        frequency_mhz=prediction.get("frequency_mhz", 1800),
        base_station_height_m=prediction.get("tower_antenna_height_m", 30),
        mobile_height_m=prediction.get("receiver_height_m", 1.5),
        metropolitan=prediction.get("cost231_metropolitan_correction", False),
        average_building_height_m=prediction.get("rma_average_building_height_m", 5),
        street_width_m=prediction.get("rma_street_width_m", 20),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
