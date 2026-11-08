"""Spatially validated ML, hybrid residual, and calibrated uncertainty models."""

from __future__ import annotations

import argparse
import json
import os
from importlib.metadata import version
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score
from xgboost import XGBRegressor

from connectroute.geometry import buffered_spatial_split
from connectroute.propagation import (
    fit_log_distance_model,
    predict_log_distance_rsrp,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"
DEFAULT_FEATURES = PROJECT_ROOT / "data" / "processed" / "salzburg_propagation_features.parquet"
DEFAULT_PREDICTIONS = PROJECT_ROOT / "data" / "processed" / "spatial_model_predictions.parquet"
DEFAULT_SUMMARY = PROJECT_ROOT / "results" / "model_summary.json"
PHYSICS_BASELINE = "fitted_log_distance"
QUANTILES = (0.05, 0.10, 0.50, 0.90, 0.95)
FORBIDDEN_FEATURE_TOKENS = (
    "rsrp",
    "rsrq",
    "sinr",
    "signal",
    "rssi",
    "datarate",
    "cell_id",
    "latitude",
    "longitude",
    "easting",
    "northing",
    "chainage",
    "spatial_fold",
    "spatial_block",
    "timestamp",
    "trip",
    "tower_1_created_age",
    "tower_2_created_age",
    "tower_3_created_age",
)


def build_model_features(
    propagation_features: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Create an explicit leakage-resistant feature matrix and feature groups."""
    required = {
        "tower_1_distance_m",
        "tower_2_distance_m",
        "tower_3_distance_m",
        "tower_1_bearing_deg",
        "tower_2_bearing_deg",
        "tower_3_bearing_deg",
        "tower_1_samples",
        "tower_2_samples",
        "tower_3_samples",
        "tower_1_range",
        "tower_2_range",
        "tower_3_range",
        "tower_1_receiver_elevation_m",
    }
    for rank in range(1, 4):
        required.update(
            {
                f"tower_{rank}_fresnel_clearance_fraction",
                f"tower_{rank}_knife_edge_diffraction_loss_db",
                f"tower_{rank}_terrain_roughness_m",
                f"tower_{rank}_tower_ground_elevation_m",
            }
        )
    missing = required.difference(propagation_features.columns)
    if missing:
        raise ValueError(
            "Propagation features are missing required model inputs: "
            + ", ".join(sorted(missing))
        )

    features = pd.DataFrame(index=propagation_features.index)
    pure_columns: list[str] = []
    for rank in range(1, 4):
        distance = pd.to_numeric(
            propagation_features[f"tower_{rank}_distance_m"],
            errors="raise",
        ).to_numpy(dtype=float)
        bearing = np.deg2rad(
            pd.to_numeric(
                propagation_features[f"tower_{rank}_bearing_deg"],
                errors="raise",
            ).to_numpy(dtype=float)
        )
        if not np.isfinite(distance).all() or np.any(distance <= 0):
            raise ValueError(f"tower_{rank}_distance_m must be positive and finite.")
        if not np.isfinite(bearing).all():
            raise ValueError(f"tower_{rank}_bearing_deg must be finite.")

        rank_columns = {
            f"tower_{rank}_distance_m": distance,
            f"tower_{rank}_distance_log10": np.log10(distance),
            f"tower_{rank}_bearing_sin": np.sin(bearing),
            f"tower_{rank}_bearing_cos": np.cos(bearing),
        }
        for source, output in (
            (f"tower_{rank}_samples", f"tower_{rank}_samples"),
            (f"tower_{rank}_range", f"tower_{rank}_range"),
            (
                f"tower_{rank}_fresnel_clearance_fraction",
                f"tower_{rank}_fresnel_clearance_fraction",
            ),
            (
                f"tower_{rank}_knife_edge_diffraction_loss_db",
                f"tower_{rank}_knife_edge_diffraction_loss_db",
            ),
            (f"tower_{rank}_terrain_roughness_m", f"tower_{rank}_terrain_roughness_m"),
            (
                f"tower_{rank}_tower_ground_elevation_m",
                f"tower_{rank}_tower_ground_elevation_m",
            ),
        ):
            rank_columns[output] = pd.to_numeric(
                propagation_features[source],
                errors="raise",
            ).to_numpy(dtype=float)

        features = pd.concat(
            [features, pd.DataFrame(rank_columns, index=features.index)],
            axis=1,
        )
        pure_columns.extend(rank_columns)

    bearings = np.column_stack(
        [
            pd.to_numeric(
                propagation_features[f"tower_{rank}_bearing_deg"],
                errors="raise",
            ).to_numpy(dtype=float)
            for rank in range(1, 4)
        ]
    )
    for first, second in ((1, 2), (1, 3), (2, 3)):
        delta = np.abs(bearings[:, first - 1] - bearings[:, second - 1])
        features[f"tower_{first}_{second}_bearing_separation_deg"] = np.minimum(
            delta,
            360 - delta,
        )
        pure_columns.append(f"tower_{first}_{second}_bearing_separation_deg")

    features["receiver_elevation_m"] = pd.to_numeric(
        propagation_features["tower_1_receiver_elevation_m"],
        errors="raise",
    ).to_numpy(dtype=float)
    pure_columns.append("receiver_elevation_m")
    for rank in range(1, 4):
        elevation_column = f"tower_{rank}_tower_ground_elevation_m"
        output_column = f"tower_{rank}_elevation_difference_m"
        features[output_column] = (
            features[elevation_column] - features["receiver_elevation_m"]
        )
        pure_columns.append(output_column)

    physics_columns = [
        column
        for column in propagation_features.columns
        if (
            column.endswith("_path_loss_db")
            or column.endswith("_in_domain")
        )
        and (column.startswith("tower_") or column.startswith("best_server_"))
    ]
    for column in physics_columns:
        features[column] = pd.to_numeric(
            propagation_features[column],
            errors="raise",
        ).to_numpy(dtype=float)

    hybrid_columns = [*pure_columns, *physics_columns]
    for column in hybrid_columns:
        if any(token in column.lower() for token in FORBIDDEN_FEATURE_TOKENS):
            raise ValueError(f"Leakage-prone feature was selected: {column}")
        values = features[column].to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError(f"Model feature {column} contains infinite values.")
    if not pure_columns or not physics_columns:
        raise ValueError("Both non-physics and propagation physics features are required.")
    return features, pure_columns, hybrid_columns


def _read_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Project config must contain a YAML mapping: {config_path}")
    return config


def _make_model(
    model_kind: str,
    params: dict[str, Any],
    seed: int,
    n_estimators: int,
    n_jobs: int,
    quantile: float | None = None,
) -> Any:
    if model_kind == "lightgbm":
        objective = "quantile" if quantile is not None else "regression"
        model_params: dict[str, Any] = {
            "objective": objective,
            "n_estimators": n_estimators,
            "random_state": seed,
            "n_jobs": n_jobs,
            "verbosity": -1,
            "deterministic": True,
            "force_col_wise": True,
        }
        if quantile is not None:
            model_params["alpha"] = quantile
        model_params.update(params)
        return lgb.LGBMRegressor(**model_params)
    if model_kind == "xgboost":
        model_params = {
            "objective": "reg:squarederror",
            "n_estimators": n_estimators,
            "random_state": seed,
            "n_jobs": n_jobs,
            "tree_method": "hist",
            "verbosity": 0,
        }
        model_params.update(params)
        return XGBRegressor(**model_params)
    raise ValueError(f"Unknown model kind: {model_kind}")


def _sample_params(trial: optuna.Trial, model_kind: str) -> dict[str, Any]:
    if model_kind == "lightgbm":
        return {
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        }
    if model_kind == "xgboost":
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        }
    raise ValueError(f"Unknown model kind: {model_kind}")


def _inner_spatial_splits(
    chainage: np.ndarray,
    folds: np.ndarray,
    allowed_training_indices: np.ndarray,
    buffer_m: float,
    maximum_folds: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    available_folds = np.unique(folds[allowed_training_indices])
    if len(available_folds) < 2:
        raise ValueError("At least two spatial groups are required for inner tuning.")
    if maximum_folds < 2:
        raise ValueError("inner_folds must be at least two.")
    if len(available_folds) > maximum_folds:
        positions = np.linspace(
            0,
            len(available_folds) - 1,
            num=maximum_folds,
            dtype=int,
        )
        selected_folds = available_folds[positions]
    else:
        selected_folds = available_folds

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    allowed_mask = np.zeros(len(chainage), dtype=bool)
    allowed_mask[allowed_training_indices] = True
    for validation_fold in selected_folds:
        validation_indices = allowed_training_indices[
            folds[allowed_training_indices] == validation_fold
        ]
        if len(validation_indices) == 0:
            continue
        start = float(chainage[validation_indices].min())
        end = float(chainage[validation_indices].max())
        candidate_mask = allowed_mask & (folds != validation_fold)
        candidate_mask &= (chainage < start - buffer_m) | (chainage > end + buffer_m)
        inner_training_indices = np.flatnonzero(candidate_mask)
        if len(inner_training_indices) and len(validation_indices):
            splits.append((inner_training_indices, validation_indices))
    if len(splits) < 2:
        raise ValueError(
            "Buffered inner spatial CV produced fewer than two usable splits."
        )
    return splits


def _calibration_training_split(
    chainage: np.ndarray,
    folds: np.ndarray,
    test_fold: int,
    calibration_fold: int,
    buffer_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    test_indices = np.flatnonzero(folds == test_fold)
    calibration_indices = np.flatnonzero(folds == calibration_fold)
    if len(test_indices) == 0 or len(calibration_indices) == 0:
        raise ValueError("Test and calibration folds must both contain observations.")

    test_start, test_end = float(chainage[test_indices].min()), float(
        chainage[test_indices].max()
    )
    calibration_start, calibration_end = float(chainage[calibration_indices].min()), float(
        chainage[calibration_indices].max()
    )
    training_mask = (folds != test_fold) & (folds != calibration_fold)
    training_mask &= (chainage < test_start - buffer_m) | (
        chainage > test_end + buffer_m
    )
    training_mask &= (chainage < calibration_start - buffer_m) | (
        chainage > calibration_end + buffer_m
    )
    training_indices = np.flatnonzero(training_mask)
    return training_indices, calibration_indices, test_indices


def _fold_physics_baseline(
    target: np.ndarray,
    distance: np.ndarray,
    training_indices: np.ndarray,
    prediction_indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    fitted = fit_log_distance_model(target, distance, training_indices)
    prediction = predict_log_distance_rsrp(
        distance[prediction_indices],
        fitted["intercept_dbm_at_1m"],
        fitted["slope_db_per_decade"],
    )
    return prediction, fitted


def _pinball_loss(actual: np.ndarray, predicted: np.ndarray, quantile: float) -> float:
    residual = actual - predicted
    return float(np.mean(np.maximum(quantile * residual, (quantile - 1) * residual)))


def _tune_model(
    model_kind: str,
    target_kind: str,
    quantile: float | None,
    features: np.ndarray,
    target: np.ndarray,
    distance: np.ndarray,
    chainage: np.ndarray,
    folds: np.ndarray,
    training_indices: np.ndarray,
    buffer_m: float,
    inner_folds: int,
    trials: int,
    n_estimators: int,
    n_jobs: int,
    seed: int,
) -> tuple[dict[str, Any], float]:
    splits = _inner_spatial_splits(
        chainage,
        folds,
        training_indices,
        buffer_m,
        inner_folds,
    )
    if trials < 1:
        raise ValueError("tuning_trials must be at least one.")

    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial, model_kind)
        fold_losses: list[float] = []
        for inner_train, inner_valid in splits:
            baseline_train, _ = _fold_physics_baseline(
                target,
                distance,
                inner_train,
                inner_train,
            )
            baseline_valid, _ = _fold_physics_baseline(
                target,
                distance,
                inner_train,
                inner_valid,
            )
            if target_kind == "pure":
                x_train = features[inner_train]
                x_valid = features[inner_valid]
                y_train = target[inner_train]
                y_valid = target[inner_valid]
            else:
                x_train = np.column_stack((features[inner_train], baseline_train))
                x_valid = np.column_stack((features[inner_valid], baseline_valid))
                y_train = target[inner_train] - baseline_train
                y_valid = target[inner_valid] - baseline_valid
            model = _make_model(
                model_kind,
                params,
                seed,
                n_estimators,
                n_jobs,
                quantile=quantile,
            )
            model.fit(x_train, y_train)
            predicted = np.asarray(model.predict(x_valid), dtype=float)
            if target_kind == "hybrid":
                predicted = predicted + baseline_valid
                loss = float(np.sqrt(np.mean((predicted - target[inner_valid]) ** 2)))
            elif target_kind == "quantile_hybrid":
                loss = _pinball_loss(y_valid, predicted, float(quantile))
            else:
                loss = float(np.sqrt(np.mean((predicted - y_valid) ** 2)))
            fold_losses.append(loss)
        return float(np.mean(fold_losses))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=trials, n_jobs=1, show_progress_bar=False)
    return dict(study.best_params), float(study.best_value)


def _regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int | None]:
    valid = np.isfinite(actual) & np.isfinite(predicted)
    if not valid.any():
        return {"evaluated_rows": 0, "rmse_db": None, "mae_db": None, "bias_db": None, "r2": None}
    residual = predicted[valid] - actual[valid]
    total_sum_squares = np.sum((actual[valid] - np.mean(actual[valid])) ** 2)
    return {
        "evaluated_rows": int(valid.sum()),
        "rmse_db": float(np.sqrt(np.mean(residual**2))),
        "mae_db": float(np.mean(np.abs(residual))),
        "bias_db": float(np.mean(residual)),
        "r2": (
            float(1 - np.sum(residual**2) / total_sum_squares)
            if total_sum_squares > 0
            else None
        ),
    }


def _conformal_adjustment(scores: np.ndarray, alpha: float) -> float:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Conformal scores must be a non-empty finite 1D array.")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("Conformal alpha must be between zero and one.")
    rank = int(np.ceil((len(values) + 1) * (1 - alpha)))
    if rank > len(values):
        return float("inf")
    return float(np.sort(values)[rank - 1])


def _quantile_outage_probability(
    quantile_predictions: np.ndarray,
    quantile_levels: np.ndarray,
    threshold_dbm: float,
) -> np.ndarray:
    values = np.maximum.accumulate(
        np.asarray(quantile_predictions, dtype=float),
        axis=1,
    )
    upper_index = np.sum(values <= threshold_dbm, axis=1)
    result = np.empty(len(values), dtype=float)
    below = upper_index == 0
    above = upper_index == values.shape[1]
    result[below] = 0.0
    result[above] = 1.0
    between = ~(below | above)
    rows = np.flatnonzero(between)
    high_indices = upper_index[between]
    low_values = values[rows, high_indices - 1]
    high_values = values[rows, high_indices]
    low_probs = quantile_levels[high_indices - 1]
    high_probs = quantile_levels[high_indices]
    span = high_values - low_values
    fraction = np.divide(
        threshold_dbm - low_values,
        span,
        out=np.ones_like(span),
        where=span > 0,
    )
    result[between] = low_probs + fraction * (high_probs - low_probs)
    return np.clip(result, 0.0, 1.0)


def _average_precision(actual: np.ndarray, probability: np.ndarray) -> float | None:
    if not np.asarray(actual, dtype=bool).any():
        return None
    return float(average_precision_score(actual, probability))


def _outage_metrics(
    actual_rsrp: np.ndarray,
    probability: np.ndarray,
    threshold_dbm: float,
) -> dict[str, float | int | None]:
    valid = np.isfinite(actual_rsrp) & np.isfinite(probability)
    actual_outage = actual_rsrp[valid] < threshold_dbm
    predicted_probability = probability[valid]
    predicted_outage = predicted_probability >= 0.5
    true_positive = int(np.sum(actual_outage & predicted_outage))
    false_positive = int(np.sum(~actual_outage & predicted_outage))
    false_negative = int(np.sum(actual_outage & ~predicted_outage))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "threshold_dbm": float(threshold_dbm),
        "evaluated_rows": int(valid.sum()),
        "observed_outage_rows": int(actual_outage.sum()),
        "observed_outage_fraction": float(actual_outage.mean()) if len(actual_outage) else None,
        "precision_at_0_5": float(precision),
        "recall_at_0_5": float(recall),
        "f1_at_0_5": float(f1),
        "pr_auc_average_precision": _average_precision(actual_outage, predicted_probability),
        "brier_score": (
            float(np.mean((predicted_probability - actual_outage.astype(float)) ** 2))
            if len(actual_outage)
            else None
        ),
    }


def _interval_metrics(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> dict[str, float | int | None]:
    valid = np.isfinite(actual) & np.isfinite(lower) & np.isfinite(upper)
    if not valid.any():
        return {
            "nominal_coverage": float(1 - alpha),
            "evaluated_rows": 0,
            "empirical_coverage": None,
            "mean_width_db": None,
            "mean_interval_score_db": None,
        }
    y = actual[valid]
    low = lower[valid]
    high = upper[valid]
    if np.any(low > high):
        raise ValueError("Prediction interval lower bounds exceed upper bounds.")
    below = np.maximum(low - y, 0)
    above = np.maximum(y - high, 0)
    score = high - low + (2 / alpha) * (below + above)
    return {
        "nominal_coverage": float(1 - alpha),
        "evaluated_rows": int(valid.sum()),
        "empirical_coverage": float(np.mean((y >= low) & (y <= high))),
        "mean_width_db": float(np.mean(high - low)),
        "mean_interval_score_db": float(np.mean(score)),
    }


def evaluate_spatial_models(
    propagation_features: pd.DataFrame,
    *,
    buffer_m: float = 500.0,
    seed: int = 42,
    tuning_trials: int = 50,
    inner_folds: int = 3,
    n_estimators: int = 300,
    n_jobs: int | None = None,
    primary_outage_threshold_dbm: float = -110.0,
    secondary_outage_threshold_dbm: float = -100.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train and evaluate S6/S7 models using outer, inner, and calibration blocks."""
    required = {"rsrp_dbm", "chainage_m", "spatial_fold", "tower_1_distance_m"}
    missing = required.difference(propagation_features.columns)
    if missing:
        raise ValueError("Propagation features are missing columns: " + ", ".join(sorted(missing)))
    if primary_outage_threshold_dbm >= secondary_outage_threshold_dbm:
        raise ValueError("The primary outage threshold must be lower than the secondary threshold.")
    if tuning_trials < 1 or inner_folds < 2 or n_estimators < 1:
        raise ValueError("tuning_trials, inner_folds, and n_estimators must be positive.")
    jobs = n_jobs or max(1, min(8, os.cpu_count() or 1))

    features, pure_columns, hybrid_columns = build_model_features(propagation_features)
    pure_values = features[pure_columns].to_numpy(dtype=np.float32)
    hybrid_static_values = features[hybrid_columns].to_numpy(dtype=np.float32)
    target = pd.to_numeric(
        propagation_features["rsrp_dbm"],
        errors="coerce",
    ).to_numpy(dtype=float)
    chainage = pd.to_numeric(
        propagation_features["chainage_m"],
        errors="raise",
    ).to_numpy(dtype=float)
    folds = pd.to_numeric(
        propagation_features["spatial_fold"],
        errors="raise",
    ).to_numpy(dtype=np.int64)
    distance = pd.to_numeric(
        propagation_features["tower_1_distance_m"],
        errors="raise",
    ).to_numpy(dtype=float)
    if not np.isfinite(chainage).all() or not np.isfinite(distance).all():
        raise ValueError("Chainage and nearest-tower distance must be finite.")
    if not np.isfinite(target).any():
        raise ValueError("At least one finite RSRP target is required.")

    unique_folds = np.unique(folds)
    if len(unique_folds) < 3:
        raise ValueError("At least three spatial folds are required for S6/S7 evaluation.")

    prediction_names = (
        "physics_baseline_rsrp_dbm",
        "pure_lightgbm_rsrp_dbm",
        "hybrid_lightgbm_rsrp_dbm",
        "hybrid_xgboost_rsrp_dbm",
        *(f"quantile_{int(q * 100):02d}_rsrp_dbm" for q in QUANTILES),
        "conformal_80_lower_dbm",
        "conformal_80_upper_dbm",
        "conformal_90_lower_dbm",
        "conformal_90_upper_dbm",
        "outage_probability_below_primary_threshold",
        "outage_probability_below_secondary_threshold",
    )
    predictions = {
        name: np.full(len(propagation_features), np.nan, dtype=float)
        for name in prediction_names
    }
    calibration_fold_by_row = np.full(len(propagation_features), -1, dtype=np.int64)
    fold_summaries: list[dict[str, Any]] = []
    quantile_levels = np.asarray(QUANTILES, dtype=float)
    tuning_records: dict[str, list[dict[str, Any]]] = {
        "pure_lightgbm": [],
        "hybrid_lightgbm": [],
        "hybrid_xgboost": [],
        "hybrid_lightgbm_quantile": [],
    }

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for fold_position, test_fold_value in enumerate(unique_folds):
        test_fold = int(test_fold_value)
        train_indices, test_indices = buffered_spatial_split(
            chainage,
            folds,
            test_fold,
            buffer_m=buffer_m,
        )
        train_indices = train_indices[np.isfinite(target[train_indices])]
        test_indices = test_indices[np.isfinite(target[test_indices])]
        if len(train_indices) < 2 or len(test_indices) == 0:
            raise ValueError(f"Spatial fold {test_fold} has insufficient train or test rows.")

        baseline_train, baseline_parameters = _fold_physics_baseline(
            target,
            distance,
            train_indices,
            train_indices,
        )
        baseline_test, _ = _fold_physics_baseline(
            target,
            distance,
            train_indices,
            test_indices,
        )
        predictions["physics_baseline_rsrp_dbm"][test_indices] = baseline_test
        hybrid_train_features = np.column_stack((hybrid_static_values[train_indices], baseline_train))
        hybrid_test_features = np.column_stack((hybrid_static_values[test_indices], baseline_test))

        fold_model_settings: dict[str, Any] = {}
        pure_params, pure_score = _tune_model(
            "lightgbm",
            "pure",
            None,
            pure_values,
            target,
            distance,
            chainage,
            folds,
            train_indices,
            buffer_m,
            inner_folds,
            tuning_trials,
            n_estimators,
            jobs,
            seed + test_fold,
        )
        pure_model = _make_model(
            "lightgbm",
            pure_params,
            seed + test_fold,
            n_estimators,
            jobs,
        )
        pure_model.fit(pure_values[train_indices], target[train_indices])
        predictions["pure_lightgbm_rsrp_dbm"][test_indices] = pure_model.predict(
            pure_values[test_indices]
        )
        fold_model_settings["pure_lightgbm"] = {
            "best_inner_rmse_db": pure_score,
            "parameters": pure_params,
        }
        tuning_records["pure_lightgbm"].append(
            {"fold": test_fold, "best_inner_rmse_db": pure_score, "parameters": pure_params}
        )

        hybrid_params, hybrid_score = _tune_model(
            "lightgbm",
            "hybrid",
            None,
            hybrid_static_values,
            target,
            distance,
            chainage,
            folds,
            train_indices,
            buffer_m,
            inner_folds,
            tuning_trials,
            n_estimators,
            jobs,
            seed + 100 + test_fold,
        )
        hybrid_model = _make_model(
            "lightgbm",
            hybrid_params,
            seed + 100 + test_fold,
            n_estimators,
            jobs,
        )
        hybrid_model.fit(
            hybrid_train_features,
            target[train_indices] - baseline_train,
        )
        predictions["hybrid_lightgbm_rsrp_dbm"][test_indices] = (
            baseline_test + hybrid_model.predict(hybrid_test_features)
        )
        fold_model_settings["hybrid_lightgbm"] = {
            "best_inner_rmse_db": hybrid_score,
            "parameters": hybrid_params,
        }
        tuning_records["hybrid_lightgbm"].append(
            {"fold": test_fold, "best_inner_rmse_db": hybrid_score, "parameters": hybrid_params}
        )

        xgb_params, xgb_score = _tune_model(
            "xgboost",
            "hybrid",
            None,
            hybrid_static_values,
            target,
            distance,
            chainage,
            folds,
            train_indices,
            buffer_m,
            inner_folds,
            tuning_trials,
            n_estimators,
            jobs,
            seed + 200 + test_fold,
        )
        xgb_model = _make_model(
            "xgboost",
            xgb_params,
            seed + 200 + test_fold,
            n_estimators,
            jobs,
        )
        xgb_model.fit(
            hybrid_train_features,
            target[train_indices] - baseline_train,
        )
        predictions["hybrid_xgboost_rsrp_dbm"][test_indices] = (
            baseline_test + xgb_model.predict(hybrid_test_features)
        )
        fold_model_settings["hybrid_xgboost"] = {
            "best_inner_rmse_db": xgb_score,
            "parameters": xgb_params,
        }
        tuning_records["hybrid_xgboost"].append(
            {"fold": test_fold, "best_inner_rmse_db": xgb_score, "parameters": xgb_params}
        )

        calibration_fold = int(unique_folds[(fold_position + 1) % len(unique_folds)])
        uncertainty_train, calibration_indices, uncertainty_test = _calibration_training_split(
            chainage,
            folds,
            test_fold,
            calibration_fold,
            buffer_m,
        )
        uncertainty_train = uncertainty_train[np.isfinite(target[uncertainty_train])]
        calibration_indices = calibration_indices[np.isfinite(target[calibration_indices])]
        uncertainty_test = uncertainty_test[np.isfinite(target[uncertainty_test])]
        if (
            len(uncertainty_train) < 2
            or len(calibration_indices) == 0
            or len(uncertainty_test) == 0
        ):
            raise ValueError(
                f"Fold {test_fold} has insufficient data for separate conformal calibration."
            )

        quantile_tune_params, quantile_tune_score = _tune_model(
            "lightgbm",
            "quantile_hybrid",
            0.50,
            hybrid_static_values,
            target,
            distance,
            chainage,
            folds,
            uncertainty_train,
            buffer_m,
            inner_folds,
            tuning_trials,
            n_estimators,
            jobs,
            seed + 300 + test_fold,
        )
        quantile_baseline_train, quantile_baseline_parameters = _fold_physics_baseline(
            target,
            distance,
            uncertainty_train,
            uncertainty_train,
        )
        quantile_baseline_calibration, _ = _fold_physics_baseline(
            target,
            distance,
            uncertainty_train,
            calibration_indices,
        )
        quantile_baseline_test, _ = _fold_physics_baseline(
            target,
            distance,
            uncertainty_train,
            uncertainty_test,
        )
        x_quantile_train = np.column_stack(
            (hybrid_static_values[uncertainty_train], quantile_baseline_train)
        )
        x_quantile_calibration = np.column_stack(
            (hybrid_static_values[calibration_indices], quantile_baseline_calibration)
        )
        x_quantile_test = np.column_stack(
            (hybrid_static_values[uncertainty_test], quantile_baseline_test)
        )
        raw_quantile_calibration: dict[float, np.ndarray] = {}
        raw_quantile_test: dict[float, np.ndarray] = {}
        for quantile in QUANTILES:
            quantile_model = _make_model(
                "lightgbm",
                quantile_tune_params,
                seed + 400 + test_fold,
                n_estimators,
                jobs,
                quantile=quantile,
            )
            quantile_model.fit(
                x_quantile_train,
                target[uncertainty_train] - quantile_baseline_train,
            )
            raw_quantile_calibration[quantile] = (
                quantile_baseline_calibration + quantile_model.predict(x_quantile_calibration)
            )
            raw_quantile_test[quantile] = (
                quantile_baseline_test + quantile_model.predict(x_quantile_test)
            )

        calibration_lower_90 = raw_quantile_calibration[0.05]
        calibration_upper_90 = raw_quantile_calibration[0.95]
        calibration_lower_80 = raw_quantile_calibration[0.10]
        calibration_upper_80 = raw_quantile_calibration[0.90]
        correction_90 = _conformal_adjustment(
            np.maximum(
                calibration_lower_90 - target[calibration_indices],
                target[calibration_indices] - calibration_upper_90,
            ),
            alpha=0.10,
        )
        correction_80 = _conformal_adjustment(
            np.maximum(
                calibration_lower_80 - target[calibration_indices],
                target[calibration_indices] - calibration_upper_80,
            ),
            alpha=0.20,
        )

        q05 = raw_quantile_test[0.05] - correction_90
        q10 = raw_quantile_test[0.10] - correction_80
        q50 = raw_quantile_test[0.50]
        q90 = raw_quantile_test[0.90] + correction_80
        q95 = raw_quantile_test[0.95] + correction_90
        q10 = np.minimum(q10, q50)
        q05 = np.minimum(q05, q10)
        q90 = np.maximum(q90, q50)
        q95 = np.maximum(q95, q90)
        calibrated_quantiles = np.column_stack((q05, q10, q50, q90, q95))
        for quantile, values in zip(QUANTILES, calibrated_quantiles.T, strict=True):
            predictions[f"quantile_{int(quantile * 100):02d}_rsrp_dbm"][uncertainty_test] = values
        predictions["conformal_80_lower_dbm"][uncertainty_test] = q10
        predictions["conformal_80_upper_dbm"][uncertainty_test] = q90
        predictions["conformal_90_lower_dbm"][uncertainty_test] = q05
        predictions["conformal_90_upper_dbm"][uncertainty_test] = q95
        predictions["outage_probability_below_primary_threshold"][uncertainty_test] = (
            _quantile_outage_probability(
                calibrated_quantiles,
                quantile_levels,
                primary_outage_threshold_dbm,
            )
        )
        predictions["outage_probability_below_secondary_threshold"][uncertainty_test] = (
            _quantile_outage_probability(
                calibrated_quantiles,
                quantile_levels,
                secondary_outage_threshold_dbm,
            )
        )
        calibration_fold_by_row[uncertainty_test] = calibration_fold
        fold_model_settings["quantile_calibration"] = {
            "calibration_fold": calibration_fold,
            "training_rows": int(len(uncertainty_train)),
            "calibration_rows": int(len(calibration_indices)),
            "test_rows": int(len(uncertainty_test)),
            "inner_pinball_loss_db": quantile_tune_score,
            "quantile_parameters": quantile_tune_params,
            "conformal_80_adjustment_db": correction_80,
            "conformal_90_adjustment_db": correction_90,
            "physics_parameters": quantile_baseline_parameters,
        }
        tuning_records["hybrid_lightgbm_quantile"].append(
            {
                "fold": test_fold,
                "calibration_fold": calibration_fold,
                "best_inner_pinball_loss_db": quantile_tune_score,
                "parameters": quantile_tune_params,
            }
        )
        fold_summaries.append(
            {
                "test_fold": test_fold,
                "train_rows": int(len(train_indices)),
                "test_rows": int(len(test_indices)),
                "physics_parameters": baseline_parameters,
                "models": fold_model_settings,
            }
        )

    prediction_table = pd.DataFrame(
        {
            "row_index": np.arange(len(propagation_features), dtype=np.int64),
            "fold": folds,
            "calibration_fold": calibration_fold_by_row,
            "rsrp_dbm": target,
            **predictions,
        }
    )
    model_columns = {
        "physics_baseline": "physics_baseline_rsrp_dbm",
        "pure_lightgbm": "pure_lightgbm_rsrp_dbm",
        "hybrid_lightgbm": "hybrid_lightgbm_rsrp_dbm",
        "hybrid_xgboost": "hybrid_xgboost_rsrp_dbm",
        "hybrid_lightgbm_quantile_median": "quantile_50_rsrp_dbm",
    }
    model_metrics = {
        name: _regression_metrics(
            prediction_table["rsrp_dbm"].to_numpy(dtype=float),
            prediction_table[column].to_numpy(dtype=float),
        )
        for name, column in model_columns.items()
    }
    fold_metrics: dict[str, list[dict[str, Any]]] = {}
    for name, column in model_columns.items():
        fold_metrics[name] = []
        for fold_value in unique_folds:
            mask = prediction_table["fold"].to_numpy() == fold_value
            fold_metrics[name].append(
                {
                    "fold": int(fold_value),
                    **_regression_metrics(
                        prediction_table.loc[mask, "rsrp_dbm"].to_numpy(dtype=float),
                        prediction_table.loc[mask, column].to_numpy(dtype=float),
                    ),
                }
            )

    uncertainty_metrics: dict[str, Any] = {
        "intervals": {
            "80_percent": _interval_metrics(
                target,
                predictions["conformal_80_lower_dbm"],
                predictions["conformal_80_upper_dbm"],
                alpha=0.20,
            ),
            "90_percent": _interval_metrics(
                target,
                predictions["conformal_90_lower_dbm"],
                predictions["conformal_90_upper_dbm"],
                alpha=0.10,
            ),
        },
        "outage": {},
        "per_fold_coverage": {},
    }
    for threshold_name, threshold, probability_name in (
        (
            "primary",
            primary_outage_threshold_dbm,
            "outage_probability_below_primary_threshold",
        ),
        (
            "secondary",
            secondary_outage_threshold_dbm,
            "outage_probability_below_secondary_threshold",
        ),
    ):
        uncertainty_metrics["outage"][threshold_name] = {
            "overall": _outage_metrics(
                target,
                predictions[probability_name],
                threshold,
            ),
            "per_fold": {
                str(int(fold_value)): _outage_metrics(
                    target[folds == fold_value],
                    predictions[probability_name][folds == fold_value],
                    threshold,
                )
                for fold_value in unique_folds
            },
        }
    for fold_value in unique_folds:
        mask = folds == fold_value
        uncertainty_metrics["per_fold_coverage"][str(int(fold_value))] = {
            "80_percent": _interval_metrics(
                target[mask],
                predictions["conformal_80_lower_dbm"][mask],
                predictions["conformal_80_upper_dbm"][mask],
                alpha=0.20,
            ),
            "90_percent": _interval_metrics(
                target[mask],
                predictions["conformal_90_lower_dbm"][mask],
                predictions["conformal_90_upper_dbm"][mask],
                alpha=0.10,
            ),
        }

    summary: dict[str, Any] = {
        "protocol": (
            "Five outer contiguous spatial folds with 500 m buffers; Optuna inner "
            "spatial CV uses only each outer training set. S7 uses the next contiguous "
            "fold for conformal calibration, excluded from fitting and tuning."
        ),
        "target": "rsrp_dbm",
        "physics_residual_baseline": PHYSICS_BASELINE,
        "seed": int(seed),
        "tuning_trials_per_model_per_outer_fold": int(tuning_trials),
        "inner_spatial_folds": int(inner_folds),
        "n_estimators": int(n_estimators),
        "resolved_n_jobs": int(jobs),
        "buffer_m": float(buffer_m),
        "outage_thresholds_dbm": {
            "primary": float(primary_outage_threshold_dbm),
            "secondary": float(secondary_outage_threshold_dbm),
        },
        "quantile_levels": list(QUANTILES),
        "feature_selection": {
            "pure_ml": pure_columns,
            "hybrid": hybrid_columns + ["fold_physics_baseline_rsrp_dbm"],
            "omitted_data_fields": [
                "signal-derived measurements and RSRP target",
                "raw coordinates, chainage, fold IDs, and trip/timestamp identifiers",
                "tower identifiers and cell IDs",
                "tower record age (snapshot records can post-date measurements)",
                "tower counts within 2 km/5 km (not present in the three-nearest feature table)",
            ],
        },
        "folds": fold_summaries,
        "tuning": tuning_records,
        "metrics": model_metrics,
        "fold_metrics": fold_metrics,
        "uncertainty": uncertainty_metrics,
    }
    return prediction_table, summary


def process_models(
    features_path: Path = DEFAULT_FEATURES,
    predictions_path: Path = DEFAULT_PREDICTIONS,
    summary_path: Path = DEFAULT_SUMMARY,
    config_path: Path = DEFAULT_CONFIG,
    tuning_trials: int | None = None,
) -> dict[str, Any]:
    config = _read_config(config_path)
    validation = config.get("validation", {})
    model_config = config.get("models", {})
    features = pd.read_parquet(features_path)
    predictions, summary = evaluate_spatial_models(
        features,
        buffer_m=float(validation.get("buffer_m", 500.0)),
        seed=int(config.get("project", {}).get("random_seed", 42)),
        tuning_trials=int(
            tuning_trials
            if tuning_trials is not None
            else model_config.get("tuning_trials", 50)
        ),
        inner_folds=int(model_config.get("inner_folds", 3)),
        n_estimators=int(model_config.get("n_estimators", 300)),
        n_jobs=(
            int(model_config["n_jobs"])
            if model_config.get("n_jobs") is not None
            else None
        ),
        primary_outage_threshold_dbm=float(
            model_config.get("primary_outage_threshold_dbm", -110.0)
        ),
        secondary_outage_threshold_dbm=float(
            model_config.get("secondary_outage_threshold_dbm", -100.0)
        ),
    )
    summary["configuration"] = config
    summary["input_feature_table"] = str(features_path)
    summary["library_versions"] = {
        package: version(package)
        for package in ("connectroute", "lightgbm", "optuna", "scikit-learn", "xgboost")
    }
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(predictions_path, index=False)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)
        file.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--trials",
        type=int,
        help="Override the configured Optuna trial count for a quick development run.",
    )
    arguments = parser.parse_args()
    summary = process_models(
        features_path=arguments.features,
        predictions_path=arguments.predictions,
        summary_path=arguments.summary,
        config_path=arguments.config,
        tuning_trials=arguments.trials,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
