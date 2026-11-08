"""Compare distance-only and terrain-augmented models on buffered spatial folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio

from connectroute.geometry import buffered_spatial_split
from connectroute.route_terrain import _terrain_position_index


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = PROJECT_ROOT / "data" / "processed" / "salzburg_terrain_features.parquet"
DEFAULT_DEM = PROJECT_ROOT / "data" / "processed" / "salzburg_dem_utm.tif"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "terrain_model_comparison.json"
BUFFER_M = 500.0
TERRAIN_COLUMNS = (
    "tower_1_knife_edge_diffraction_loss_db",
    "tower_1_fresnel_clearance_fraction",
    "tower_1_terrain_roughness_m",
    "tower_1_tower_ground_elevation_m",
    "tower_1_receiver_elevation_m",
)


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = target - prediction
    return {
        "rmse_db": float(np.sqrt(np.mean(residual**2))),
        "mae_db": float(np.mean(np.abs(residual))),
    }


def compare_models(
    features_path: Path,
    dem_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Write held-out metrics for two least-squares candidates and their differences."""
    columns = [
        "rsrp_dbm",
        "tower_1_distance_m",
        "chainage_m",
        "spatial_fold",
        "easting_m",
        "northing_m",
        *TERRAIN_COLUMNS,
    ]
    features = pd.read_parquet(features_path, columns=columns)
    if features.empty:
        raise ValueError(f"No training rows found in {features_path}.")
    values = features.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Terrain model comparison requires finite feature values.")
    distance = features["tower_1_distance_m"].to_numpy(dtype=float)
    if (distance <= 0).any():
        raise ValueError("Tower distances must be positive.")

    target = features["rsrp_dbm"].to_numpy(dtype=float)
    chainage = features["chainage_m"].to_numpy(dtype=float)
    folds = features["spatial_fold"].to_numpy(dtype=np.int64)
    log_distance = np.log10(distance)
    baseline_matrix = np.column_stack((np.ones(len(features)), log_distance))
    terrain_matrix = np.column_stack(
        (
            baseline_matrix,
            features["tower_1_knife_edge_diffraction_loss_db"].to_numpy(dtype=float),
            np.clip(
                features["tower_1_fresnel_clearance_fraction"].to_numpy(dtype=float),
                -10.0,
                20.0,
            ),
            features["tower_1_terrain_roughness_m"].to_numpy(dtype=float),
            (
                features["tower_1_tower_ground_elevation_m"].to_numpy(dtype=float)
                - features["tower_1_receiver_elevation_m"].to_numpy(dtype=float)
            ),
        )
    )
    with rasterio.open(dem_path) as dem:
        terrain_position_index = _terrain_position_index(
            dem,
            features["easting_m"].to_numpy(dtype=float),
            features["northing_m"].to_numpy(dtype=float),
        )
    valley_matrix = np.column_stack((terrain_matrix, terrain_position_index))
    model_matrices = {
        "distance_only": baseline_matrix,
        "distance_plus_terrain": terrain_matrix,
        "distance_terrain_and_valley_position": valley_matrix,
    }
    predictions = {
        name: np.full(len(features), np.nan) for name in model_matrices
    }
    fold_results: list[dict[str, Any]] = []
    for fold in sorted(np.unique(folds)):
        train_indices, test_indices = buffered_spatial_split(
            chainage,
            folds,
            int(fold),
            buffer_m=BUFFER_M,
        )
        fold_metrics: dict[str, Any] = {
            "fold": int(fold),
            "training_rows": int(len(train_indices)),
            "test_rows": int(len(test_indices)),
        }
        for name, matrix in model_matrices.items():
            coefficients = np.linalg.lstsq(
                matrix[train_indices],
                target[train_indices],
                rcond=None,
            )[0]
            predictions[name][test_indices] = matrix[test_indices] @ coefficients
            fold_metrics[name] = _metrics(
                target[test_indices],
                predictions[name][test_indices],
            )
        fold_results.append(fold_metrics)
    if any(not np.isfinite(prediction).all() for prediction in predictions.values()):
        raise ValueError("Some observations did not receive held-out predictions.")

    model_metrics = {
        name: _metrics(target, prediction) for name, prediction in predictions.items()
    }
    summary: dict[str, Any] = {
        "features_path": str(features_path),
        "dem_path": str(dem_path),
        "rows": int(len(features)),
        "buffer_m": BUFFER_M,
        "validation": "five contiguous spatial folds with a 500 m training buffer",
        "models": model_metrics,
        "rmse_change_vs_distance_only_db": {
            name: metrics["rmse_db"] - model_metrics["distance_only"]["rmse_db"]
            for name, metrics in model_metrics.items()
        },
        "terrain_features": [
            *TERRAIN_COLUMNS,
            "terrain_position_index_500m_m",
        ],
        "folds": fold_results,
        "interpretation": (
            "The terrain-augmented candidate is not selected unless held-out "
            "performance improves; results remain specific to the Salzburg corridor."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--dem", type=Path, default=DEFAULT_DEM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(json.dumps(compare_models(args.features, args.dem, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
