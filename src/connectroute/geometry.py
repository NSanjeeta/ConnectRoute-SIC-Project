"""Project Salzburg measurements and assign route chainage and spatial folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pyproj import Transformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"
MEASUREMENT_COLUMNS = {"latitude", "longitude"}


def build_route_centerline(
    easting_m: np.ndarray,
    northing_m: np.ndarray,
    bin_width_m: float = 25.0,
    smoothing_window_bins: int = 5,
) -> pd.DataFrame:
    """Estimate a monotone-easting route centerline using median GPS positions."""
    easting = np.asarray(easting_m, dtype=float)
    northing = np.asarray(northing_m, dtype=float)
    if easting.ndim != 1 or northing.ndim != 1 or easting.size != northing.size:
        raise ValueError("Projected easting and northing must be equal-length 1D arrays.")
    if easting.size < 2 or not np.isfinite(easting).all() or not np.isfinite(northing).all():
        raise ValueError("At least two finite projected measurement coordinates are required.")
    if not np.isfinite(bin_width_m) or bin_width_m <= 0:
        raise ValueError("bin_width_m must be a positive finite number.")
    if smoothing_window_bins < 1 or smoothing_window_bins % 2 == 0:
        raise ValueError("smoothing_window_bins must be a positive odd integer.")
    if easting.max() <= easting.min():
        raise ValueError("Median track easting is not strictly increasing.")

    start_easting = float(easting.min())
    bin_number = np.floor((easting - start_easting) / bin_width_m).astype(np.int64)
    track = (
        pd.DataFrame({"bin": bin_number, "easting_m": easting, "northing_m": northing})
        .groupby("bin", sort=True)
        .median()
    )
    smooth_northing = track["northing_m"].rolling(
        window=smoothing_window_bins,
        center=True,
        min_periods=1,
    ).median()

    center_easting = track["easting_m"].to_numpy(dtype=float, copy=True)
    center_northing = smooth_northing.to_numpy(dtype=float, copy=True)
    if center_easting[0] > start_easting:
        center_easting = np.insert(center_easting, 0, start_easting)
        center_northing = np.insert(center_northing, 0, center_northing[0])
    else:
        center_easting[0] = start_easting
    end_easting = float(easting.max())
    if center_easting[-1] < end_easting:
        center_easting = np.append(center_easting, end_easting)
        center_northing = np.append(center_northing, center_northing[-1])
    else:
        center_easting[-1] = end_easting
    if np.any(np.diff(center_easting) <= 0):
        raise ValueError("Median track easting is not strictly increasing.")

    segment_lengths = np.hypot(np.diff(center_easting), np.diff(center_northing))
    centerline_chainage = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    return pd.DataFrame(
        {
            "easting_m": center_easting,
            "northing_m": center_northing,
            "chainage_m": centerline_chainage,
        }
    )


def project_to_centerline(
    easting_m: np.ndarray,
    northing_m: np.ndarray,
    centerline: pd.DataFrame,
) -> np.ndarray:
    """Project points onto the local centerline segment and return chainage."""
    easting = np.asarray(easting_m, dtype=float)
    northing = np.asarray(northing_m, dtype=float)
    if easting.ndim != 1 or northing.ndim != 1 or easting.size != northing.size:
        raise ValueError("Projected easting and northing must be equal-length 1D arrays.")
    required = {"easting_m", "northing_m", "chainage_m"}
    if not required.issubset(centerline.columns):
        raise ValueError("Centerline must contain easting_m, northing_m, and chainage_m.")
    if len(centerline) < 2:
        raise ValueError("Centerline must contain at least two points.")
    if not np.isfinite(easting).all() or not np.isfinite(northing).all():
        raise ValueError("Projected measurement coordinates must be finite.")

    route_x = centerline["easting_m"].to_numpy(dtype=float)
    route_y = centerline["northing_m"].to_numpy(dtype=float)
    route_chainage = centerline["chainage_m"].to_numpy(dtype=float)
    if np.any(np.diff(route_x) <= 0) or np.any(np.diff(route_chainage) <= 0):
        raise ValueError("Centerline easting and chainage must be strictly increasing.")

    segment = np.searchsorted(route_x, easting, side="right") - 1
    segment = np.clip(segment, 0, len(route_x) - 2)
    dx = route_x[segment + 1] - route_x[segment]
    dy = route_y[segment + 1] - route_y[segment]
    relative_x = easting - route_x[segment]
    relative_y = northing - route_y[segment]
    fraction = np.clip((relative_x * dx + relative_y * dy) / (dx * dx + dy * dy), 0, 1)
    return route_chainage[segment] + fraction * (
        route_chainage[segment + 1] - route_chainage[segment]
    )


def assign_spatial_folds(
    chainage_m: np.ndarray,
    block_length_m: float = 2000.0,
    n_folds: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign contiguous chainage blocks to deterministic contiguous folds."""
    chainage = np.asarray(chainage_m, dtype=float)
    if chainage.ndim != 1 or chainage.size == 0 or not np.isfinite(chainage).all():
        raise ValueError("chainage_m must be a non-empty 1D array of finite values.")
    if np.any(chainage < 0):
        raise ValueError("chainage_m cannot contain negative values.")
    if not np.isfinite(block_length_m) or block_length_m <= 0:
        raise ValueError("block_length_m must be a positive finite number.")
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")

    block_ids = np.floor(chainage / block_length_m).astype(np.int64)
    ordered_blocks = np.unique(block_ids)
    if len(ordered_blocks) < n_folds:
        raise ValueError(
            f"At least {n_folds} spatial blocks are required; found {len(ordered_blocks)}."
        )
    fold_ids = np.empty(len(chainage), dtype=np.int64)
    for fold_id, fold_blocks in enumerate(np.array_split(ordered_blocks, n_folds)):
        fold_ids[np.isin(block_ids, fold_blocks)] = fold_id
    return block_ids, fold_ids


def buffered_spatial_split(
    chainage_m: np.ndarray,
    fold_ids: np.ndarray,
    fold: int,
    buffer_m: float = 500.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return positional train/test indices, excluding nearby training points."""
    chainage = np.asarray(chainage_m, dtype=float)
    folds = np.asarray(fold_ids, dtype=np.int64)
    if chainage.ndim != 1 or folds.ndim != 1 or chainage.size != folds.size:
        raise ValueError("chainage_m and fold_ids must be equal-length 1D arrays.")
    if not np.isfinite(chainage).all():
        raise ValueError("chainage_m must contain only finite values.")
    if not np.isfinite(buffer_m) or buffer_m < 0:
        raise ValueError("buffer_m must be a non-negative finite number.")
    test_mask = folds == fold
    if not test_mask.any():
        raise ValueError(f"Spatial fold {fold} has no test points.")

    test_start = float(chainage[test_mask].min())
    test_end = float(chainage[test_mask].max())
    train_mask = ~test_mask & (
        (chainage < test_start - buffer_m) | (chainage > test_end + buffer_m)
    )
    return np.flatnonzero(train_mask), np.flatnonzero(test_mask)


def _read_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Project config must contain a YAML mapping: {config_path}")
    return config


def process_geometry(
    measurements_path: Path,
    geometry_path: Path,
    centerline_path: Path,
    summary_path: Path,
    projected_crs: str = "EPSG:32633",
    block_length_m: float = 2000.0,
    buffer_m: float = 500.0,
    n_folds: int = 5,
    bin_width_m: float = 25.0,
) -> dict[str, Any]:
    """Write projected measurement geometry, centerline, and fold audit summary."""
    if not measurements_path.is_file():
        raise FileNotFoundError(f"Processed measurements not found: {measurements_path}")
    measurements = pd.read_parquet(measurements_path)
    missing_columns = MEASUREMENT_COLUMNS.difference(measurements.columns)
    if missing_columns:
        raise ValueError(
            "Processed measurements are missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    transformer = Transformer.from_crs("EPSG:4326", projected_crs, always_xy=True)
    easting, northing = transformer.transform(
        measurements["longitude"].to_numpy(dtype=float),
        measurements["latitude"].to_numpy(dtype=float),
    )
    centerline = build_route_centerline(
        easting,
        northing,
        bin_width_m=bin_width_m,
    )
    chainage = project_to_centerline(easting, northing, centerline)
    block_ids, fold_ids = assign_spatial_folds(
        chainage,
        block_length_m=block_length_m,
        n_folds=n_folds,
    )

    geometry = measurements.copy()
    geometry["easting_m"] = easting
    geometry["northing_m"] = northing
    geometry["chainage_m"] = chainage
    geometry["spatial_block"] = block_ids
    geometry["spatial_fold"] = fold_ids

    fold_summary = []
    for fold_id in range(n_folds):
        train_idx, test_idx = buffered_spatial_split(
            chainage,
            fold_ids,
            fold=fold_id,
            buffer_m=buffer_m,
        )
        test_chainage = chainage[test_idx]
        fold_summary.append(
            {
                "fold": fold_id,
                "test_rows": int(len(test_idx)),
                "train_rows_after_buffer": int(len(train_idx)),
                "test_start_m": float(test_chainage.min()),
                "test_end_m": float(test_chainage.max()),
            }
        )

    geometry_path.parent.mkdir(parents=True, exist_ok=True)
    centerline_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    geometry.to_parquet(geometry_path, index=False)
    centerline.to_parquet(centerline_path, index=False)
    summary: dict[str, Any] = {
        "measurements_path": str(measurements_path),
        "geometry_path": str(geometry_path),
        "centerline_path": str(centerline_path),
        "projected_crs": projected_crs,
        "input_rows": int(len(measurements)),
        "centerline_bin_width_m": bin_width_m,
        "centerline_vertices": int(len(centerline)),
        "route_length_m": float(centerline["chainage_m"].iloc[-1]),
        "block_length_m": block_length_m,
        "buffer_m": buffer_m,
        "n_folds": n_folds,
        "n_spatial_blocks": int(geometry["spatial_block"].nunique()),
        "folds": fold_summary,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    config = _read_config(args.config)
    validation = config.get("validation", {})
    data = config.get("data", {})
    summary = process_geometry(
        PROJECT_ROOT / data.get("processed_measurements", "data/processed/salzburg_measurements.parquet"),
        PROJECT_ROOT / "data" / "processed" / "salzburg_geometry.parquet",
        PROJECT_ROOT / "data" / "processed" / "salzburg_route_centerline.parquet",
        PROJECT_ROOT / "results" / "route_geometry_summary.json",
        projected_crs=validation.get("projected_crs", "EPSG:32633"),
        block_length_m=validation.get("block_length_m", 2000),
        buffer_m=validation.get("buffer_m", 500),
        n_folds=validation.get("folds", 5),
        bin_width_m=validation.get("centerline_bin_width_m", 25),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
