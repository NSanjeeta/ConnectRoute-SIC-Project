"""Build terrain obstruction and diffraction features from a local Copernicus DEM."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import yaml
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from rasterio.windows import Window
from scipy.ndimage import map_coordinates


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"
EARTH_RADIUS_M = 6_371_000.0
SPEED_OF_LIGHT_M_S = 299_792_458.0


def knife_edge_loss_db(v_parameter: np.ndarray | float) -> np.ndarray:
    """Return the ITU single knife-edge diffraction loss for Fresnel v."""
    v = np.asarray(v_parameter, dtype=float)
    inner = np.sqrt((v - 0.1) ** 2 + 1.0) + v - 0.1
    with np.errstate(divide="ignore", invalid="ignore"):
        loss = np.where(v > -0.78, 6.9 + 20 * np.log10(inner), 0.0)
    return np.maximum(loss, 0.0)


def deygout_diffraction_loss_db(
    distances_m: np.ndarray,
    effective_heights_m: np.ndarray,
    sample_counts: np.ndarray,
    frequency_mhz: float,
) -> np.ndarray:
    """Calculate Deygout multi-edge knife-edge loss for batched terrain profiles."""
    distances = np.asarray(distances_m, dtype=float)
    heights = np.asarray(effective_heights_m, dtype=float)
    counts = np.asarray(sample_counts, dtype=np.int64)
    if distances.ndim != 2 or heights.shape != distances.shape:
        raise ValueError("Profile distances and heights must be equal-shaped 2D arrays.")
    if counts.ndim != 1 or len(counts) != len(distances):
        raise ValueError("sample_counts must have one count per profile.")
    if not np.isfinite(distances).all() or not np.isfinite(heights).all():
        raise ValueError("Profile distances and heights must be finite.")
    if np.any(counts < 1) or np.any(counts + 2 > distances.shape[1]):
        raise ValueError("Every profile must contain at least one valid obstruction sample.")
    profile_rows = np.arange(len(distances))
    interior_step_mask = np.arange(distances.shape[1] - 2)[None, :] < counts[:, None]
    interior_steps = np.diff(distances, axis=1)[:, :-1]
    endpoint_steps = distances[profile_rows, -1] - distances[profile_rows, counts]
    if np.any(interior_steps[interior_step_mask] <= 0) or np.any(endpoint_steps <= 0):
        raise ValueError("Valid profile distances must be strictly increasing.")
    if not np.isfinite(frequency_mhz) or frequency_mhz <= 0:
        raise ValueError("frequency_mhz must be a positive finite number.")

    wavelength_m = SPEED_OF_LIGHT_M_S / (frequency_mhz * 1_000_000)
    loss = np.zeros(len(distances), dtype=float)
    candidate_indices = np.arange(1, distances.shape[1] - 1, dtype=np.int64)
    pending: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = [
        (
            np.arange(len(distances), dtype=np.int64),
            np.zeros(len(distances), dtype=np.int64),
            np.full(len(distances), distances.shape[1] - 1, dtype=np.int64),
        )
    ]

    while pending:
        profile_ids, left, right = pending.pop()
        if not len(profile_ids):
            continue
        path_distances = distances[profile_ids]
        path_heights = heights[profile_ids]
        candidates = candidate_indices[None, :]
        segment_length = (
            path_distances[np.arange(len(profile_ids)), right]
            - path_distances[np.arange(len(profile_ids)), left]
        )
        left_height = path_heights[np.arange(len(profile_ids)), left]
        right_height = path_heights[np.arange(len(profile_ids)), right]
        left_distance = path_distances[:, candidate_indices] - path_distances[
            np.arange(len(profile_ids)), left
        ][:, None]
        right_distance = path_distances[
            np.arange(len(profile_ids)), right
        ][:, None] - path_distances[:, candidate_indices]
        valid = (
            (candidates > left[:, None])
            & (candidates < right[:, None])
            & (candidates <= counts[profile_ids, None])
            & (left_distance > 0)
            & (right_distance > 0)
        )
        line_height = left_height[:, None] + (
            (right_height - left_height)[:, None]
            * left_distance
            / segment_length[:, None]
        )
        denominator = np.where(
            valid,
            wavelength_m * left_distance * right_distance,
            1.0,
        )
        v_values = (path_heights[:, candidates[0]] - line_height) * np.sqrt(
            2 * segment_length[:, None] / denominator
        )
        v_values[~valid] = -np.inf
        peak_positions = np.argmax(v_values, axis=1)
        rows = np.arange(len(profile_ids))
        peak_v = v_values[rows, peak_positions]
        active = np.isfinite(peak_v) & (peak_v > -0.78)
        if not active.any():
            continue

        edge_indices = peak_positions + 1
        active_ids = profile_ids[active]
        active_edges = edge_indices[active]
        loss[active_ids] += knife_edge_loss_db(peak_v[active])

        left_children = active_edges - left[active] > 1
        if left_children.any():
            pending.append(
                (
                    active_ids[left_children],
                    left[active][left_children],
                    active_edges[left_children],
                )
            )
        right_children = right[active] - active_edges > 1
        if right_children.any():
            pending.append(
                (
                    active_ids[right_children],
                    active_edges[right_children],
                    right[active][right_children],
                )
            )

    return loss


def _projected_bounds(
    source_dem: Path,
    longitude: np.ndarray,
    latitude: np.ndarray,
    projected_crs: str,
    margin_m: float,
) -> tuple[float, float, float, float]:
    if not np.isfinite(margin_m) or margin_m < 0:
        raise ValueError("margin_m must be a non-negative finite number.")
    with rasterio.open(source_dem) as source:
        if source.crs is None:
            raise ValueError(f"Source DEM has no coordinate reference system: {source_dem}")
        source_to_wgs84 = Transformer.from_crs(source.crs, "EPSG:4326", always_xy=True)
        left, bottom = source_to_wgs84.transform(source.bounds.left, source.bounds.bottom)
        right, top = source_to_wgs84.transform(source.bounds.right, source.bounds.top)
        if (
            np.min(longitude) < min(left, right)
            or np.max(longitude) > max(left, right)
            or np.min(latitude) < min(bottom, top)
            or np.max(latitude) > max(bottom, top)
        ):
            raise ValueError(
                "The downloaded DEM tile does not cover every receiver and matched tower. "
                f"Tile bounds are {(left, bottom, right, top)}."
            )

    transformer = Transformer.from_crs("EPSG:4326", projected_crs, always_xy=True)
    x, y = transformer.transform(longitude, latitude)
    return (
        float(np.min(x) - margin_m),
        float(np.min(y) - margin_m),
        float(np.max(x) + margin_m),
        float(np.max(y) + margin_m),
    )


def reproject_dem(
    source_dem: Path,
    output_dem: Path,
    longitude: np.ndarray,
    latitude: np.ndarray,
    projected_crs: str = "EPSG:32633",
    resolution_m: float = 30.0,
    margin_m: float = 1000.0,
) -> dict[str, Any]:
    """Warp a source DEM into a compact, metre-based local GeoTIFF."""
    if not source_dem.is_file():
        raise FileNotFoundError(f"Source DEM not found: {source_dem}")
    if not np.isfinite(resolution_m) or resolution_m <= 0:
        raise ValueError("resolution_m must be a positive finite number.")
    west, south, east, north = _projected_bounds(
        source_dem,
        np.asarray(longitude, dtype=float),
        np.asarray(latitude, dtype=float),
        projected_crs,
        margin_m,
    )
    west = math.floor(west / resolution_m) * resolution_m
    south = math.floor(south / resolution_m) * resolution_m
    east = math.ceil(east / resolution_m) * resolution_m
    north = math.ceil(north / resolution_m) * resolution_m
    width = max(1, int(round((east - west) / resolution_m)))
    height = max(1, int(round((north - south) / resolution_m)))
    transform = from_origin(west, north, resolution_m, resolution_m)
    output_dem.parent.mkdir(parents=True, exist_ok=True)
    nodata = -9999.0

    with rasterio.open(source_dem) as source:
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "float32",
            "crs": projected_crs,
            "transform": transform,
            "nodata": nodata,
            "compress": "deflate",
            "predictor": 3,
            "tiled": True,
        }
        with rasterio.open(output_dem, "w", **profile) as destination:
            reproject(
                source=rasterio.band(source, 1),
                destination=rasterio.band(destination, 1),
                src_transform=source.transform,
                src_crs=source.crs,
                src_nodata=source.nodata,
                dst_transform=transform,
                dst_crs=projected_crs,
                dst_nodata=nodata,
                resampling=Resampling.bilinear,
            )
    return {
        "source_dem": str(source_dem),
        "projected_dem": str(output_dem),
        "projected_crs": projected_crs,
        "resolution_m": resolution_m,
        "margin_m": margin_m,
        "width_pixels": width,
        "height_pixels": height,
        "nodata": nodata,
    }


def _sample_dem(
    dem: rasterio.DatasetReader,
    easting_m: np.ndarray,
    northing_m: np.ndarray,
) -> np.ndarray:
    inverse = ~dem.transform
    x = np.asarray(easting_m, dtype=float)
    y = np.asarray(northing_m, dtype=float)
    if x.shape != y.shape:
        raise ValueError("Easting and northing arrays must have matching shapes.")
    cols = inverse.a * x.ravel() + inverse.b * y.ravel() + inverse.c - 0.5
    rows = inverse.d * x.ravel() + inverse.e * y.ravel() + inverse.f - 0.5
    elevation = np.empty(len(rows), dtype=float)
    if len(rows) == 0:
        return elevation.reshape(x.shape)

    block_height, block_width = dem.block_shapes[0]
    top_left_rows = np.floor(rows).astype(np.int64)
    top_left_cols = np.floor(cols).astype(np.int64)
    block_rows = np.clip(top_left_rows, 0, dem.height - 1) // block_height
    block_cols = np.clip(top_left_cols, 0, dem.width - 1) // block_width
    block_keys = np.column_stack((block_rows, block_cols))
    nodata = float(dem.nodata)
    for block_row, block_col in np.unique(block_keys, axis=0):
        indexes = np.flatnonzero(
            (block_rows == block_row) & (block_cols == block_col)
        )
        row_start = int(block_row) * block_height
        col_start = int(block_col) * block_width
        window_height = min(block_height + 1, dem.height - row_start)
        window_width = min(block_width + 1, dem.width - col_start)
        data = dem.read(
            1,
            window=Window(col_start, row_start, window_width, window_height),
            masked=False,
        )
        coordinates = np.vstack(
            (rows[indexes] - row_start, cols[indexes] - col_start)
        )
        elevation[indexes] = map_coordinates(
            data,
            coordinates,
            order=1,
            mode="constant",
            cval=nodata,
            prefilter=False,
        )
    elevation = elevation.reshape(x.shape)
    invalid = ~np.isfinite(elevation) | np.isclose(elevation, dem.nodata, atol=1e-3)
    if invalid.any():
        raise ValueError(
            f"DEM sampling encountered {int(invalid.sum())} no-data or out-of-coverage points."
        )
    return elevation


def _profile_features(
    dem: rasterio.DatasetReader,
    receiver_xy: np.ndarray,
    tower_xy: np.ndarray,
    frequency_mhz: float,
    receiver_height_m: float,
    tower_height_m: float,
    profile_spacing_m: float,
    effective_earth_radius_factor: float,
) -> dict[str, np.ndarray]:
    if not np.isfinite(frequency_mhz) or frequency_mhz <= 0:
        raise ValueError("frequency_mhz must be a positive finite number.")
    if not np.isfinite(profile_spacing_m) or profile_spacing_m <= 0:
        raise ValueError("profile_spacing_m must be a positive finite number.")
    if not np.isfinite(effective_earth_radius_factor) or effective_earth_radius_factor <= 0:
        raise ValueError("effective_earth_radius_factor must be a positive finite number.")

    delta = tower_xy - receiver_xy
    total_distance = np.hypot(delta[:, 0], delta[:, 1])
    if not np.isfinite(total_distance).all() or np.any(total_distance <= 0):
        raise ValueError("Every matched tower must have a positive finite separation.")
    sample_counts = np.maximum(1, np.ceil(total_distance / profile_spacing_m).astype(np.int64) - 1)
    max_samples = int(sample_counts.max())
    sample_index = np.arange(1, max_samples + 1, dtype=np.int64)[None, :]
    fractions = sample_index / (sample_counts[:, None] + 1)
    valid_samples = sample_index <= sample_counts[:, None]
    fractions = np.where(valid_samples, fractions, 0.5)

    receiver_ground = _sample_dem(dem, receiver_xy[:, 0], receiver_xy[:, 1])
    tower_ground = _sample_dem(dem, tower_xy[:, 0], tower_xy[:, 1])
    sample_x = receiver_xy[:, 0, None] + delta[:, 0, None] * fractions
    sample_y = receiver_xy[:, 1, None] + delta[:, 1, None] * fractions
    terrain_elevation = np.zeros_like(fractions, dtype=float)
    terrain_elevation[valid_samples] = _sample_dem(
        dem,
        sample_x[valid_samples],
        sample_y[valid_samples],
    )

    distance_to_sample = total_distance[:, None] * fractions
    remaining_distance = total_distance[:, None] - distance_to_sample
    curvature_bulge = (
        distance_to_sample
        * remaining_distance
        / (2 * EARTH_RADIUS_M * effective_earth_radius_factor)
    )
    tx_absolute = tower_ground + tower_height_m
    rx_absolute = receiver_ground + receiver_height_m
    line_height = tx_absolute[:, None] + (
        rx_absolute - tx_absolute
    )[:, None] * fractions
    wavelength_m = SPEED_OF_LIGHT_M_S / (frequency_mhz * 1_000_000)
    fresnel_radius = np.sqrt(
        wavelength_m * distance_to_sample * remaining_distance / total_distance[:, None]
    )
    clearance_fraction = (line_height - terrain_elevation - curvature_bulge) / fresnel_radius
    clearance_fraction[~valid_samples] = np.inf
    clearance_fraction = np.min(clearance_fraction, axis=1)

    masked_elevation = np.where(valid_samples, terrain_elevation, np.nan)
    roughness = np.nanstd(masked_elevation, axis=1)
    interior_distance = np.where(valid_samples, distance_to_sample, total_distance[:, None])
    profile_distances = np.column_stack(
        (np.zeros(len(total_distance)), interior_distance, total_distance)
    )
    terrain_with_curvature = terrain_elevation + curvature_bulge
    profile_heights = np.column_stack(
        (tx_absolute, terrain_with_curvature, rx_absolute)
    )
    diffraction_loss = deygout_diffraction_loss_db(
        profile_distances,
        profile_heights,
        sample_counts,
        frequency_mhz,
    )
    return {
        "receiver_elevation_m": receiver_ground,
        "tower_ground_elevation_m": tower_ground,
        "fresnel_clearance_fraction": clearance_fraction,
        "knife_edge_diffraction_loss_db": diffraction_loss,
        "terrain_roughness_m": roughness,
        "profile_sample_count": sample_counts,
    }


def add_terrain_features(
    features: pd.DataFrame,
    projected_dem: Path,
    projected_crs: str = "EPSG:32633",
    k: int = 3,
    frequency_mhz: float = 1800.0,
    receiver_height_m: float = 1.5,
    tower_height_m: float = 30.0,
    profile_spacing_m: float = 30.0,
    effective_earth_radius_factor: float = 4.0 / 3.0,
    batch_size: int = 10_000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add sampled DEM and terrain obstruction features for each matched tower."""
    required = {"easting_m", "northing_m"}
    for rank in range(1, k + 1):
        required.update({f"tower_{rank}_latitude", f"tower_{rank}_longitude"})
    missing = required.difference(features.columns)
    if missing:
        raise ValueError("Spatial features are missing columns: " + ", ".join(sorted(missing)))
    if not projected_dem.is_file():
        raise FileNotFoundError(f"Projected DEM not found: {projected_dem}")
    if batch_size < 1:
        raise ValueError("batch_size must be greater than zero.")

    result = features.reset_index(drop=True).copy()
    transformer = Transformer.from_crs("EPSG:4326", projected_crs, always_xy=True)
    with rasterio.open(projected_dem) as dem:
        if dem.crs is None or dem.crs.to_string() != projected_crs:
            raise ValueError(
                f"Projected DEM CRS must be {projected_crs}; found {dem.crs}."
            )
        for rank in range(1, k + 1):
            tower_x, tower_y = transformer.transform(
                pd.to_numeric(result[f"tower_{rank}_longitude"], errors="coerce").to_numpy(
                    dtype=float
                ),
                pd.to_numeric(result[f"tower_{rank}_latitude"], errors="coerce").to_numpy(
                    dtype=float
                ),
            )
            tower_xy_all = np.column_stack((tower_x, tower_y))
            receiver_xy_all = result[["easting_m", "northing_m"]].to_numpy(dtype=float)
            column_names = (
                "receiver_elevation_m",
                "tower_ground_elevation_m",
                "fresnel_clearance_fraction",
                "knife_edge_diffraction_loss_db",
                "terrain_roughness_m",
                "profile_sample_count",
            )
            collected = {name: np.empty(len(result), dtype=float) for name in column_names}
            for start in range(0, len(result), batch_size):
                end = min(start + batch_size, len(result))
                batch_features = _profile_features(
                    dem,
                    receiver_xy_all[start:end],
                    tower_xy_all[start:end],
                    frequency_mhz=frequency_mhz,
                    receiver_height_m=receiver_height_m,
                    tower_height_m=tower_height_m,
                    profile_spacing_m=profile_spacing_m,
                    effective_earth_radius_factor=effective_earth_radius_factor,
                )
                for name in column_names:
                    collected[name][start:end] = batch_features[name]
            for name in column_names:
                result[f"tower_{rank}_{name}"] = collected[name]

    summary = {
        "rows": int(len(result)),
        "k": int(k),
        "frequency_mhz": float(frequency_mhz),
        "receiver_height_m": float(receiver_height_m),
        "tower_height_m": float(tower_height_m),
        "profile_spacing_m": float(profile_spacing_m),
        "effective_earth_radius_factor": float(effective_earth_radius_factor),
        "batch_size": int(batch_size),
        "dem_path": str(projected_dem),
        "features": {},
    }
    for rank in range(1, k + 1):
        summary["features"][f"tower_{rank}"] = {
            name: {
                "median": float(result[f"tower_{rank}_{name}"].median()),
                "p95": float(result[f"tower_{rank}_{name}"].quantile(0.95)),
            }
            for name in (
                "fresnel_clearance_fraction",
                "knife_edge_diffraction_loss_db",
                "terrain_roughness_m",
                "profile_sample_count",
            )
        }
    return result, summary


def _read_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Project config must contain a YAML mapping: {config_path}")
    return config


def process_terrain(
    features_path: Path,
    source_dem: Path,
    projected_dem: Path,
    output_path: Path,
    summary_path: Path,
    projected_crs: str = "EPSG:32633",
    k: int = 3,
    frequency_mhz: float = 1800.0,
    receiver_height_m: float = 1.5,
    tower_height_m: float = 30.0,
    profile_spacing_m: float = 30.0,
    effective_earth_radius_factor: float = 4.0 / 3.0,
    dem_resolution_m: float = 30.0,
    dem_margin_m: float = 1000.0,
    batch_size: int = 10_000,
) -> dict[str, Any]:
    """Warp the source DSM, calculate profiles, and write terrain features."""
    if not features_path.is_file():
        raise FileNotFoundError(f"Spatial feature table not found: {features_path}")
    features = pd.read_parquet(features_path)
    tower_longitudes = np.concatenate(
        [features[f"tower_{rank}_longitude"].to_numpy(dtype=float) for rank in range(1, k + 1)]
    )
    tower_latitudes = np.concatenate(
        [features[f"tower_{rank}_latitude"].to_numpy(dtype=float) for rank in range(1, k + 1)]
    )
    all_longitudes = np.concatenate((features["longitude"].to_numpy(dtype=float), tower_longitudes))
    all_latitudes = np.concatenate((features["latitude"].to_numpy(dtype=float), tower_latitudes))
    dem_summary = reproject_dem(
        source_dem,
        projected_dem,
        all_longitudes,
        all_latitudes,
        projected_crs=projected_crs,
        resolution_m=dem_resolution_m,
        margin_m=dem_margin_m,
    )
    terrain, terrain_summary = add_terrain_features(
        features,
        projected_dem,
        projected_crs=projected_crs,
        k=k,
        frequency_mhz=frequency_mhz,
        receiver_height_m=receiver_height_m,
        tower_height_m=tower_height_m,
        profile_spacing_m=profile_spacing_m,
        effective_earth_radius_factor=effective_earth_radius_factor,
        batch_size=batch_size,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    terrain.to_parquet(output_path, index=False)
    summary: dict[str, Any] = {
        "source_product": "Copernicus DEM GLO-30 Public (DSM)",
        "source_license": "Copernicus DEM terms and conditions",
        "features_path": str(features_path),
        "output_path": str(output_path),
        "dem": dem_summary,
        "terrain": terrain_summary,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    config = _read_config(args.config)
    validation = config.get("validation", {})
    prediction = config.get("prediction", {})
    terrain_config = config.get("terrain", {})
    summary = process_terrain(
        PROJECT_ROOT / "data" / "processed" / "salzburg_spatial_features.parquet",
        PROJECT_ROOT
        / "data"
        / "raw"
        / "Copernicus_DSM_COG_10_N47_00_E013_00_DEM.tif",
        PROJECT_ROOT / "data" / "processed" / "salzburg_dem_utm.tif",
        PROJECT_ROOT / "data" / "processed" / "salzburg_terrain_features.parquet",
        PROJECT_ROOT / "results" / "terrain_summary.json",
        projected_crs=validation.get("projected_crs", "EPSG:32633"),
        k=prediction.get("nearest_towers", 3),
        frequency_mhz=prediction.get("frequency_mhz", 1800),
        receiver_height_m=prediction.get("receiver_height_m", 1.5),
        tower_height_m=prediction.get("tower_antenna_height_m", 30),
        profile_spacing_m=terrain_config.get("profile_spacing_m", 30),
        effective_earth_radius_factor=terrain_config.get(
            "effective_earth_radius_factor",
            4.0 / 3.0,
        ),
        dem_resolution_m=terrain_config.get("dem_resolution_m", 30),
        dem_margin_m=terrain_config.get("dem_margin_m", 1000),
        batch_size=terrain_config.get("batch_size", 10_000),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
