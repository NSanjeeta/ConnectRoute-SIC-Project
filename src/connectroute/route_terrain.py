"""Terrain profiles for route samples and their nearest LTE tower records."""

from __future__ import annotations

import math
from contextlib import ExitStack
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS, Transformer
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window
from scipy.ndimage import map_coordinates, uniform_filter

from connectroute.terrain import _profile_features, _sample_dem


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROUTE_DEM_DIR = PROJECT_ROOT / "data" / "processed" / "route_dems"
COPERNICUS_TILE_URL = (
    "https://copernicus-dem-30m.s3.amazonaws.com/"
    "Copernicus_DSM_COG_10_{tile}_DEM/Copernicus_DSM_COG_10_{tile}_DEM.tif"
)
DEM_RESOLUTION_M = 30.0
PROFILE_SPACING_M = 30.0
FREQUENCY_MHZ = 1800.0
TOWER_HEIGHT_M = 30.0
RECEIVER_HEIGHT_M = 1.5
EFFECTIVE_EARTH_RADIUS_FACTOR = 4.0 / 3.0
TPI_RADIUS_M = 500.0
def _tile_component(value: int, positive: str, negative: str, digits: int) -> str:
    hemisphere = positive if value >= 0 else negative
    return f"{hemisphere}{abs(value):0{digits}d}_00"


def _tile_name(latitude_degree: int, longitude_degree: int) -> str:
    return (
        f"{_tile_component(latitude_degree, 'N', 'S', 2)}_"
        f"{_tile_component(longitude_degree, 'E', 'W', 3)}"
    )


def _utm_crs(longitude: np.ndarray, latitude: np.ndarray) -> CRS:
    mean_longitude = float(np.mean(longitude))
    mean_latitude = float(np.mean(latitude))
    zone = min(60, max(1, int((mean_longitude + 180) // 6) + 1))
    return CRS.from_dict(
        {
            "proj": "utm",
            "zone": zone,
            "south": mean_latitude < 0,
        }
    )


def _tile_source(tile_name: str) -> str:
    url = COPERNICUS_TILE_URL.format(tile=tile_name)
    return f"/vsicurl/{url}"


def _cached_dem_is_usable(
    path: Path,
    crs: CRS,
    width: int,
    height: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        with rasterio.open(path) as existing:
            return (
                existing.crs is not None
                and existing.crs.to_string() == crs.to_string()
                and existing.width == width
                and existing.height == height
            )
    except (OSError, rasterio.errors.RasterioError):
        return False


def prepare_route_dem(
    route_coordinates: np.ndarray,
    tower_coordinates: np.ndarray,
    *,
    output_directory: Path = DEFAULT_ROUTE_DEM_DIR,
    margin_m: float = 1500.0,
) -> tuple[Path, CRS]:
    """Read needed remote GLO-30 tiles and cache a local projected route DEM."""
    route_xy = np.asarray(route_coordinates, dtype=float)
    tower_xy = np.asarray(tower_coordinates, dtype=float)
    if route_xy.ndim != 2 or route_xy.shape[1] != 2 or not len(route_xy):
        raise ValueError("route_coordinates must be a non-empty [longitude, latitude] array.")
    if tower_xy.ndim != 2 or tower_xy.shape[1] != 2 or not len(tower_xy):
        raise ValueError("tower_coordinates must be a non-empty [longitude, latitude] array.")
    coordinates = np.vstack((route_xy, tower_xy))
    if not np.isfinite(coordinates).all():
        raise ValueError("Route and tower coordinates must be finite.")
    if (
        (coordinates[:, 0] < -180).any()
        or (coordinates[:, 0] > 180).any()
        or (coordinates[:, 1] < -90).any()
        or (coordinates[:, 1] > 90).any()
    ):
        raise ValueError("Route and tower coordinates must be valid longitude/latitude.")
    if not np.isfinite(margin_m) or margin_m <= 0:
        raise ValueError("margin_m must be positive and finite.")

    latitude = coordinates[:, 1]
    longitude = coordinates[:, 0]
    latitude_margin = margin_m / 110_000.0
    mean_latitude = float(np.mean(latitude))
    longitude_margin = margin_m / max(
        10_000.0,
        110_000.0 * math.cos(math.radians(mean_latitude)),
    )
    min_latitude = max(-90.0, float(latitude.min()) - latitude_margin)
    max_latitude = min(90.0, float(latitude.max()) + latitude_margin)
    min_longitude = max(-180.0, float(longitude.min()) - longitude_margin)
    max_longitude = min(180.0, float(longitude.max()) + longitude_margin)
    latitude_degrees = range(math.floor(min_latitude), math.floor(max_latitude) + 1)
    longitude_degrees = range(math.floor(min_longitude), math.floor(max_longitude) + 1)
    tile_names = [
        _tile_name(lat_degree, lon_degree)
        for lat_degree in latitude_degrees
        for lon_degree in longitude_degrees
    ]
    tiles = [_tile_source(name) for name in tile_names]

    crs = _utm_crs(longitude, latitude)
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    projected_x, projected_y = transformer.transform(longitude, latitude)
    west = math.floor((float(np.min(projected_x)) - margin_m) / DEM_RESOLUTION_M)
    south = math.floor((float(np.min(projected_y)) - margin_m) / DEM_RESOLUTION_M)
    east = math.ceil((float(np.max(projected_x)) + margin_m) / DEM_RESOLUTION_M)
    north = math.ceil((float(np.max(projected_y)) + margin_m) / DEM_RESOLUTION_M)
    width = max(1, east - west)
    height = max(1, north - south)
    transform = from_origin(
        west * DEM_RESOLUTION_M,
        north * DEM_RESOLUTION_M,
        DEM_RESOLUTION_M,
        DEM_RESOLUTION_M,
    )
    tile_key = "_".join(tile_names)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / (
        f"route_dem_{crs.to_authority()[1]}_{tile_key}_"
        f"{west}_{south}_{east}_{north}.tif"
    )
    if _cached_dem_is_usable(output_path, crs, width, height):
        return output_path, crs
    output_path.unlink(missing_ok=True)

    nodata = -9999.0
    profile: dict[str, Any] = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
    }
    destination_data = np.full((height, width), nodata, dtype=np.float32)
    try:
        with rasterio.Env(
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            GDAL_HTTP_MULTIRANGE="YES",
        ):
            with ExitStack() as stack:
                sources = [stack.enter_context(rasterio.open(tile)) for tile in tiles]
                for source in sources:
                    reproject(
                        source=rasterio.band(source, 1),
                        destination=destination_data,
                        src_transform=source.transform,
                        src_crs=source.crs,
                        src_nodata=source.nodata,
                        dst_transform=transform,
                        dst_crs=crs,
                        dst_nodata=nodata,
                        resampling=Resampling.bilinear,
                        init_dest_nodata=False,
                    )
    except rasterio.errors.RasterioError as error:
        raise ValueError(
            "Could not read the Copernicus DEM tile data. Check internet access "
            "and retry; the projected terrain DEM was not saved."
        ) from error
    temporary_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.",
        suffix=".tif",
        dir=output_directory,
    )
    os.close(temporary_descriptor)
    temporary_path = Path(temporary_name)
    try:
        with rasterio.open(temporary_path, "w", **profile) as destination:
            destination.write(destination_data, 1)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path, crs


def _terrain_position_index(
    dem: rasterio.DatasetReader,
    easting_m: np.ndarray,
    northing_m: np.ndarray,
    radius_m: float = TPI_RADIUS_M,
) -> np.ndarray:
    """Return elevation minus the local mean; negative values indicate low ground."""
    if not np.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("radius_m must be positive and finite.")
    resolution = float(np.mean(np.abs(dem.res)))
    window_size = 2 * max(1, int(round(radius_m / resolution))) + 1
    point_elevation = _sample_dem(dem, easting_m, northing_m)
    inverse = ~dem.transform
    columns = inverse.a * easting_m + inverse.b * northing_m + inverse.c
    rows = inverse.d * easting_m + inverse.e * northing_m + inverse.f
    center_rows = np.asarray(rows, dtype=float).ravel() - 0.5
    center_columns = np.asarray(columns, dtype=float).ravel() - 0.5
    local_elevation = np.empty(len(center_rows), dtype=float)
    block_height, block_width = dem.block_shapes[0]
    top_left_rows = np.floor(center_rows).astype(np.int64)
    top_left_columns = np.floor(center_columns).astype(np.int64)
    block_rows = np.clip(top_left_rows, 0, dem.height - 1) // block_height
    block_columns = np.clip(top_left_columns, 0, dem.width - 1) // block_width
    block_keys = np.column_stack((block_rows, block_columns))
    halo = window_size // 2 + 1
    for block_row, block_column in np.unique(block_keys, axis=0):
        indexes = np.flatnonzero(
            (block_rows == block_row) & (block_columns == block_column)
        )
        block_row_start = int(block_row) * block_height
        block_column_start = int(block_column) * block_width
        row_start = max(0, block_row_start - halo)
        column_start = max(0, block_column_start - halo)
        row_end = min(dem.height, block_row_start + block_height + halo)
        column_end = min(dem.width, block_column_start + block_width + halo)
        elevation = dem.read(
            1,
            window=Window(
                column_start,
                row_start,
                column_end - column_start,
                row_end - row_start,
            ),
            masked=False,
        ).astype(float)
        valid = np.isfinite(elevation)
        if dem.nodata is not None:
            valid &= ~np.isclose(elevation, dem.nodata, atol=1e-3)
        valid_values = np.where(valid, elevation, 0.0)
        valid_weights = valid.astype(float)
        local_sum = uniform_filter(valid_values, size=window_size, mode="constant")
        local_count = uniform_filter(valid_weights, size=window_size, mode="constant")
        local_mean = local_sum / np.maximum(local_count, 1e-12)
        sample_coordinates = np.vstack(
            (
                center_rows[indexes] - row_start,
                center_columns[indexes] - column_start,
            )
        )
        local_elevation[indexes] = map_coordinates(
            local_mean,
            sample_coordinates,
            order=1,
            mode="nearest",
            prefilter=False,
        )
    return point_elevation - local_elevation.reshape(np.asarray(easting_m).shape)


def add_route_terrain_features(
    profile: pd.DataFrame,
    projected_dem: Path,
    projected_crs: CRS,
) -> pd.DataFrame:
    """Add elevation, Fresnel, roughness, diffraction, and valley-context features."""
    required = {
        "latitude",
        "longitude",
        "nearest_tower_latitude",
        "nearest_tower_longitude",
    }
    missing = required.difference(profile.columns)
    if missing:
        raise ValueError(
            "Route profile is missing columns: " + ", ".join(sorted(missing))
        )
    route = profile.copy()
    transformer = Transformer.from_crs("EPSG:4326", projected_crs, always_xy=True)
    receiver_x, receiver_y = transformer.transform(
        route["longitude"].to_numpy(dtype=float),
        route["latitude"].to_numpy(dtype=float),
    )
    tower_x, tower_y = transformer.transform(
        route["nearest_tower_longitude"].to_numpy(dtype=float),
        route["nearest_tower_latitude"].to_numpy(dtype=float),
    )
    receiver_xy = np.column_stack((receiver_x, receiver_y))
    tower_xy = np.column_stack((tower_x, tower_y))
    with rasterio.open(projected_dem) as dem:
        terrain = _profile_features(
            dem,
            receiver_xy,
            tower_xy,
            frequency_mhz=FREQUENCY_MHZ,
            receiver_height_m=RECEIVER_HEIGHT_M,
            tower_height_m=TOWER_HEIGHT_M,
            profile_spacing_m=PROFILE_SPACING_M,
            effective_earth_radius_factor=EFFECTIVE_EARTH_RADIUS_FACTOR,
        )
        tpi = _terrain_position_index(dem, receiver_xy[:, 0], receiver_xy[:, 1])
    route["receiver_elevation_m"] = terrain["receiver_elevation_m"]
    route["tower_ground_elevation_m"] = terrain["tower_ground_elevation_m"]
    route["terrain_min_fresnel_clearance_fraction"] = terrain[
        "fresnel_clearance_fraction"
    ]
    route["terrain_diffraction_loss_db"] = terrain[
        "knife_edge_diffraction_loss_db"
    ]
    route["terrain_roughness_m"] = terrain["terrain_roughness_m"]
    route["terrain_position_index_500m_m"] = tpi
    route["terrain_context"] = np.select(
        [tpi <= -10.0, tpi >= 10.0],
        ["local low ground (possible valley)", "local high ground"],
        default="near local terrain mean",
    )
    return route
