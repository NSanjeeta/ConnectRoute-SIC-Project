import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS, Transformer
from scipy.ndimage import map_coordinates
from rasterio.transform import from_origin
from scipy.ndimage import uniform_filter

from connectroute.route_terrain import (
    _cached_dem_is_usable,
    _terrain_position_index,
    _tile_name,
    add_route_terrain_features,
)
from connectroute.terrain import _sample_dem


def test_tile_name_formats_northern_and_southern_coordinates():
    assert _tile_name(12, 76) == "N12_00_E076_00"
    assert _tile_name(-3, -71) == "S03_00_W071_00"


def test_corrupt_cached_dem_is_not_reused(tmp_path):
    dem_path = tmp_path / "corrupt.tif"
    dem_path.write_bytes(b"not a GeoTIFF")

    assert not _cached_dem_is_usable(
        dem_path,
        CRS.from_epsg(32643),
        width=10,
        height=10,
    )


def test_sample_dem_interpolates_across_raster_block_boundaries(tmp_path):
    dem_path = tmp_path / "blocks.tif"
    profile = {
        "driver": "GTiff",
        "height": 300,
        "width": 300,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32643",
        "transform": from_origin(0, 9000, 30, 30),
        "nodata": -9999.0,
        "tiled": True,
    }
    rows, columns = np.mgrid[:300, :300]
    elevation = (2 * rows + 3 * columns).astype(np.float32)
    sample_rows = np.array([10.25, 255.7, 298.2])
    sample_columns = np.array([30.5, 255.7, 298.2])
    eastings = (sample_columns + 0.5) * 30
    northings = 9000 - (sample_rows + 0.5) * 30
    with rasterio.open(dem_path, "w", **profile) as dem:
        dem.write(elevation, 1)

    with rasterio.open(dem_path) as dem:
        actual = _sample_dem(dem, eastings, northings)
    expected = map_coordinates(
        elevation,
        np.vstack((sample_rows, sample_columns)),
        order=1,
        mode="constant",
        cval=profile["nodata"],
        prefilter=False,
    )

    np.testing.assert_allclose(actual, expected)


def test_terrain_position_index_identifies_a_local_depression(tmp_path):
    dem_path = tmp_path / "depression.tif"
    profile = {
        "driver": "GTiff",
        "height": 101,
        "width": 101,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32643",
        "transform": from_origin(0, 3030, 30, 30),
        "nodata": -9999.0,
    }
    elevation = np.full((101, 101), 100.0, dtype=np.float32)
    row, col = np.ogrid[:101, :101]
    elevation[(row - 50) ** 2 + (col - 50) ** 2 <= 5**2] = 80.0
    with rasterio.open(dem_path, "w", **profile) as dem:
        dem.write(elevation, 1)

    with rasterio.open(dem_path) as dem:
        tpi = _terrain_position_index(
            dem,
            np.array([1515.0]),
            np.array([1515.0]),
            radius_m=300,
        )

    assert tpi[0] < -10


def test_terrain_position_index_matches_full_raster_near_block_edges(tmp_path):
    dem_path = tmp_path / "block_tpi.tif"
    profile = {
        "driver": "GTiff",
        "height": 300,
        "width": 300,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32643",
        "transform": from_origin(0, 9000, 30, 30),
        "nodata": -9999.0,
        "tiled": True,
    }
    rows, columns = np.mgrid[:300, :300]
    elevation = (rows * 0.4 + columns * 0.2).astype(np.float32)
    sample_rows = np.array([10.25, 255.7, 298.2])
    sample_columns = np.array([30.5, 255.7, 298.2])
    eastings = (sample_columns + 0.5) * 30
    northings = 9000 - (sample_rows + 0.5) * 30
    with rasterio.open(dem_path, "w", **profile) as dem:
        dem.write(elevation, 1)

    window_size = 7
    valid = np.isfinite(elevation)
    if profile["nodata"] is not None:
        valid &= ~np.isclose(elevation, profile["nodata"], atol=1e-3)
    local_sum = uniform_filter(
        np.where(valid, elevation, 0).astype(float),
        size=window_size,
        mode="constant",
    )
    local_count = uniform_filter(
        valid.astype(float),
        size=window_size,
        mode="constant",
    )
    local_mean = local_sum / np.maximum(local_count, 1e-12)
    expected = map_coordinates(
        elevation,
        np.vstack((sample_rows, sample_columns)),
        order=1,
        mode="constant",
        cval=profile["nodata"],
        prefilter=False,
    ) - map_coordinates(
        local_mean,
        np.vstack((sample_rows, sample_columns)),
        order=1,
        mode="nearest",
        prefilter=False,
    )
    with rasterio.open(dem_path) as dem:
        actual = _terrain_position_index(
            dem,
            eastings,
            northings,
            radius_m=90,
        )

    np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_add_route_terrain_features_reports_flat_profile(tmp_path):
    dem_path = tmp_path / "flat.tif"
    profile = {
        "driver": "GTiff",
        "height": 100,
        "width": 100,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32643",
        "transform": from_origin(499000, 1001500, 30, 30),
        "nodata": -9999.0,
    }
    with rasterio.open(dem_path, "w", **profile) as dem:
        dem.write(np.full((100, 100), 100.0, dtype=np.float32), 1)

    transformer = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True)
    receiver_longitude, receiver_latitude = transformer.transform(500500, 1000000)
    tower_longitude, tower_latitude = transformer.transform(500900, 1000000)
    route = pd.DataFrame(
        [
            {
                "distance_along_route_m": 0.0,
                "latitude": receiver_latitude,
                "longitude": receiver_longitude,
                "nearest_tower_latitude": tower_latitude,
                "nearest_tower_longitude": tower_longitude,
            }
        ]
    )

    result = add_route_terrain_features(route, dem_path, CRS.from_epsg(32643))

    assert result.loc[0, "receiver_elevation_m"] == 100
    assert result.loc[0, "terrain_diffraction_loss_db"] == 0
    assert result.loc[0, "terrain_position_index_500m_m"] == 0
    assert result.loc[0, "terrain_context"] == "near local terrain mean"
