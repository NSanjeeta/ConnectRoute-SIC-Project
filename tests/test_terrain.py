import numpy as np
import pytest
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from connectroute.terrain import (
    _profile_features,
    deygout_diffraction_loss_db,
    knife_edge_loss_db,
)


def test_knife_edge_loss_is_zero_below_clearance_threshold():
    loss = knife_edge_loss_db(np.array([-1.0, -0.78, 0.0, 1.0]))

    assert loss[0] == 0
    assert loss[1] == 0
    assert loss[2] > 0
    assert loss[3] > loss[2]


def test_deygout_loss_is_zero_for_a_clear_profile():
    distances = np.array([[0.0, 500.0, 1000.0]])
    heights = np.array([[31.5, 0.0, 31.5]])

    loss = deygout_diffraction_loss_db(
        distances,
        heights,
        np.array([1]),
        frequency_mhz=1800,
    )

    assert loss[0] == 0


def test_deygout_handles_different_profile_lengths_in_a_batch():
    distances = np.array(
        [
            [0.0, 25.0, 50.0, 75.0, 100.0],
            [0.0, 50.0, 100.0, 100.0, 100.0],
        ]
    )
    heights = np.array(
        [
            [30.0, 0.0, 0.0, 0.0, 1.5],
            [30.0, 0.0, 0.0, 0.0, 1.5],
        ]
    )

    loss = deygout_diffraction_loss_db(
        distances,
        heights,
        np.array([3, 1]),
        frequency_mhz=1800,
    )

    assert np.all(loss == 0)


def test_deygout_loss_increases_for_a_higher_obstruction():
    distances = np.array(
        [
            [0.0, 500.0, 1000.0],
            [0.0, 500.0, 1000.0],
        ]
    )
    heights = np.array(
        [
            [31.5, 20.0, 31.5],
            [31.5, 40.0, 31.5],
        ]
    )

    loss = deygout_diffraction_loss_db(
        distances,
        heights,
        np.array([1, 1]),
        frequency_mhz=1800,
    )

    assert loss[0] == 0
    assert loss[1] > 0


def test_deygout_supports_multiple_obstacles_and_rejects_bad_shapes():
    distances = np.array([[0, 200, 400, 600, 800]], dtype=float)
    heights = np.array([[30, 45, 10, 50, 30]], dtype=float)
    loss = deygout_diffraction_loss_db(
        distances,
        heights,
        np.array([3]),
        frequency_mhz=1800,
    )
    assert loss[0] > 0

    with pytest.raises(ValueError, match="equal-shaped"):
        deygout_diffraction_loss_db(
            distances,
            heights[:, :-1],
            np.array([3]),
            frequency_mhz=1800,
        )


def test_flat_dem_profile_has_zero_roughness_and_no_diffraction():
    profile = {
        "driver": "GTiff",
        "height": 20,
        "width": 20,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32633",
        "transform": from_origin(0, 200, 10, 10),
        "nodata": -9999.0,
    }
    with MemoryFile() as memory_file:
        with memory_file.open(**profile) as dem:
            dem.write(np.full((20, 20), 100.0, dtype=np.float32), 1)
            output = _profile_features(
                dem,
                receiver_xy=np.array([[55.0, 95.0]]),
                tower_xy=np.array([[155.0, 95.0]]),
                frequency_mhz=1800,
                receiver_height_m=1.5,
                tower_height_m=30,
                profile_spacing_m=30,
                effective_earth_radius_factor=4 / 3,
            )

    assert output["receiver_elevation_m"][0] == pytest.approx(100.0)
    assert output["tower_ground_elevation_m"][0] == pytest.approx(100.0)
    assert output["terrain_roughness_m"][0] == pytest.approx(0.0)
    assert output["knife_edge_diffraction_loss_db"][0] == pytest.approx(0.0)
    assert output["fresnel_clearance_fraction"][0] > 0
