import numpy as np
import pandas as pd
import pytest

from connectroute.geometry import (
    assign_spatial_folds,
    build_route_centerline,
    buffered_spatial_split,
    project_to_centerline,
)


def test_centerline_and_projection_follow_median_route():
    easting = np.array([0, 5, 10, 15, 20, 25, 30], dtype=float)
    northing = np.array([0, 1, 2, 3, 4, 5, 6], dtype=float)
    centerline = build_route_centerline(
        easting,
        northing,
        bin_width_m=10,
        smoothing_window_bins=1,
    )

    chainage = project_to_centerline(
        np.array([0, 10, 20, 30], dtype=float),
        np.array([0, 2, 4, 6], dtype=float),
        centerline,
    )

    assert np.all(np.diff(centerline["easting_m"]) > 0)
    assert np.all(np.diff(chainage) > 0)
    assert chainage[0] == pytest.approx(0)
    assert chainage[-1] == pytest.approx(centerline["chainage_m"].iloc[-1])


def test_centerline_rejects_non_increasing_median_track():
    with pytest.raises(ValueError, match="strictly increasing"):
        build_route_centerline(
            np.array([0, 0, 0], dtype=float),
            np.array([0, 1, 2], dtype=float),
            bin_width_m=25,
        )


def test_spatial_folds_are_contiguous_and_buffered():
    chainage = np.arange(0, 10_000, 100, dtype=float)
    blocks, folds = assign_spatial_folds(chainage, block_length_m=2000, n_folds=5)

    assert len(np.unique(blocks)) == 5
    for fold in range(5):
        train, test = buffered_spatial_split(chainage, folds, fold, buffer_m=500)
        assert len(test) > 0
        assert not np.intersect1d(train, test).size
        start, end = chainage[test].min(), chainage[test].max()
        assert np.all(
            (chainage[train] < start - 500) | (chainage[train] > end + 500)
        )
        test_blocks = np.unique(blocks[test])
        assert np.array_equal(test_blocks, np.arange(test_blocks.min(), test_blocks.max() + 1))


def test_buffered_spatial_split_rejects_unknown_fold():
    with pytest.raises(ValueError, match="has no test points"):
        buffered_spatial_split(
            np.arange(1000, dtype=float),
            np.zeros(1000, dtype=int),
            fold=1,
        )
