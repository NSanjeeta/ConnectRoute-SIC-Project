import pandas as pd
import pytest

from connectroute.ingest import prepare_measurements


def test_prepare_measurements_filters_invalid_rows_and_normalizes_fields():
    raw = pd.DataFrame(
        {
            "time": [
                "2018-01-15 16:43:52",
                "2018-01-15 16:43:53",
                "2018-01-15 16:43:54",
                "2018-01-15 16:43:55",
            ],
            "lat": [47.8, 47.9, None, 47.8],
            "long": [13.1, 13.2, 13.3, 13.1],
            "rsrp": [-65.0, None, -70.0, -68.0],
            "technology": ["LTE", "LTE", "LTE", "UMTS"],
            "cell_id": [101, 102, 103, 104],
        }
    )

    cleaned, summary = prepare_measurements(raw)

    assert len(cleaned) == 1
    assert cleaned.loc[0, "latitude"] == 47.8
    assert cleaned.loc[0, "longitude"] == 13.1
    assert cleaned.loc[0, "rsrp_dbm"] == -65.0
    assert cleaned.loc[0, "timestamp"] == pd.Timestamp("2018-01-15 16:43:52")
    assert cleaned.loc[0, "cell_id"] == 101
    assert summary == {
        "input_rows": 4,
        "non_lte_rows": 1,
        "missing_or_invalid_gps_rows": 1,
        "missing_or_invalid_rsrp_rows": 1,
        "exact_duplicate_rows_preserved": 0,
        "output_rows": 1,
    }


def test_prepare_measurements_rejects_missing_required_columns():
    with pytest.raises(ValueError, match="required columns"):
        prepare_measurements(pd.DataFrame({"lat": [47.8]}))


def test_prepare_measurements_rejects_out_of_range_coordinates():
    raw = pd.DataFrame(
        {"lat": [95.0], "long": [13.1], "rsrp": [-65.0], "technology": ["LTE"]}
    )

    with pytest.raises(ValueError, match="outside valid"):
        prepare_measurements(raw)


def test_prepare_measurements_preserves_exact_duplicate_measurements():
    row = {
        "lat": 47.8,
        "long": 13.1,
        "rsrp": -65.0,
        "technology": "LTE",
    }
    cleaned, summary = prepare_measurements(pd.DataFrame([row, row]))

    assert len(cleaned) == 2
    assert summary["exact_duplicate_rows_preserved"] == 1
