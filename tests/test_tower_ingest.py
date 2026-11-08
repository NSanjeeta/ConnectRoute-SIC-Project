import lzma

import pandas as pd

from connectroute.tower_ingest import extract_country_lte_towers


def test_extract_country_lte_towers_filters_and_keeps_source_metadata(tmp_path):
    rows = pd.DataFrame(
        [
            {
                "radio": "LTE",
                "mcc": 232,
                "net": 1,
                "area": 10,
                "cell": 100,
                "unit": -1,
                "lon": 13.0,
                "lat": 47.0,
                "range": 1000,
                "samples": 5,
                "changeable": 1,
                "created": "2021-01-01",
                "updated": "2021-02-01",
                "averageSignal": 0,
            },
            {
                "radio": "GSM",
                "mcc": 232,
                "net": 1,
                "area": 10,
                "cell": 101,
                "unit": -1,
                "lon": 13.1,
                "lat": 47.1,
                "range": 2000,
                "samples": 8,
                "changeable": 1,
                "created": "2021-01-01",
                "updated": "2021-02-01",
                "averageSignal": 0,
            },
            {
                "radio": "LTE",
                "mcc": 310,
                "net": 1,
                "area": 10,
                "cell": 102,
                "unit": -1,
                "lon": 13.2,
                "lat": 47.2,
                "range": 3000,
                "samples": 9,
                "changeable": 1,
                "created": "2021-01-01",
                "updated": "2021-02-01",
                "averageSignal": 0,
            },
        ]
    )
    path = tmp_path / "tower_snapshot.csv.xz"
    with lzma.open(path, "wt", encoding="utf-8") as file:
        rows.to_csv(file, index=False)

    towers, summary = extract_country_lte_towers(path, chunk_size=2)

    assert len(towers) == 1
    assert towers.loc[0, "latitude"] == 47.0
    assert towers.loc[0, "longitude"] == 13.0
    assert towers.loc[0, "samples"] == 5
    assert towers.loc[0, "range"] == 1000
    assert summary == {
        "input_rows": 3,
        "country_code": 232,
        "country_rows": 2,
        "non_lte_country_rows": 1,
        "invalid_gps_lte_rows": 0,
        "output_rows": 1,
    }

    multi_country_towers, multi_country_summary = extract_country_lte_towers(
        path,
        country_code=(232, 310),
        chunk_size=2,
    )

    assert len(multi_country_towers) == 2
    assert multi_country_summary["country_codes"] == [232, 310]
    assert multi_country_summary["country_rows"] == 3
