import numpy as np
import pytest

from connectroute.routing import (
    calculate_route_risk_metrics,
    fetch_route_alternatives,
    geocode_place,
    sample_route_coordinates,
)


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _Session:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _Response(self.payload)


def test_geocode_uses_nominatim_and_returns_normalized_coordinates():
    session = _Session(
        [
            {
                "display_name": "Bengaluru, India",
                "lat": "12.97",
                "lon": "77.59",
            }
        ]
    )

    result = geocode_place(" Bengaluru ", session=session)

    assert result == {
        "display_name": "Bengaluru, India",
        "latitude": 12.97,
        "longitude": 77.59,
    }
    assert session.calls[0][1]["params"]["q"] == "Bengaluru"
    assert session.calls[0][1]["headers"]["User-Agent"].startswith("ConnectRoute2")


def test_geocode_reports_empty_results_and_rejects_empty_query():
    with pytest.raises(ValueError, match="cannot be empty"):
        geocode_place(" ")
    with pytest.raises(ValueError, match="no match"):
        geocode_place("unknown place", session=_Session([]))


def test_route_request_returns_osrm_alternatives_and_validates_count():
    route = {
        "distance": 1000,
        "duration": 120,
        "geometry": {"type": "LineString", "coordinates": [[77.0, 12.0], [77.01, 12.0]]},
    }
    session = _Session({"code": "Ok", "routes": [route]})
    origin = {"longitude": 77.0, "latitude": 12.0}
    destination = {"longitude": 77.01, "latitude": 12.0}

    result = fetch_route_alternatives(
        origin,
        destination,
        alternatives=2,
        session=session,
    )

    assert result == [route]
    assert session.calls[0][1]["params"]["alternatives"] == 2
    with pytest.raises(ValueError, match="between one and three"):
        fetch_route_alternatives(origin, destination, alternatives=4, session=session)


def test_sample_route_returns_endpoints_and_roughly_100m_spacing():
    samples = sample_route_coordinates(
        [[77.0, 12.0], [77.01, 12.0]],
        spacing_m=100,
    )

    assert samples[0]["distance_along_route_m"] == pytest.approx(0)
    assert samples[-1]["distance_along_route_m"] > samples[-2]["distance_along_route_m"]
    assert samples[-1]["longitude"] == pytest.approx(77.01)
    assert samples[-1]["latitude"] == pytest.approx(12.0)
    gaps = np.diff([sample["distance_along_route_m"] for sample in samples])
    assert np.max(gaps) <= 100
    assert np.median(gaps) == pytest.approx(100)


def test_route_risk_summaries_do_not_claim_joint_outage_probability():
    samples = [
        {"distance_along_route_m": 0.0},
        {"distance_along_route_m": 100.0},
        {"distance_along_route_m": 200.0},
        {"distance_along_route_m": 300.0},
    ]

    metrics = calculate_route_risk_metrics(
        samples,
        np.array([0.8, 0.9, 0.1, 0.2]),
        probability_cutoff=0.5,
    )

    assert metrics["expected_outage_distance_m"] == pytest.approx(180)
    assert metrics["expected_outage_fraction"] == pytest.approx(180 / 300)
    assert metrics["longest_consecutive_high_risk_stretch_m"] == pytest.approx(200)
    with pytest.raises(ValueError, match="between zero and one"):
        calculate_route_risk_metrics(samples, np.array([0.8, 1.2, 0.1, 0.2]))
