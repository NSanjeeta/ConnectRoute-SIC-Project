"""Nominatim geocoding, OSRM route alternatives, and route sampling helpers."""

from __future__ import annotations

import os
import threading
import time
from typing import Any
from urllib.parse import quote

import numpy as np
import requests
from pyproj import Geod


NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OSRM_BASE_URL = "https://router.project-osrm.org"
REQUEST_TIMEOUT_SECONDS = 30
_GEOD = Geod(ellps="WGS84")
_LAST_GEOCODE_REQUEST = 0.0
_GEOCODE_LOCK = threading.Lock()


def geocode_place(place: str, *, session: requests.Session | None = None) -> dict[str, Any]:
    """Resolve a place with Nominatim and return its first matching result."""
    query = place.strip()
    if not query:
        raise ValueError("Place name cannot be empty.")
    client = session or requests.Session()
    user_agent = "ConnectRoute2/0.1 (route-planning research demonstration)"
    contact = os.getenv("CONNECTROUTE_CONTACT_EMAIL")
    if contact:
        user_agent += f"; contact: {contact}"
    global _LAST_GEOCODE_REQUEST
    with _GEOCODE_LOCK:
        wait = 1.0 - (time.monotonic() - _LAST_GEOCODE_REQUEST)
        if wait > 0:
            time.sleep(wait)
        _LAST_GEOCODE_REQUEST = time.monotonic()
        response = client.get(
            NOMINATIM_URL,
            params={"q": query, "format": "jsonv2", "limit": 1},
            headers={"User-Agent": user_agent},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    response.raise_for_status()
    results = response.json()
    if not results:
        raise ValueError(f"Nominatim returned no match for {query!r}.")
    result = results[0]
    return {
        "display_name": str(result["display_name"]),
        "latitude": float(result["lat"]),
        "longitude": float(result["lon"]),
    }


def fetch_route_alternatives(
    origin: dict[str, Any],
    destination: dict[str, Any],
    *,
    alternatives: int = 3,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Fetch driving route alternatives from the public OSRM demo service."""
    if not 1 <= alternatives <= 3:
        raise ValueError("OSRM demo alternatives must be between one and three.")
    coordinates = (
        f"{float(origin['longitude'])},{float(origin['latitude'])};"
        f"{float(destination['longitude'])},{float(destination['latitude'])}"
    )
    url = f"{OSRM_BASE_URL}/route/v1/driving/{quote(coordinates, safe=',;')}"
    response = (session or requests.Session()).get(
        url,
        params={
            "alternatives": alternatives,
            "overview": "full",
            "geometries": "geojson",
            "steps": "false",
        },
        headers={"User-Agent": "ConnectRoute2/0.1 route research demonstration"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    result = response.json()
    if result.get("code") != "Ok" or not result.get("routes"):
        raise ValueError(
            "OSRM could not find a drivable route for these places. "
            f"Service response: {result.get('code', 'unknown')}"
        )
    return result["routes"]


def sample_route_coordinates(
    coordinates_lon_lat: list[list[float]],
    spacing_m: float = 100.0,
) -> list[dict[str, float]]:
    """Resample a GeoJSON line at roughly fixed geodesic distance intervals."""
    if not np.isfinite(spacing_m) or spacing_m <= 0:
        raise ValueError("spacing_m must be positive and finite.")
    coordinates = np.asarray(coordinates_lon_lat, dtype=float)
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 2
        or len(coordinates) < 2
        or not np.isfinite(coordinates).all()
    ):
        raise ValueError("A route must contain at least two finite [longitude, latitude] points.")

    segment_lengths = np.empty(len(coordinates) - 1, dtype=float)
    segment_azimuths = np.empty_like(segment_lengths)
    for index in range(len(segment_lengths)):
        azimuth, _, distance = _GEOD.inv(
            coordinates[index, 0],
            coordinates[index, 1],
            coordinates[index + 1, 0],
            coordinates[index + 1, 1],
        )
        segment_azimuths[index] = azimuth
        segment_lengths[index] = distance
    total_distance = float(segment_lengths.sum())
    if total_distance <= 0:
        raise ValueError("Route geometry has zero length.")

    sample_distances = np.arange(0.0, total_distance, spacing_m)
    if len(sample_distances) == 0 or total_distance - sample_distances[-1] > spacing_m * 0.25:
        sample_distances = np.append(sample_distances, total_distance)
    else:
        sample_distances[-1] = total_distance

    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    samples: list[dict[str, float]] = []
    for sample_index, distance_along in enumerate(sample_distances):
        if distance_along >= total_distance:
            lon, lat = coordinates[-1]
        else:
            segment = int(np.searchsorted(cumulative, distance_along, side="right") - 1)
            segment = min(segment, len(segment_lengths) - 1)
            offset = float(distance_along - cumulative[segment])
            lon, lat, _ = _GEOD.fwd(
                coordinates[segment, 0],
                coordinates[segment, 1],
                segment_azimuths[segment],
                offset,
            )
        samples.append(
            {
                "sample_index": int(sample_index),
                "distance_along_route_m": float(distance_along),
                "longitude": float(lon),
                "latitude": float(lat),
            }
        )
    return samples


def calculate_route_risk_metrics(
    samples: list[dict[str, float]],
    outage_probabilities: np.ndarray,
    *,
    probability_cutoff: float = 0.5,
) -> dict[str, float]:
    """Summarize marginal expected outage distance without assuming independence."""
    probabilities = np.asarray(outage_probabilities, dtype=float)
    if len(samples) != len(probabilities) or len(samples) < 2:
        raise ValueError("At least two aligned route samples and outage probabilities are required.")
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise ValueError("Outage probabilities must be finite values between zero and one.")
    distances = np.asarray(
        [sample["distance_along_route_m"] for sample in samples],
        dtype=float,
    )
    if np.any(np.diff(distances) <= 0):
        raise ValueError("Route sample distances must be strictly increasing.")
    segment_lengths = np.diff(distances)
    expected_outage_distance = float(np.sum(probabilities[:-1] * segment_lengths))
    route_length = float(distances[-1] - distances[0])

    risky = probabilities >= probability_cutoff
    longest_risky_stretch = 0.0
    current_stretch = 0.0
    for index, length in enumerate(segment_lengths):
        if risky[index]:
            current_stretch += float(length)
            longest_risky_stretch = max(longest_risky_stretch, current_stretch)
        else:
            current_stretch = 0.0
    return {
        "route_length_m": route_length,
        "expected_outage_fraction": expected_outage_distance / route_length,
        "expected_outage_distance_m": expected_outage_distance,
        "longest_consecutive_high_risk_stretch_m": longest_risky_stretch,
        "high_risk_probability_cutoff": float(probability_cutoff),
    }
