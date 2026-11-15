"""Pure analytics helpers used by the interactive ConnectRoute dashboard."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def signal_band(rsrp_dbm: float, weak_threshold: float = -100.0) -> str:
    """Return a human-readable signal category for an RSRP value."""
    if rsrp_dbm >= -80:
        return "Strong"
    if rsrp_dbm >= weak_threshold:
        return "Usable"
    if rsrp_dbm >= weak_threshold - 10:
        return "Weak"
    return "Critical"


def profile_metrics(
    profile: pd.DataFrame,
    *,
    weak_threshold: float = -100.0,
    outage_threshold: float = -110.0,
) -> dict[str, float]:
    """Calculate distance-weighted route coverage metrics."""
    required = {"distance_along_route_m", "predicted_rsrp_dbm"}
    missing = required.difference(profile.columns)
    if missing:
        raise ValueError(f"Profile is missing columns: {', '.join(sorted(missing))}")
    if weak_threshold <= outage_threshold:
        raise ValueError("weak_threshold must be greater than outage_threshold.")
    distances = profile["distance_along_route_m"].to_numpy(dtype=float)
    signal = profile["predicted_rsrp_dbm"].to_numpy(dtype=float)
    if len(distances) < 2 or not np.isfinite(distances).all() or not np.isfinite(signal).all():
        raise ValueError("Profile must contain at least two finite samples.")
    if np.any(np.diff(distances) <= 0):
        raise ValueError("Profile distances must be strictly increasing.")
    segments = np.diff(distances)
    route_length = float(distances[-1] - distances[0])

    def weighted_fraction(mask: np.ndarray) -> float:
        return float(np.sum(segments * mask[:-1]) / route_length)

    return {
        "route_length_m": route_length,
        "mean_rsrp_dbm": float(np.average(signal, weights=np.r_[segments, segments[-1]])),
        "min_rsrp_dbm": float(signal.min()),
        "median_rsrp_dbm": float(np.median(signal)),
        "usable_fraction": weighted_fraction(signal >= weak_threshold),
        "weak_fraction": weighted_fraction(
            (signal < weak_threshold) & (signal >= outage_threshold)
        ),
        "outage_fraction": weighted_fraction(signal < outage_threshold),
        "longest_weak_stretch_m": _longest_stretch(
            segments, signal[:-1] < weak_threshold
        ),
        "longest_outage_stretch_m": _longest_stretch(
            segments, signal[:-1] < outage_threshold
        ),
        "threshold_rsrp_dbm": weak_threshold,
    }


def _longest_stretch(lengths: np.ndarray, mask: np.ndarray) -> float:
    longest = 0.0
    current = 0.0
    for length, selected in zip(lengths, mask, strict=True):
        current = current + float(length) if selected else 0.0
        longest = max(longest, current)
    return longest


def add_dashboard_fields(
    profile: pd.DataFrame,
    *,
    weak_threshold: float = -100.0,
    outage_threshold: float = -110.0,
) -> pd.DataFrame:
    """Add signal category and uncertainty width columns without mutating input."""
    result = profile.copy()
    result["signal_band"] = result["predicted_rsrp_dbm"].map(
        lambda value: signal_band(float(value), weak_threshold)
    )
    result["uncertainty_width_db"] = (
        result["estimated_upper_90_dbm"] - result["estimated_lower_90_dbm"]
    )
    result["below_weak_threshold"] = result["predicted_rsrp_dbm"] < weak_threshold
    result["below_outage_threshold"] = result["predicted_rsrp_dbm"] < outage_threshold
    return result


def find_hotspots(
    profile: pd.DataFrame,
    *,
    weak_threshold: float = -100.0,
    minimum_length_m: float = 500.0,
) -> pd.DataFrame:
    """Group consecutive weak samples into ranked route hotspots."""
    if minimum_length_m <= 0:
        raise ValueError("minimum_length_m must be positive.")
    enriched = add_dashboard_fields(profile, weak_threshold=weak_threshold)
    distances = enriched["distance_along_route_m"].to_numpy(dtype=float)
    weak = enriched["below_weak_threshold"].to_numpy(dtype=bool)
    groups: list[dict[str, Any]] = []
    start: int | None = None
    for index, is_weak in enumerate(np.r_[weak, False]):
        if is_weak and start is None:
            start = index
        elif not is_weak and start is not None:
            end = max(start, index - 1)
            length = float(distances[end] - distances[start])
            if end < len(distances) - 1:
                length += float(distances[end + 1] - distances[end])
            if length >= minimum_length_m:
                section = enriched.iloc[start : end + 1]
                groups.append(
                    {
                        "Hotspot": len(groups) + 1,
                        "Start (km)": distances[start] / 1000,
                        "End (km)": distances[end] / 1000,
                        "Length (km)": length / 1000,
                        "Mean RSRP (dBm)": float(section["predicted_rsrp_dbm"].mean()),
                        "Nearest tower (km)": float(
                            section["nearest_tower_distance_m"].median() / 1000
                        ),
                        "Latitude": float(section["latitude"].mean()),
                        "Longitude": float(section["longitude"].mean()),
                    }
                )
            start = None
    if not groups:
        return pd.DataFrame(
            columns=[
                "Hotspot",
                "Start (km)",
                "End (km)",
                "Length (km)",
                "Mean RSRP (dBm)",
                "Nearest tower (km)",
                "Latitude",
                "Longitude",
            ]
        )
    return pd.DataFrame(groups).sort_values("Length (km)", ascending=False).reset_index(
        drop=True
    )


def score_routes(
    rows: pd.DataFrame,
    *,
    priority: str = "Balanced",
) -> pd.DataFrame:
    """Rank routes using transparent, normalized signal and travel metrics."""
    required = {
        "Mean estimated RSRP (dBm)",
        "Estimated route below -110 dBm (%)",
        "Distance (km)",
        "Travel time (min)",
    }
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Route table is missing columns: {', '.join(sorted(missing))}")
    result = rows.copy()

    def normalize(column: str, reverse: bool = False) -> pd.Series:
        values = pd.to_numeric(result[column], errors="raise")
        span = float(values.max() - values.min())
        normalized = (
            pd.Series(1.0, index=result.index)
            if span == 0
            else (values - values.min()) / span
        )
        if span == 0:
            return normalized
        return 1 - normalized if reverse else normalized

    signal = normalize("Mean estimated RSRP (dBm)")
    risk = normalize("Estimated route below -110 dBm (%)", reverse=True)
    speed = normalize("Travel time (min)", reverse=True)
    distance = normalize("Distance (km)", reverse=True)
    weights = {
        "Best signal": (0.55, 0.25, 0.10, 0.10),
        "Lowest outage risk": (0.25, 0.55, 0.10, 0.10),
        "Fastest": (0.20, 0.15, 0.55, 0.10),
        "Shortest": (0.20, 0.15, 0.10, 0.55),
        "Balanced": (0.40, 0.30, 0.20, 0.10),
    }
    if priority not in weights:
        raise ValueError(f"Unknown route priority: {priority}")
    signal_weight, risk_weight, speed_weight, distance_weight = weights[priority]
    result["Route score"] = 100 * (
        signal_weight * signal
        + risk_weight * risk
        + speed_weight * speed
        + distance_weight * distance
    )
    result["Rank"] = result["Route score"].rank(method="min", ascending=False).astype(int)
    return result.sort_values(["Rank", "Route"]).reset_index(drop=True)


def recommend_tower_sites(
    profile: pd.DataFrame,
    *,
    weak_threshold: float = -100.0,
    minimum_length_m: float = 500.0,
    limit: int = 3,
) -> pd.DataFrame:
    """Suggest candidate tower sites at the center of the largest weak hotspots."""
    if limit < 1:
        raise ValueError("limit must be positive.")
    hotspots = find_hotspots(
        profile,
        weak_threshold=weak_threshold,
        minimum_length_m=minimum_length_m,
    )
    if hotspots.empty:
        return hotspots
    recommendations = hotspots.head(limit).copy()
    recommendations.insert(
        1,
        "Priority",
        np.arange(1, len(recommendations) + 1),
    )
    recommendations["Estimated improvement opportunity (km)"] = (
        recommendations["Length (km)"] * 0.65
    ).round(1)
    return recommendations
