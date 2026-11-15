"""Illustrative route alternatives dashboard with automatic signal estimates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import folium
import numpy as np
import pandas as pd
import requests
import streamlit as st
from pyproj import CRS

from connectroute.evaluate import build_benchmark_tables
from connectroute.dashboard import (
    add_dashboard_fields,
    find_hotspots,
    profile_metrics,
    recommend_tower_sites,
    score_routes,
)
from connectroute.route_predict import (
    DEFAULT_SALZBURG_FEATURES,
    DEFAULT_SALZBURG_PREDICTIONS,
    DEFAULT_TOWERS,
    fit_route_projection_model,
    predict_route_signal,
)
from connectroute.route_terrain import (
    add_route_terrain_features,
    prepare_route_dem,
)
from connectroute.routing import (
    fetch_route_alternatives,
    geocode_place,
    sample_route_coordinates,
)


st.set_page_config(page_title="ConnectRoute 2.0", page_icon="📶", layout="wide")
st.markdown(
    """
    <style>
    :root {
        --cr-ink: #202124;
        --cr-muted: #5f6368;
        --cr-line: #dadce0;
        --cr-blue: #1a73e8;
        --cr-teal: #188038;
        --cr-surface: #f8f9fa;
        --cr-lilac: #7c3aed;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 78% 10%, rgba(170, 221, 239, .84) 0 5%, transparent 5.4% 18%),
            radial-gradient(ellipse at 8% 82%, rgba(176, 224, 194, .9) 0 10%, transparent 10.5% 25%),
            radial-gradient(ellipse at 57% 69%, rgba(185, 228, 199, .88) 0 8%, transparent 8.5% 20%),
            radial-gradient(ellipse at 36% 29%, rgba(174, 221, 187, .72) 0 6%, transparent 6.5% 18%),
            linear-gradient(115deg, rgba(255,255,255,.42), rgba(244,239,222,.2)),
            repeating-linear-gradient(25deg, transparent 0 108px, rgba(139, 158, 183, .6) 109px 111px, rgba(255,255,255,.65) 112px 114px, transparent 115px 220px),
            repeating-linear-gradient(155deg, transparent 0 166px, rgba(192, 151, 127, .46) 167px 169px, rgba(255,255,255,.62) 170px 172px, transparent 173px 310px),
            linear-gradient(180deg, #dce8d7 0%, #f3eddf 48%, #d7e7d5 100%);
        color: var(--cr-ink);
    }
    .stApp::before {
        background:
            linear-gradient(72deg, transparent 0 43%, rgba(255,255,255,.9) 43.1% 43.32%, rgba(121,151,192,.72) 43.4% 43.7%, rgba(255,255,255,.9) 43.8% 44%, transparent 44.1%),
            linear-gradient(164deg, transparent 0 62%, rgba(255,255,255,.88) 62.1% 62.32%, rgba(126,154,190,.7) 62.4% 62.7%, rgba(255,255,255,.88) 62.8% 63%, transparent 63.1%),
            radial-gradient(ellipse at 25% 27%, rgba(105, 198, 226, .72) 0 3%, transparent 3.5% 12%),
            radial-gradient(ellipse at 73% 80%, rgba(86, 164, 112, .48) 0 10%, transparent 10.5% 19%),
            repeating-linear-gradient(6deg, transparent 0 285px, rgba(117, 141, 181, .62) 286px 288px, transparent 289px 430px);
        content: "";
        inset: 0;
        opacity: .95;
        pointer-events: none;
        position: fixed;
        z-index: 0;
    }
    [data-testid="stAppViewContainer"] > .main {
        position: relative;
        z-index: 1;
    }
    [data-testid="stHeader"] {
        background: transparent;
        height: 0;
        visibility: hidden;
    }
    [data-testid="stMain"] .block-container {
        max-width: 1440px;
        padding: 4.2rem clamp(1rem, 3vw, 3rem) 4rem;
    }
    [data-testid="stMarkdownContainer"] p { color: var(--cr-muted); }
    .cr-eyebrow {
        align-items: center;
        color: #2563eb;
        display: flex;
        gap: .55rem;
        font-size: .75rem;
        font-weight: 800;
        letter-spacing: .16em;
        text-transform: uppercase;
        margin: 0 0 .65rem;
    }
    .cr-brandbar {
        align-items: center;
        animation: cr-rise .35s ease-out both;
        background: rgba(255, 255, 255, .96);
        border: 1px solid #dadce0;
        border-radius: 999px;
        box-shadow: 0 6px 20px rgba(49, 77, 111, .08);
        display: flex;
        gap: .75rem;
        margin: 0 0 1.55rem;
        padding: .45rem .9rem .45rem .5rem;
        width: fit-content;
    }
    .cr-brandmark {
        align-items: center;
        background: linear-gradient(135deg, #1a73e8, #188038);
        border-radius: 12px;
        box-shadow: 0 8px 18px rgba(37, 99, 235, .2);
        color: #fff;
        display: flex;
        font-size: 1.1rem;
        font-weight: 850;
        height: 2.45rem;
        justify-content: center;
        letter-spacing: -.04em;
        width: 2.45rem;
    }
    .cr-brandname {
        color: var(--cr-ink);
        font-size: 1.15rem;
        font-weight: 850;
        letter-spacing: -.04em;
    }
    .cr-brandtag {
        color: var(--cr-muted);
        font-size: .73rem;
        letter-spacing: .08em;
        text-transform: uppercase;
    }
    .cr-eyebrow::before {
        background: linear-gradient(90deg, var(--cr-blue), var(--cr-teal));
        border-radius: 999px;
        content: "";
        height: .45rem;
        width: 2.2rem;
    }
    .cr-title {
        animation: cr-rise .55s ease-out both;
        color: var(--cr-ink);
        font-size: clamp(2.1rem, 4vw, 3.35rem);
        font-weight: 800;
        letter-spacing: -.055em;
        line-height: 1.04;
        margin: 0;
    }
    .cr-subtitle {
        animation: cr-rise .7s ease-out both;
        color: var(--cr-muted);
        font-size: 1.04rem;
        line-height: 1.6;
        margin: .8rem 0 1.35rem;
        max-width: 760px;
    }
    [data-testid="stForm"] {
        animation: cr-rise .85s ease-out both;
        background: rgba(255, 255, 255, .93);
        backdrop-filter: blur(7px);
        border: 1px solid var(--cr-line);
        border-radius: 18px;
        box-shadow: 0 8px 26px rgba(60, 64, 67, .22);
        margin-top: -.25rem;
        padding: 1.2rem 1.3rem .5rem;
    }
    .cr-control-card {
        background: rgba(255, 255, 255, .94);
        backdrop-filter: blur(7px);
        border: 1px solid #dadce0;
        border-radius: 16px;
        box-shadow: 0 6px 20px rgba(60, 64, 67, .16);
        margin: .9rem 0 1.1rem;
        padding: .9rem 1rem .35rem;
    }
    .cr-control-title {
        color: #202124;
        font-size: .92rem;
        font-weight: 750;
        margin: 0 0 .45rem;
    }
    .cr-control-note {
        color: #5f6368;
        font-size: .78rem;
        margin: 0 0 .65rem;
    }
    .cr-map-shell {
        animation: cr-rise .55s ease-out both;
        background: #e8f0e3;
        border: 1px solid #c9d6c5;
        border-radius: 20px;
        box-shadow: 0 16px 38px rgba(49, 77, 111, .12);
        margin: .55rem 0 1.35rem;
        overflow: hidden;
        padding: .35rem;
    }
    .cr-map-toolbar {
        align-items: center;
        background: rgba(255, 255, 255, .96);
        border: 1px solid #dadce0;
        border-radius: 12px;
        color: #5f6368;
        display: flex;
        font-size: .78rem;
        gap: .55rem;
        margin: .35rem .35rem .65rem;
        padding: .55rem .75rem;
    }
    .cr-map-live-dot {
        background: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 0 4px rgba(16, 185, 129, .12);
        height: .55rem;
        width: .55rem;
    }
    .cr-map-toolbar strong { color: #202124; }
    [data-testid="stForm"] label {
        color: var(--cr-ink);
        font-size: .85rem;
        font-weight: 700;
    }
    [data-testid="stTextInput"] input,
    [data-testid="stSelectbox"] [role="combobox"] {
        border-radius: 10px;
        min-height: 2.8rem;
    }
    .stButton button, [data-testid="stFormSubmitButton"] button {
        background: #1a73e8;
        border: 0;
        border-radius: 10px;
        color: white;
        font-weight: 750;
        min-height: 2.8rem;
        transition: transform .2s ease, box-shadow .2s ease, filter .2s ease;
        width: 100%;
    }
    .stButton button:hover, [data-testid="stFormSubmitButton"] button:hover {
        box-shadow: 0 3px 9px rgba(26, 115, 232, .3);
        color: white;
        transform: translateY(-1px);
        filter: saturate(1.15);
    }
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, .97);
        border-right: 1px solid #dadce0;
        box-shadow: 4px 0 18px rgba(60, 64, 67, .12);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #53657d !important;
    }
    [data-testid="stSidebar"] h2 {
        color: #17243b;
        letter-spacing: -.03em;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"],
    [data-testid="stSidebar"] [data-testid="stSlider"],
    [data-testid="stSidebar"] [data-testid="stCheckbox"] {
        background: #f8f9fa;
        border: 1px solid #dadce0;
        border-radius: 14px;
        margin: .45rem 0 .8rem;
        padding: .55rem .65rem;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [role="combobox"] {
        background: #fff;
        border-color: #bdc1c6;
        color: #202124;
    }
    [data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
        background: #2563eb;
    }
    .cr-studio-card {
        background: linear-gradient(135deg, #eaf3ff, #e9fbf8);
        border: 1px solid #d1e4f8;
        border-radius: 16px;
        color: #385273;
        margin: .65rem 0 1rem;
        padding: .85rem;
    }
    .cr-studio-card strong { color: #173c70; }
    .cr-studio-card small { color: #5b718e; }
    .cr-divider {
        background: linear-gradient(90deg, transparent, #dce5f2, transparent);
        height: 1px;
        margin: 1.5rem 0;
    }
    @keyframes cr-rise {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @media (prefers-reduced-motion: reduce) {
        .cr-title, .cr-subtitle, [data-testid="stForm"] { animation: none; }
    }
    .cr-section-title {
        color: var(--cr-ink);
        font-size: 1.22rem;
        font-weight: 780;
        letter-spacing: -.025em;
        margin: .4rem 0 .15rem;
    }
    .cr-section-note { color: var(--cr-muted); font-size: .88rem; margin: 0 0 .8rem; }
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, .96);
        border: 1px solid var(--cr-line);
        border-radius: 14px;
        padding: .9rem 1rem;
    }
    [data-testid="stMetricLabel"] p {
        color: var(--cr-muted);
        font-size: .78rem;
        font-weight: 650;
    }
    [data-testid="stMetricValue"] {
        color: var(--cr-ink);
        font-size: 1.55rem;
        font-weight: 780;
    }
    [data-testid="stTabs"] button { font-weight: 700; }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--cr-line);
        border-radius: 12px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="cr-brandbar">
      <div class="cr-brandmark">CR</div>
      <div><div class="cr-brandname">ConnectRoute</div>
      <div class="cr-brandtag">Signal-aware journey planning</div></div>
    </div>
    <div class="cr-eyebrow">ConnectRoute <span aria-hidden="true">/</span> Route intelligence</div>
    <h1 class="cr-title">Plan your journey.<br>See the signal ahead.</h1>
    <p class="cr-subtitle">
      Compare driving routes and explore estimated LTE signal and terrain along the way.
      Start with two places below.
    </p>
    """,
    unsafe_allow_html=True,
)
with st.form("route_search_form", border=False):
    start_column, destination_column, spacing_column, submit_column = st.columns(
        [1.2, 1.2, 0.75, 0.8],
        vertical_alignment="bottom",
    )
    with start_column:
        origin_text = st.text_input(
            "Starting point",
            "Bengaluru, Karnataka, India",
            placeholder="City, address, or landmark",
        )
    with destination_column:
        destination_text = st.text_input(
            "Destination",
            "Mysuru, Karnataka, India",
            placeholder="City, address, or landmark",
        )
    with spacing_column:
        sample_spacing = st.selectbox(
            "Profile detail",
            options=[50, 100, 200],
            index=1,
            format_func=lambda value: f"{value} m",
        )
    with submit_column:
        submitted = st.form_submit_button(
            "Explore routes",
            type="primary",
            icon=":material/route:",
        )
st.caption(
    "Map and routing: OpenStreetMap / OSRM · Tower records: OpenCelliD, CC BY-SA 4.0 · "
    "Elevation: Copernicus DEM GLO-30 DSM"
)
with st.container():
    st.markdown(
        """
        <div class="cr-control-card">
          <div class="cr-control-title">Map controls</div>
          <div class="cr-control-note">
            Choose what matters most for your journey and tune the coverage view.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    priority_column, signal_column, coverage_column = st.columns(
        [1.15, 1, 1], gap="medium"
    )
    with priority_column:
        route_priority = st.selectbox(
            "Route priority",
            ["Balanced", "Best signal", "Lowest outage risk", "Fastest", "Shortest"],
            help="Controls the transparent route score shown in the comparison table.",
        )
    with signal_column:
        weak_threshold = st.slider(
            "Weak signal threshold (dBm)",
            min_value=-115,
            max_value=-85,
            value=-100,
            step=1,
        )
        outage_threshold = st.slider(
            "Outage threshold (dBm)",
            min_value=-120,
            max_value=-95,
            value=-110,
            step=1,
        )
    with coverage_column:
        hotspot_length_km = st.slider(
            "Minimum hotspot length (km)",
            min_value=0.2,
            max_value=10.0,
            value=0.5,
            step=0.1,
        )
        show_tower_points = st.checkbox(
            "Show nearest LTE towers",
            value=True,
            help="Displays only towers matched to sampled route points, keeping the map readable.",
        )
        show_uncertainty = st.checkbox("Show uncertainty band", value=True)
if weak_threshold <= outage_threshold:
    st.error("Weak threshold must be higher than outage threshold.")
    weak_threshold, outage_threshold = -100, -110

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_FILES = {
    "propagation": PROJECT_ROOT / "data" / "processed" / "spatial_baseline_predictions.parquet",
    "ml": PROJECT_ROOT / "data" / "processed" / "spatial_model_predictions.parquet",
    "features": PROJECT_ROOT / "data" / "processed" / "salzburg_propagation_features.parquet",
    "summary": PROJECT_ROOT / "results" / "model_summary.json",
}


@st.cache_data(show_spinner=False, ttl=3600)
def _geocode(place: str) -> dict[str, Any]:
    return geocode_place(place)


@st.cache_data(show_spinner=False, ttl=900)
def _routes(origin: dict[str, Any], destination: dict[str, Any]) -> list[dict[str, Any]]:
    return fetch_route_alternatives(origin, destination, alternatives=3)


@st.cache_data(show_spinner=False)
def _india_towers(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def _terrain_profile(
    profile: pd.DataFrame,
    dem_path: str,
    crs_string: str,
) -> pd.DataFrame:
    return add_route_terrain_features(
        profile,
        Path(dem_path),
        CRS.from_user_input(crs_string),
    )


@st.cache_data(show_spinner=False)
def _route_model(
    feature_path: str,
    prediction_path: str,
) -> tuple[dict[str, float | int], dict[str, float]]:
    return fit_route_projection_model(Path(feature_path), Path(prediction_path))


@st.cache_data(show_spinner=False)
def _model_benchmark(
    propagation_path: str,
    ml_path: str,
    features_path: str,
    summary_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    propagation_columns = [
        "fold",
        "rsrp_dbm",
        "predicted_free_space_rsrp_dbm",
        "predicted_cost231_hata_rsrp_dbm",
        "predicted_hata_rural_rsrp_dbm",
        "predicted_3gpp_rma_los_rsrp_dbm",
        "predicted_3gpp_rma_nlos_rsrp_dbm",
        "predicted_3gpp_uma_los_rsrp_dbm",
        "predicted_3gpp_uma_nlos_rsrp_dbm",
        "predicted_fitted_log_distance_rsrp_dbm",
    ]
    ml_columns = [
        "fold",
        "rsrp_dbm",
        "pure_lightgbm_rsrp_dbm",
        "hybrid_lightgbm_rsrp_dbm",
        "hybrid_xgboost_rsrp_dbm",
        "quantile_50_rsrp_dbm",
    ]
    feature_columns = [
        "rsrp_dbm",
        "best_server_cost231_hata_in_domain",
        "best_server_hata_rural_in_domain",
        "best_server_3gpp_rma_los_in_domain",
        "best_server_3gpp_rma_nlos_in_domain",
        "best_server_3gpp_uma_los_in_domain",
        "best_server_3gpp_uma_nlos_in_domain",
    ]
    propagation_metrics, ml_metrics = build_benchmark_tables(
        pd.read_parquet(propagation_path, columns=propagation_columns),
        pd.read_parquet(ml_path, columns=ml_columns),
        pd.read_parquet(features_path, columns=feature_columns),
    )
    with Path(summary_path).open(encoding="utf-8") as summary_file:
        model_summary = json.load(summary_file)
    return propagation_metrics, ml_metrics, model_summary


def _signal_profiles(
    routes: list[dict[str, Any]],
    samples_by_route: list[list[dict[str, float]]],
) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    if not DEFAULT_TOWERS.is_file():
        raise FileNotFoundError(
            "India LTE tower data is not installed. Run "
            r".\.venv\Scripts\python.exe -m connectroute.tower_ingest "
            "--country-code 404 405 --output data\\processed\\"
            "opencellid_india_lte_2021.parquet --summary "
            "results\\india_tower_ingest_summary.json."
        )
    towers = _india_towers(str(DEFAULT_TOWERS))
    fitted, residuals = _route_model(
        str(DEFAULT_SALZBURG_FEATURES),
        str(DEFAULT_SALZBURG_PREDICTIONS),
    )
    profiles: list[pd.DataFrame] = []
    details: list[dict[str, Any]] = []
    for route, samples in zip(routes, samples_by_route, strict=True):
        profile, summary = predict_route_signal(samples, towers, fitted, residuals)
        profile["route_distance_m"] = float(route["distance"])
        profile["route_duration_s"] = float(route["duration"])
        profiles.append(profile)
        details.append(summary)
    return profiles, {
        "eligible_india_towers": int(
            max(item["eligible_lte_towers"] for item in details)
        ),
        "route_details": details,
        "model": fitted,
        "residual_calibration": residuals,
    }


def _terrain_profiles(
    profiles: list[pd.DataFrame],
) -> tuple[list[pd.DataFrame], dict[str, str]]:
    route_coordinates = np.vstack(
        [
            profile[["longitude", "latitude"]].to_numpy(dtype=float)
            for profile in profiles
        ]
    )
    nearest_tower_coordinates = np.vstack(
        [
            profile[["nearest_tower_longitude", "nearest_tower_latitude"]].to_numpy(
                dtype=float
            )
            for profile in profiles
        ]
    )
    projected_dem, projected_crs = prepare_route_dem(
        route_coordinates,
        nearest_tower_coordinates,
    )
    profiles = [
        _terrain_profile(profile, str(projected_dem), projected_crs.to_string())
        for profile in profiles
    ]
    return profiles, {
        "terrain_dem": "Copernicus DEM GLO-30 Public (DSM)",
        "terrain_projected_crs": projected_crs.to_string(),
    }


def _route_color(
    rsrp: float,
    *,
    weak_threshold: float = -100.0,
    outage_threshold: float = -110.0,
) -> str:
    if rsrp >= -80:
        return "#15803d"
    if rsrp >= weak_threshold:
        return "#eab308"
    if rsrp >= outage_threshold:
        return "#f97316"
    return "#dc2626"


with st.expander("Salzburg model benchmark", expanded=True):
    st.caption(
        "These tables compare saved out-of-fold predictions against measured Salzburg RSRP. "
        "They are separate from the illustrative Karnataka route projection below."
    )
    missing_benchmark_files = [
        path for path in BENCHMARK_FILES.values() if not path.is_file()
    ]
    if missing_benchmark_files:
        missing_names = ", ".join(
            str(path.relative_to(PROJECT_ROOT)) for path in missing_benchmark_files
        )
        st.info(
            "Saved benchmark results are unavailable. Generate the feature, propagation, "
            "and model outputs with `experiments\\run_pipeline.ps1`. Missing: "
            f"{missing_names}"
        )
    else:
        benchmark_paths = tuple(
            str(BENCHMARK_FILES[key])
            for key in ("propagation", "ml", "features", "summary")
        )
        propagation_metrics, ml_metrics, model_summary = _model_benchmark(
            *benchmark_paths
        )
        propagation_tab, ml_tab = st.tabs(
            ["Propagation and Hata", "Machine learning"]
        )
        with propagation_tab:
            shown_metrics = propagation_metrics.copy()
            shown_metrics[["RMSE (dB)", "MAE (dB)", "Bias (dB)"]] = shown_metrics[
                ["RMSE (dB)", "MAE (dB)", "Bias (dB)"]
            ].round(2)
            shown_metrics["Best-server in domain (%)"] = shown_metrics[
                "Best-server in domain (%)"
            ].round(1)
            st.dataframe(shown_metrics, hide_index=True, width="stretch")
            st.warning(
                "COST-231 Hata and rural Hata are outside their stated applicability "
                "domain for all best-server Salzburg samples (0% in domain). Their "
                "RMSE values are extrapolations, not evidence that Hata works well "
                "or poorly in-domain."
            )
        with ml_tab:
            trial_count = model_summary["tuning_trials_per_model_per_outer_fold"]
            old_baseline = model_summary["physics_residual_baseline"]
            st.warning(
                f"Historical development run: {trial_count} tuning trials per model "
                f"and the old {old_baseline} physics baseline. The current model code "
                "uses a fitted log-distance baseline, so these ML scores are stale "
                "and are not a fair comparison with the propagation table. Rerun the "
                "full pipeline before selecting a model."
            )
            shown_ml_metrics = ml_metrics.round(2)
            st.dataframe(shown_ml_metrics, hide_index=True, width="stretch")
        st.caption(
            "All scores are dB error against measured RSRP. Lower RMSE/MAE is better; "
            "bias is prediction minus measurement. Spatial evaluation uses five "
            "contiguous folds with a 500 m training buffer."
        )

if submitted:
    for key in (
        "route_options",
        "route_samples",
        "signal_profiles",
        "signal_summary",
        "origin",
        "destination",
    ):
        st.session_state.pop(key, None)
    try:
        with st.spinner("Finding routes and profiling terrain…"):
            origin = _geocode(origin_text)
            destination = _geocode(destination_text)
            route_options = _routes(origin, destination)
            samples_by_route = [
                sample_route_coordinates(
                    route["geometry"]["coordinates"],
                    spacing_m=float(sample_spacing),
                )
                for route in route_options
            ]
            signal_profiles, signal_summary = _signal_profiles(
                route_options,
                samples_by_route,
            )
        st.session_state["route_options"] = route_options
        st.session_state["route_samples"] = samples_by_route
        st.session_state["signal_profiles"] = signal_profiles
        st.session_state["signal_summary"] = signal_summary
        st.session_state["origin"] = origin
        st.session_state["destination"] = destination
    except FileNotFoundError as error:
        st.error(str(error))
    except (ValueError, KeyError) as error:
        st.error(f"Could not build route alternatives: {error}")
    except requests.RequestException as error:
        st.error(
            "A route service request failed. Check your internet connection and "
            f"the public service status, then try again. Details: {error}"
        )

routes = st.session_state.get("route_options")
route_samples = st.session_state.get("route_samples")
signal_profiles = st.session_state.get("signal_profiles")
if routes and route_samples and signal_profiles:
    has_terrain = all(
        "terrain_diffraction_loss_db" in profile.columns
        for profile in signal_profiles
    )
    route_rows: list[dict[str, float | int | str]] = []
    for index, (route, samples, profile) in enumerate(
        zip(routes, route_samples, signal_profiles, strict=True),
        start=1,
    ):
        metrics = profile_metrics(
            profile,
            weak_threshold=weak_threshold,
            outage_threshold=outage_threshold,
        )
        row: dict[str, float | int | str] = {
            "Route": index,
            "Distance (km)": float(route["distance"]) / 1000,
            "Travel time (min)": float(route["duration"]) / 60,
            "Sampled points": len(samples),
            "Mean estimated RSRP (dBm)": metrics["mean_rsrp_dbm"],
            "Estimated route below -110 dBm (%)": 100 * metrics["outage_fraction"],
            "Weak coverage (%)": 100 * (metrics["weak_fraction"] + metrics["outage_fraction"]),
            "Longest weak stretch (km)": metrics["longest_weak_stretch_m"] / 1000,
            "Median nearest tower (km)": float(
                profile["nearest_tower_distance_m"].median() / 1000
            ),
        }
        if has_terrain:
            row["Median terrain diffraction loss (dB)"] = float(
                profile["terrain_diffraction_loss_db"].median()
            )
            row["Route samples in possible valley terrain (%)"] = float(
                100
                * profile["terrain_context"]
                .eq("local low ground (possible valley)")
                .mean()
            )
        route_rows.append(row)

    route_table = score_routes(pd.DataFrame(route_rows), priority=route_priority)
    st.markdown('<div class="cr-section-title">Your route options</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="cr-section-note">Choose a route to inspect its signal profile. '
        'Terrain context can be calculated separately.</p>',
        unsafe_allow_html=True,
    )
    with st.expander(f"Compare all {len(routes)} routes", expanded=len(routes) > 1):
        display_columns = [
            "Rank",
            "Route",
            "Distance (km)",
            "Travel time (min)",
            "Mean estimated RSRP (dBm)",
            "Weak coverage (%)",
            "Estimated route below -110 dBm (%)",
            "Route score",
        ]
        if has_terrain:
            display_columns.append("Median terrain diffraction loss (dB)")
        display_table = route_table[display_columns].round(2)
        st.dataframe(display_table, hide_index=True, width="stretch")
        st.caption(
            f"Route score prioritizes **{route_priority.lower()}**. "
            "Score is a transparent decision aid, not a new prediction model."
        )
    selected_route_index = st.selectbox(
        "Route to inspect",
        options=list(range(len(routes))),
        format_func=lambda index: (
            f"Route {index + 1} · {route_rows[index]['Distance (km)']:.1f} km · "
            f"{route_rows[index]['Travel time (min)']:.0f} min"
        ),
        label_visibility="collapsed",
        key="selected_route_index",
    )
    selected_route = routes[selected_route_index]
    selected_samples = route_samples[selected_route_index]
    selected_profile = add_dashboard_fields(
        signal_profiles[selected_route_index],
        weak_threshold=weak_threshold,
        outage_threshold=outage_threshold,
    )
    summary_row = route_table.loc[
        route_table["Route"].eq(selected_route_index + 1)
    ].iloc[0]
    mean_rsrp = float(summary_row["Mean estimated RSRP (dBm)"])
    st.markdown(
        f'<div class="cr-section-title">Route {selected_route_index + 1}</div>'
        f'<p class="cr-section-note">{st.session_state["origin"]["display_name"]} '
        f'<span aria-hidden="true">→</span> {st.session_state["destination"]["display_name"]}</p>',
        unsafe_allow_html=True,
    )
    metric_columns = st.columns(4)
    metric_columns[0].metric("Distance", f"{summary_row['Distance (km)']:.1f} km")
    metric_columns[1].metric("Drive time", f"{summary_row['Travel time (min)']:.0f} min")
    metric_columns[2].metric("Avg. estimated RSRP", f"{mean_rsrp:.1f} dBm")
    metric_columns[3].metric(
        "Route score",
        f"{summary_row['Route score']:.0f}/100",
    )
    st.caption(
        f"Rank {int(summary_row['Rank'])} for the **{route_priority.lower()}** priority. "
        f"Median nearest tower: {summary_row['Median nearest tower (km)']:.1f} km."
    )
    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Weak coverage",
        f"{summary_row['Weak coverage (%)']:.1f}%",
    )
    metric_columns[1].metric(
        "Estimated outage",
        f"{summary_row['Estimated route below -110 dBm (%)']:.1f}%",
    )
    metric_columns[2].metric(
        "Longest weak stretch",
        f"{summary_row['Longest weak stretch (km)']:.1f} km",
    )
    metric_columns[3].metric(
        "Nearest tower median",
        f"{summary_row['Median nearest tower (km)']:.1f} km",
    )
    map_column, profile_column = st.columns((1.15, 0.85), gap="large")
    center = [
        float(selected_profile["latitude"].mean()),
        float(selected_profile["longitude"].mean()),
    ]
    route_map = folium.Map(
        location=center,
        zoom_start=8,
        control_scale=True,
        tiles="OpenStreetMap",
        prefer_canvas=True,
    )
    for route_index, (route, profile) in enumerate(
        zip(routes, signal_profiles, strict=True)
    ):
        if route_index == selected_route_index:
            for point_index in range(len(profile) - 1):
                first = profile.iloc[point_index]
                second = profile.iloc[point_index + 1]
                mean_signal = (
                    float(first["predicted_rsrp_dbm"])
                    + float(second["predicted_rsrp_dbm"])
                ) / 2
                folium.PolyLine(
                    [
                        (float(first["latitude"]), float(first["longitude"])),
                        (float(second["latitude"]), float(second["longitude"])),
                    ],
                    color=_route_color(
                        mean_signal,
                        weak_threshold=weak_threshold,
                        outage_threshold=outage_threshold,
                    ),
                    weight=7,
                    opacity=0.9,
                    tooltip=(
                        f"Estimated RSRP {mean_signal:.1f} dBm"
                        + (
                            f" · terrain diffraction "
                            f"{first['terrain_diffraction_loss_db']:.1f} dB"
                            if has_terrain
                            else ""
                        )
                    ),
                ).add_to(route_map)
            for point_index in range(0, len(profile), max(1, len(profile) // 80)):
                record = profile.iloc[point_index]
                folium.CircleMarker(
                    location=(record["latitude"], record["longitude"]),
                    radius=4,
                    color=_route_color(
                        float(record["predicted_rsrp_dbm"]),
                        weak_threshold=weak_threshold,
                        outage_threshold=outage_threshold,
                    ),
                    fill=True,
                    fill_opacity=0.85,
                    tooltip=(
                        f"Estimated RSRP {record['predicted_rsrp_dbm']:.1f} dBm"
                        + (
                            f" · elevation {record['receiver_elevation_m']:.0f} m · "
                            f"{record['terrain_context']}"
                            if has_terrain
                            else ""
                        )
                    ),
                ).add_to(route_map)
            if show_tower_points:
                tower_points = profile.iloc[:: max(1, len(profile) // 35)]
                for _, tower in tower_points.iterrows():
                    folium.CircleMarker(
                        location=(
                            float(tower["nearest_tower_latitude"]),
                            float(tower["nearest_tower_longitude"]),
                        ),
                        radius=5,
                        color="#2563eb",
                        fill=True,
                        fill_color="#60a5fa",
                        fill_opacity=0.8,
                        tooltip=(
                            f"LTE tower · {float(tower['nearest_tower_distance_m']) / 1000:.2f} km "
                            f"from route · {int(tower['nearest_tower_samples']):,} samples"
                        ),
                    ).add_to(route_map)
        else:
            alternate_coordinates = [
                (float(latitude), float(longitude))
                for longitude, latitude in route["geometry"]["coordinates"]
            ]
            folium.PolyLine(
                alternate_coordinates,
                color="#9aa8bb",
                weight=3,
                opacity=0.55,
                tooltip=f"Alternative route {route_index + 1}",
            ).add_to(route_map)
    start_point = selected_profile.iloc[0]
    end_point = selected_profile.iloc[-1]
    folium.Marker(
        [start_point["latitude"], start_point["longitude"]],
        tooltip="Start",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(route_map)
    folium.Marker(
        [end_point["latitude"], end_point["longitude"]],
        tooltip="Destination",
        icon=folium.Icon(color="red", icon="flag"),
    ).add_to(route_map)
    route_map.fit_bounds(
        [
            [
                float(selected_profile["latitude"].min()),
                float(selected_profile["longitude"].min()),
            ],
            [
                float(selected_profile["latitude"].max()),
                float(selected_profile["longitude"].max()),
            ],
        ],
        padding=(24, 24),
    )
    with map_column:
        st.markdown(
            f'<div class="cr-map-toolbar"><span>Signal legend:</span>'
            f'<strong style="color:#15803d">Strong</strong> · '
            f'<strong style="color:#eab308">Usable</strong> · '
            f'<strong style="color:#f97316">Weak</strong> · '
            f'<strong style="color:#dc2626">Critical</strong>'
            f'<span>({weak_threshold} / {outage_threshold} dBm thresholds)</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="cr-map-shell">', unsafe_allow_html=True)
        st.iframe(
            route_map.get_root().render(),
            width="stretch",
            height=500,
            tab_index=0,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with profile_column:
        with st.container(border=True):
            st.markdown("**Route intelligence**")
            signal_tab, terrain_tab = st.tabs(["Signal profile", "Terrain profile"])
            with signal_tab:
                st.line_chart(
                    selected_profile,
                    x="distance_along_route_m",
                    y=(
                        [
                            "predicted_rsrp_dbm",
                            "estimated_lower_90_dbm",
                            "estimated_upper_90_dbm",
                        ]
                        if show_uncertainty
                        else ["predicted_rsrp_dbm"]
                    ),
                    x_label="Distance along route (m)",
                    y_label="Estimated RSRP (dBm)",
                    height=320,
                    width="stretch",
                )
            with terrain_tab:
                if has_terrain:
                    low_ground_fraction = float(
                        selected_profile["terrain_context"]
                        .eq("local low ground (possible valley)")
                        .mean()
                    )
                    st.line_chart(
                        selected_profile,
                        x="distance_along_route_m",
                        y="receiver_elevation_m",
                        x_label="Distance along route (m)",
                        y_label="Surface elevation (m)",
                        height=220,
                        width="stretch",
                    )
                    st.line_chart(
                        selected_profile,
                        x="distance_along_route_m",
                        y="terrain_diffraction_loss_db",
                        x_label="Distance along route (m)",
                        y_label="Terrain diffraction (dB)",
                        height=220,
                        width="stretch",
                    )
                    st.caption(
                        f"{low_ground_fraction:.0%} of samples are at least 10 m below "
                        "their 500 m local terrain mean (possible low ground, not a "
                        "validated valley classification). Terrain does not adjust RSRP."
                    )
                else:
                    st.caption(
                        "Terrain is optional and may take longer on first use while "
                        "Copernicus elevation data is prepared."
                    )
                    if st.button(
                        "Calculate terrain context",
                        key=f"calculate_terrain_{selected_route_index}",
                        icon=":material/landscape:",
                    ):
                        try:
                            with st.spinner("Preparing terrain profiles…"):
                                terrain_profiles, terrain_summary = _terrain_profiles(
                                    signal_profiles
                                )
                            st.session_state["signal_profiles"] = terrain_profiles
                            st.session_state["signal_summary"].update(terrain_summary)
                            st.rerun()
                        except (OSError, ValueError) as error:
                            st.error(f"Could not calculate terrain context: {error}")
    selected_metrics = profile_metrics(
        selected_profile,
        weak_threshold=weak_threshold,
        outage_threshold=outage_threshold,
    )
    hotspots = find_hotspots(
        selected_profile,
        weak_threshold=weak_threshold,
        minimum_length_m=hotspot_length_km * 1000,
    )
    recommendations = recommend_tower_sites(
        selected_profile,
        weak_threshold=weak_threshold,
        minimum_length_m=hotspot_length_km * 1000,
    )
    st.markdown('<div class="cr-section-title">Coverage intelligence</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="cr-section-note">Turn the signal profile into decisions: where coverage is weak, '
        'how long the risk lasts, and where a planning intervention could help.</p>',
        unsafe_allow_html=True,
    )
    coverage_tab, hotspot_tab, planning_tab, data_tab = st.tabs(
        ["Coverage summary", "Weak-signal hotspots", "Network planning", "Route data"]
    )
    with coverage_tab:
        coverage_columns = st.columns(5)
        coverage_columns[0].metric("Usable coverage", f"{selected_metrics['usable_fraction']:.1%}")
        coverage_columns[1].metric("Weak coverage", f"{selected_metrics['weak_fraction']:.1%}")
        coverage_columns[2].metric("Estimated outage", f"{selected_metrics['outage_fraction']:.1%}")
        coverage_columns[3].metric(
            "Longest weak stretch", f"{selected_metrics['longest_weak_stretch_m'] / 1000:.1f} km"
        )
        coverage_columns[4].metric("Minimum RSRP", f"{selected_metrics['min_rsrp_dbm']:.1f} dBm")
        chart_profile = selected_profile[
            ["distance_along_route_m", "predicted_rsrp_dbm"]
        ].copy()
        chart_profile["weak threshold"] = weak_threshold
        chart_profile["outage threshold"] = outage_threshold
        st.line_chart(
            chart_profile,
            x="distance_along_route_m",
            y=["predicted_rsrp_dbm", "weak threshold", "outage threshold"],
            x_label="Distance along route (m)",
            y_label="RSRP (dBm)",
            height=300,
        )
    with hotspot_tab:
        if hotspots.empty:
            st.success("No weak-signal segment meets the selected minimum hotspot length.")
        else:
            st.dataframe(
                hotspots.drop(columns=["Latitude", "Longitude"]).round(2),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "Hotspots group consecutive samples below the weak threshold. "
                "They are screening indicators, not measured coverage gaps."
            )
    with planning_tab:
        st.warning(
            "Planning suggestions are heuristic candidate locations. They are not a "
            "radio-engineering design, propagation guarantee, or land-suitability assessment."
        )
        if recommendations.empty:
            st.success("No candidate tower intervention is suggested for this threshold.")
        else:
            st.dataframe(
                recommendations.drop(columns=["Latitude", "Longitude"]).round(2),
                hide_index=True,
                width="stretch",
            )
            st.map(
                recommendations.rename(
                    columns={"Latitude": "latitude", "Longitude": "longitude"}
                )[["latitude", "longitude"]],
                latitude="latitude",
                longitude="longitude",
                size=80,
                color="#2563eb",
            )
    with data_tab:
        export_columns = [
            "distance_along_route_m",
            "latitude",
            "longitude",
            "predicted_rsrp_dbm",
            "estimated_lower_90_dbm",
            "estimated_upper_90_dbm",
            "nearest_tower_distance_m",
            "nearest_tower_samples",
            "signal_band",
            "below_weak_threshold",
            "below_outage_threshold",
        ]
        st.dataframe(selected_profile[export_columns].round(3), hide_index=True, width="stretch")
        st.download_button(
            "Download selected route profile (CSV)",
            data=selected_profile[export_columns].to_csv(index=False),
            file_name=f"connectroute_route_{selected_route_index + 1}_profile.csv",
            mime="text/csv",
            icon=":material/download:",
        )
    with st.expander("How this estimate is produced"):
        st.write(
            "The route is sampled at the selected spacing. Each sample is matched "
            "to its nearest eligible local LTE record; a distance-only regression "
            "fitted to Salzburg measurements estimates RSRP. Elevation, terrain "
            "diffraction, roughness, and a local low-ground indicator are calculated "
            "from Copernicus GLO-30, but are not used to modify signal. OpenStreetMap "
            "and OSRM provide geocoding and driving routes."
        )
        st.caption(
            f"Using {st.session_state['signal_summary']['eligible_india_towers']:,} "
            "eligible LTE tower records. "
            + (
                "Terrain source: Copernicus DEM GLO-30 Public (DSM)."
                if has_terrain
                else "Terrain profile is not loaded."
            )
        )
    st.download_button(
        "Download route alternatives (GeoJSON)",
        data=json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "route_index": index,
                            "distance_m": route["distance"],
                            "duration_s": route["duration"],
                            "mean_estimated_rsrp_dbm": float(
                                signal_profiles[index - 1]["predicted_rsrp_dbm"].mean()
                            ),
                            "median_terrain_diffraction_loss_db": (
                                float(
                                    signal_profiles[index - 1][
                                        "terrain_diffraction_loss_db"
                                    ].median()
                                )
                                if has_terrain
                                else None
                            ),
                        },
                        "geometry": route["geometry"],
                    }
                    for index, route in enumerate(routes, start=1)
                ],
            },
            indent=2,
        ),
        file_name="connectroute_alternatives.geojson",
        mime="application/geo+json",
    )
