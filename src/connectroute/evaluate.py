"""Generate spatially grouped S8 metrics, figures, and paired block-bootstrap CIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"
DEFAULT_FEATURES = PROJECT_ROOT / "data" / "processed" / "salzburg_propagation_features.parquet"
DEFAULT_PREDICTIONS = PROJECT_ROOT / "data" / "processed" / "spatial_model_predictions.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "evaluation"
DEFAULT_MODEL_SUMMARY = PROJECT_ROOT / "results" / "model_summary.json"
MODEL_COLUMNS = {
    "Fitted log-distance physics baseline": "physics_baseline_rsrp_dbm",
    "Pure LightGBM": "pure_lightgbm_rsrp_dbm",
    "LightGBM hybrid residual": "hybrid_lightgbm_rsrp_dbm",
    "XGBoost hybrid residual": "hybrid_xgboost_rsrp_dbm",
    "LightGBM quantile median": "quantile_50_rsrp_dbm",
}
PROPAGATION_BENCHMARK_COLUMNS = {
    "Free space": ("predicted_free_space_rsrp_dbm", None),
    "COST-231 Hata": (
        "predicted_cost231_hata_rsrp_dbm",
        "best_server_cost231_hata_in_domain",
    ),
    "Rural Hata": (
        "predicted_hata_rural_rsrp_dbm",
        "best_server_hata_rural_in_domain",
    ),
    "3GPP RMa LOS": (
        "predicted_3gpp_rma_los_rsrp_dbm",
        "best_server_3gpp_rma_los_in_domain",
    ),
    "3GPP RMa NLOS": (
        "predicted_3gpp_rma_nlos_rsrp_dbm",
        "best_server_3gpp_rma_nlos_in_domain",
    ),
    "3GPP UMa LOS": (
        "predicted_3gpp_uma_los_rsrp_dbm",
        "best_server_3gpp_uma_los_in_domain",
    ),
    "3GPP UMa NLOS": (
        "predicted_3gpp_uma_nlos_rsrp_dbm",
        "best_server_3gpp_uma_nlos_in_domain",
    ),
    "Fitted log distance": (
        "predicted_fitted_log_distance_rsrp_dbm",
        None,
    ),
}
ML_BENCHMARK_COLUMNS = {
    "Pure LightGBM": "pure_lightgbm_rsrp_dbm",
    "LightGBM hybrid residual": "hybrid_lightgbm_rsrp_dbm",
    "XGBoost hybrid residual": "hybrid_xgboost_rsrp_dbm",
    "LightGBM quantile median": "quantile_50_rsrp_dbm",
}


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int | None]:
    valid = np.isfinite(actual) & np.isfinite(predicted)
    if not valid.any():
        return {
            "n": 0,
            "rmse_db": None,
            "mae_db": None,
            "bias_db": None,
            "r2": None,
        }
    y = actual[valid]
    estimate = predicted[valid]
    error = estimate - y
    total = np.sum((y - y.mean()) ** 2)
    return {
        "n": int(len(y)),
        "rmse_db": float(np.sqrt(np.mean(error**2))),
        "mae_db": float(np.mean(np.abs(error))),
        "bias_db": float(np.mean(error)),
        "r2": float(1 - np.sum(error**2) / total) if total > 0 else None,
    }


def build_benchmark_tables(
    propagation_predictions: pd.DataFrame,
    ml_predictions: pd.DataFrame,
    propagation_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize saved spatial out-of-fold predictions for the Streamlit model lab."""
    for name, frame in (
        ("ML predictions", ml_predictions),
        ("Propagation features", propagation_features),
    ):
        if len(frame) != len(propagation_predictions):
            raise ValueError(f"{name} and propagation predictions must have equal row counts.")
        if not np.array_equal(
            propagation_predictions["rsrp_dbm"].to_numpy(dtype=float),
            frame["rsrp_dbm"].to_numpy(dtype=float),
            equal_nan=True,
        ):
            raise ValueError(f"{name} targets are not row-aligned with propagation predictions.")

    if "fold" in ml_predictions and not np.array_equal(
        propagation_predictions["fold"].to_numpy(),
        ml_predictions["fold"].to_numpy(),
    ):
        raise ValueError("ML and propagation predictions use different spatial folds.")

    actual = propagation_predictions["rsrp_dbm"].to_numpy(dtype=float)
    propagation_rows: list[dict[str, Any]] = []
    for label, (column, domain_column) in PROPAGATION_BENCHMARK_COLUMNS.items():
        metrics = _metrics(actual, propagation_predictions[column].to_numpy(dtype=float))
        in_domain_fraction = (
            float(propagation_features[domain_column].astype(bool).mean())
            if domain_column
            else None
        )
        propagation_rows.append(
            {
                "Model": label,
                "RMSE (dB)": metrics["rmse_db"],
                "MAE (dB)": metrics["mae_db"],
                "Bias (dB)": metrics["bias_db"],
                "Best-server in domain (%)": (
                    100 * in_domain_fraction if in_domain_fraction is not None else None
                ),
            }
        )

    ml_rows = [
        {"Model": label, **_metrics(actual, ml_predictions[column].to_numpy(dtype=float))}
        for label, column in ML_BENCHMARK_COLUMNS.items()
    ]
    ml_metrics = pd.DataFrame(ml_rows).rename(
        columns={"rmse_db": "RMSE (dB)", "mae_db": "MAE (dB)", "bias_db": "Bias (dB)"}
    )
    return (
        pd.DataFrame(propagation_rows).sort_values("RMSE (dB)").reset_index(drop=True),
        ml_metrics.sort_values("RMSE (dB)").reset_index(drop=True),
    )


def _paired_block_bootstrap(
    actual: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    block_ids: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    valid = np.isfinite(actual) & np.isfinite(baseline) & np.isfinite(candidate)
    if not valid.any():
        raise ValueError("No paired finite predictions are available for bootstrap.")
    unique_blocks = np.unique(block_ids[valid])
    if len(unique_blocks) < 2:
        raise ValueError("At least two valid spatial blocks are required for bootstrap.")
    if replicates < 1:
        raise ValueError("Bootstrap replicates must be positive.")
    block_indices = {
        block: np.flatnonzero(
            (block_ids == block)
            & np.isfinite(actual)
            & np.isfinite(baseline)
            & np.isfinite(candidate)
        )
        for block in unique_blocks
    }
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=float)
    for iteration in range(replicates):
        sampled_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        sampled_indices = np.concatenate([block_indices[block] for block in sampled_blocks])
        y = actual[sampled_indices]
        base_error = baseline[sampled_indices] - y
        candidate_error = candidate[sampled_indices] - y
        deltas[iteration] = np.sqrt(np.mean(candidate_error**2)) - np.sqrt(
            np.mean(base_error**2)
        )
    return {
        "spatial_blocks": int(len(unique_blocks)),
        "bootstrap_replicates": int(replicates),
        "seed": int(seed),
        "rmse_delta_candidate_minus_baseline_db": float(
            np.sqrt(np.mean((candidate[valid] - actual[valid]) ** 2))
            - np.sqrt(np.mean((baseline[valid] - actual[valid]) ** 2))
        ),
        "ci_95_lower_db": float(np.quantile(deltas, 0.025)),
        "ci_95_upper_db": float(np.quantile(deltas, 0.975)),
        "bootstrap_median_delta_db": float(np.median(deltas)),
        "fraction_candidate_better": float(np.mean(deltas < 0)),
    }


def _outage_report(
    actual_rsrp: np.ndarray,
    probability: np.ndarray,
    threshold_dbm: float,
) -> dict[str, float | int | None]:
    valid = np.isfinite(actual_rsrp) & np.isfinite(probability)
    y = (actual_rsrp[valid] < threshold_dbm).astype(np.int8)
    p = np.clip(probability[valid], 0, 1)
    predicted = p >= 0.5
    precision, recall, f1, _ = precision_recall_fscore_support(
        y,
        predicted,
        average="binary",
        zero_division=0,
    )
    return {
        "threshold_dbm": float(threshold_dbm),
        "n": int(len(y)),
        "prevalence": float(y.mean()) if len(y) else None,
        "precision_at_0_5": float(precision),
        "recall_at_0_5": float(recall),
        "f1_at_0_5": float(f1),
        "pr_auc_average_precision": (
            float(average_precision_score(y, p)) if y.sum() else None
        ),
        "brier_score": float(np.mean((p - y) ** 2)) if len(y) else None,
    }


def _reliability_bins(
    actual_rsrp: np.ndarray,
    probability: np.ndarray,
    threshold_dbm: float,
    bins: int = 10,
) -> pd.DataFrame:
    valid = np.isfinite(actual_rsrp) & np.isfinite(probability)
    data = pd.DataFrame(
        {
            "probability": np.clip(probability[valid], 0, 1),
            "observed_outage": (actual_rsrp[valid] < threshold_dbm).astype(float),
        }
    )
    data["probability_bin"] = pd.cut(
        data["probability"],
        bins=np.linspace(0, 1, bins + 1),
        include_lowest=True,
    )
    return (
        data.groupby("probability_bin", observed=False)
        .agg(
            count=("probability", "size"),
            mean_predicted_probability=("probability", "mean"),
            observed_outage_fraction=("observed_outage", "mean"),
        )
        .reset_index()
    )


def _plot_regression(metrics: pd.DataFrame, output_path: Path) -> None:
    ordered = metrics.sort_values("rmse_db")
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.barh(ordered["model"], ordered["rmse_db"], color="#3478a8")
    axis.set_xlabel("Held-out RMSE (dB)")
    axis.set_title("Spatially blocked Salzburg model comparison")
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_reliability(
    actual: np.ndarray,
    probabilities: dict[str, np.ndarray],
    thresholds: dict[str, float],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", label="Ideal")
    for name, threshold in thresholds.items():
        table = _reliability_bins(actual, probabilities[name], threshold)
        table = table[table["count"] > 0]
        axis.plot(
            table["mean_predicted_probability"],
            table["observed_outage_fraction"],
            marker="o",
            label=f"{name} ({threshold:g} dBm)",
        )
    axis.set(
        xlabel="Mean predicted outage probability",
        ylabel="Observed outage fraction",
        title="Out-of-fold outage reliability",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for values in frame.itertuples(index=False, name=None):
        cells = []
        for value in values:
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def evaluate_results(
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    model_summary: dict[str, Any],
    *,
    output_directory: Path,
    bootstrap_replicates: int = 2000,
) -> dict[str, Any]:
    if len(features) != len(predictions):
        raise ValueError("Feature and prediction tables must have equal row counts.")
    if not np.array_equal(
        pd.to_numeric(features["rsrp_dbm"], errors="coerce").to_numpy(dtype=float),
        predictions["rsrp_dbm"].to_numpy(dtype=float),
        equal_nan=True,
    ):
        raise ValueError("Feature and prediction targets are not row-aligned.")
    output_directory.mkdir(parents=True, exist_ok=True)

    actual = predictions["rsrp_dbm"].to_numpy(dtype=float)
    chainage = pd.to_numeric(features["chainage_m"], errors="raise").to_numpy(dtype=float)
    spatial_block = np.floor(chainage / 2000.0).astype(np.int64)
    point_metrics = pd.DataFrame(
        [
            {"model": label, **_metrics(actual, predictions[column].to_numpy(dtype=float))}
            for label, column in MODEL_COLUMNS.items()
        ]
    )
    point_metrics.to_csv(output_directory / "point_metrics.csv", index=False)

    binned = pd.DataFrame(
        {
            "chainage_bin_50m": np.floor(chainage / 50.0).astype(np.int64),
            "fold": predictions["fold"].to_numpy(dtype=np.int64),
            "actual": actual,
        }
    )
    for label, column in MODEL_COLUMNS.items():
        binned[label] = predictions[column].to_numpy(dtype=float)
    bin_medians = (
        binned.groupby(["fold", "chainage_bin_50m"], sort=True)
        .median(numeric_only=True)
        .reset_index()
    )
    binned_metrics = pd.DataFrame(
        [
            {"model": label, **_metrics(bin_medians["actual"].to_numpy(), bin_medians[label].to_numpy())}
            for label in MODEL_COLUMNS
        ]
    )
    binned_metrics.to_csv(output_directory / "chainage_50m_bin_metrics.csv", index=False)

    fold_metrics: list[dict[str, Any]] = []
    for fold in sorted(predictions["fold"].unique()):
        mask = predictions["fold"].to_numpy() == fold
        for label, column in MODEL_COLUMNS.items():
            fold_metrics.append(
                {
                    "fold": int(fold),
                    "model": label,
                    **_metrics(actual[mask], predictions[column].to_numpy(dtype=float)[mask]),
                }
            )
    fold_metric_frame = pd.DataFrame(fold_metrics)
    fold_metric_frame.to_csv(output_directory / "fold_metrics.csv", index=False)

    physics_column = MODEL_COLUMNS["Fitted log-distance physics baseline"]
    bootstrap: dict[str, Any] = {}
    for index, (label, column) in enumerate(MODEL_COLUMNS.items()):
        if label == "Fitted log-distance physics baseline":
            continue
        bootstrap[label] = _paired_block_bootstrap(
            actual,
            predictions[physics_column].to_numpy(dtype=float),
            predictions[column].to_numpy(dtype=float),
            spatial_block,
            replicates=bootstrap_replicates,
            seed=int(model_summary["seed"]) + index,
        )
    pd.DataFrame(
        [{"model": name, **values} for name, values in bootstrap.items()]
    ).to_csv(output_directory / "paired_block_bootstrap.csv", index=False)

    uncertainty = model_summary["uncertainty"]
    thresholds = model_summary["outage_thresholds_dbm"]
    outage_reports: dict[str, Any] = {}
    reliability_tables: list[pd.DataFrame] = []
    for threshold_name, probability_column in (
        ("primary", "outage_probability_below_primary_threshold"),
        ("secondary", "outage_probability_below_secondary_threshold"),
    ):
        threshold = float(thresholds[threshold_name])
        probability = predictions[probability_column].to_numpy(dtype=float)
        outage_reports[threshold_name] = _outage_report(actual, probability, threshold)
        reliability = _reliability_bins(actual, probability, threshold)
        reliability.insert(0, "threshold_name", threshold_name)
        reliability.insert(1, "threshold_dbm", threshold)
        reliability_tables.append(reliability)
    reliability_frame = pd.concat(reliability_tables, ignore_index=True)
    reliability_frame.to_csv(output_directory / "outage_reliability_bins.csv", index=False)

    _plot_regression(point_metrics, output_directory / "spatial_model_rmse.png")
    _plot_reliability(
        actual,
        {
            "Primary": predictions[
                "outage_probability_below_primary_threshold"
            ].to_numpy(dtype=float),
            "Secondary": predictions[
                "outage_probability_below_secondary_threshold"
            ].to_numpy(dtype=float),
        },
        {
            "Primary": float(thresholds["primary"]),
            "Secondary": float(thresholds["secondary"]),
        },
        output_directory / "outage_reliability.png",
    )

    summary = {
        "protocol": (
            "All prediction metrics use out-of-fold predictions from the contiguous "
            "spatial CV. Chainage bins are 50 m medians. Paired bootstrap resamples "
            "2 km spatial blocks, not individual observations."
        ),
        "rows": int(len(predictions)),
        "seed": int(model_summary["seed"]),
        "bootstrap_replicates": int(bootstrap_replicates),
        "point_metrics": {
            row["model"]: {
                key: (None if pd.isna(value) else value)
                for key, value in row.items()
                if key != "model"
            }
            for row in point_metrics.to_dict(orient="records")
        },
        "chainage_50m_bin_metrics": {
            row["model"]: {
                key: (None if pd.isna(value) else value)
                for key, value in row.items()
                if key != "model"
            }
            for row in binned_metrics.to_dict(orient="records")
        },
        "paired_block_bootstrap_rmse_difference_vs_fitted_log_distance": bootstrap,
        "outage": outage_reports,
        "intervals": uncertainty["intervals"],
        "interval_coverage_per_fold": uncertainty["per_fold_coverage"],
    }
    with (output_directory / "evaluation_summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2, allow_nan=False)
        file.write("\n")

    comparison = point_metrics[["model", "rmse_db", "mae_db", "bias_db", "r2"]].copy()
    report_lines = [
        "# ConnectRoute spatial evaluation",
        "",
        "All reported predictions are out-of-fold under contiguous spatial folds. "
        "This report does not imply external geographic validation.",
        "",
        "## Point-level regression metrics",
        "",
        _markdown_table(comparison),
        "",
        "## 50 m chainage-bin median metrics",
        "",
        _markdown_table(
            binned_metrics[["model", "n", "rmse_db", "mae_db", "bias_db", "r2"]]
        ),
        "",
        "## Paired 2 km block-bootstrap RMSE differences",
        "",
        "Negative candidate-minus-baseline differences favor the candidate. "
        "Confidence intervals are over spatial blocks, not independent points.",
        "",
        _markdown_table(
            pd.DataFrame(
                [
                    {
                        "model": name,
                        "delta_RMSE_dB": result["rmse_delta_candidate_minus_baseline_db"],
                        "CI95_lower": result["ci_95_lower_db"],
                        "CI95_upper": result["ci_95_upper_db"],
                    }
                    for name, result in bootstrap.items()
                ]
            )
        ),
        "",
        "## Outage and uncertainty",
        "",
        f"Outage reports: `{(output_directory / 'evaluation_summary.json').name}`. "
        "Coverage varies by spatial fold; aggregate conformal coverage is not "
        "evidence of reliable coverage under route-level distribution shift.",
        "",
        "## Limitations and remaining protocol work",
        "",
        "- This evaluates the two-trial development models unless the model pipeline "
        "has been rerun with the configured full search.",
        "- Ablations, the 20-seed tower-position/removal study, SHAP explanations, "
        "and application validation are separate studies and are not inferred from "
        "these metrics.",
        "- The only measured validation corridor is Salzburg; any Karnataka map is "
        "illustrative and unvalidated.",
        "",
    ]
    (output_directory / "evaluation_report.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--model-summary", type=Path, default=DEFAULT_MODEL_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    arguments = parser.parse_args()
    with arguments.config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    features = pd.read_parquet(arguments.features)
    predictions = pd.read_parquet(arguments.predictions)
    model_summary = json.loads(arguments.model_summary.read_text(encoding="utf-8"))
    summary = evaluate_results(
        features,
        predictions,
        model_summary,
        output_directory=arguments.output,
        bootstrap_replicates=arguments.bootstrap_replicates,
    )
    summary["configuration"] = config
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
