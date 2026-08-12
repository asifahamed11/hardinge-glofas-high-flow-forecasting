"""Canonical tables and diagnostics derived from saved experiment predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .evaluation import (
    binary_metrics,
    calibration_slope_intercept,
    cost_loss_analysis,
    event_catalog,
    event_metrics,
)


def canonicalize_metric_names(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace the historical ``pr_auc`` label with ``average_precision``.

    Earlier releases correctly computed scikit-learn average precision but
    labelled the result as PR-AUC.  This migration helper preserves the values
    while assigning the scientifically correct name to columns and paired-test
    metric identifiers.
    """

    renamed = {
        column: column.replace("pr_auc", "average_precision")
        for column in frame.columns
        if "pr_auc" in column
    }
    collisions = {
        target
        for source, target in renamed.items()
        if source != target and target in frame.columns
    }
    if collisions:
        raise ValueError(
            "Cannot canonicalize metric names because columns already exist: "
            f"{sorted(collisions)}"
        )
    result = frame.rename(columns=renamed).copy()
    if "metric" in result.columns:
        result["metric"] = result["metric"].replace(
            {"pr_auc": "average_precision"}
        )
    return result


def constant_decision_threshold(group: pd.DataFrame) -> float:
    """Return a run's stored threshold without numerically averaging it."""

    values = group["threshold"].dropna().to_numpy(dtype=float)
    if not len(values):
        raise ValueError("A prediction group has no decision threshold.")
    threshold = float(values[0])
    if not np.allclose(values, threshold, rtol=0.0, atol=1e-12):
        raise ValueError("Decision threshold changes within one fitted run.")
    return threshold


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Create the seed-level summary used by the manuscript tables."""

    metrics = canonicalize_metric_names(metrics)
    return (
        metrics.groupby(["model", "horizon_days"], observed=True)
        .agg(
            runs=("seed", "count"),
            average_precision_mean=("average_precision", "mean"),
            average_precision_std=("average_precision", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            recall_mean=("recall", "mean"),
            precision_mean=("precision", "mean"),
            brier_mean=("brier_score", "mean"),
            event_detection_rate_mean=("event_detection_rate", "mean"),
            false_alarm_ratio_mean=("false_alarm_ratio", "mean"),
        )
        .reset_index()
    )


def add_multiple_comparison_control(comparisons: pd.DataFrame) -> pd.DataFrame:
    """Add descriptive two-sided tail areas and Holm-adjusted values."""

    result = canonicalize_metric_names(comparisons)
    if result.empty or "probability_first_better" not in result:
        return result
    probabilities = result["probability_first_better"].to_numpy(dtype=float)
    result["bootstrap_two_sided_tail_area"] = np.minimum(
        1.0,
        2 * np.minimum(probabilities, 1 - probabilities),
    )
    result["holm_adjusted_tail_area"] = np.nan
    grouping = ["horizon_days", "metric", "reference_model"]
    for _, indices in result.groupby(grouping, observed=True).groups.items():
        indices = list(indices)
        raw = result.loc[indices, "bootstrap_two_sided_tail_area"].to_numpy(
            dtype=float
        )
        order = np.argsort(raw)
        adjusted_sorted = np.maximum.accumulate(
            (len(raw) - np.arange(len(raw))) * raw[order]
        )
        adjusted = np.empty(len(raw), dtype=float)
        adjusted[order] = np.minimum(1.0, adjusted_sorted)
        result.loc[indices, "holm_adjusted_tail_area"] = adjusted
    result["interval_excludes_zero"] = (
        (result["ci_low"] > 0) | (result["ci_high"] < 0)
    )
    result["inference_role"] = "exploratory_model_comparison"
    return result


def diagnostic_tables(
    predictions: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rebuild decision, annual, magnitude, and event diagnostics."""

    predictions = predictions.copy()
    predictions["target_date"] = pd.to_datetime(predictions["target_date"])
    predictions["issue_date"] = pd.to_datetime(predictions["issue_date"])
    cost_rows = []
    annual_rows = []
    magnitude_rows = []
    event_frames = []
    eligible_cost_loss_models = set(
        map(str, config["evaluation"].get("cost_loss_models", []))
    )
    grouping = ["model", "horizon_days", "seed"]
    for keys, group in predictions.groupby(grouping, observed=True):
        model, horizon, seed = keys
        group = group.sort_values("target_date")
        threshold = constant_decision_threshold(group)
        if not eligible_cost_loss_models or str(model) in eligible_cost_loss_models:
            costs = cost_loss_analysis(
                group["target_high_flow"].to_numpy(),
                group["probability"].to_numpy(),
                list(map(float, config["evaluation"]["cost_loss_ratios"])),
            )
            costs.insert(0, "probability_status", "probabilistic_or_calibrated")
            costs.insert(0, "seed", int(seed))
            costs.insert(0, "horizon_days", int(horizon))
            costs.insert(0, "model", str(model))
            cost_rows.extend(costs.to_dict(orient="records"))

        events = event_catalog(
            group["target_high_flow"].to_numpy(),
            group["probability"].to_numpy(),
            threshold,
            pd.DatetimeIndex(group["target_date"]),
            pd.DatetimeIndex(group["issue_date"]),
            group["target_value"].to_numpy(),
            int(config["evaluation"]["event_gap_days"]),
        )
        if not events.empty:
            events.insert(0, "seed", int(seed))
            events.insert(0, "horizon_days", int(horizon))
            events.insert(0, "model", str(model))
            event_frames.append(events)

        for year, annual in group.groupby(
            group["target_date"].dt.year,
            observed=True,
        ):
            values = binary_metrics(
                annual["target_high_flow"].to_numpy(),
                annual["probability"].to_numpy(),
                threshold,
                int(config["evaluation"]["reliability_bins"]),
            )
            annual_rows.append(
                {
                    "model": model,
                    "horizon_days": int(horizon),
                    "seed": int(seed),
                    "year": int(year),
                    **values,
                }
            )

        positives = group[group["target_high_flow"] == 1].copy()
        if len(positives) >= 3:
            positives["magnitude_group"] = pd.qcut(
                positives["target_value"].rank(method="first"),
                q=3,
                labels=["lower", "middle", "upper"],
            )
            positives["detected"] = positives["probability"] >= threshold
            for magnitude, magnitude_data in positives.groupby(
                "magnitude_group",
                observed=True,
            ):
                magnitude_rows.append(
                    {
                        "model": model,
                        "horizon_days": int(horizon),
                        "seed": int(seed),
                        "magnitude_group": str(magnitude),
                        "positive_days": int(len(magnitude_data)),
                        "detection_rate": float(magnitude_data["detected"].mean()),
                        "target_value_median": float(
                            magnitude_data["target_value"].median()
                        ),
                    }
                )
    return (
        pd.DataFrame(cost_rows),
        pd.DataFrame(annual_rows),
        pd.DataFrame(magnitude_rows),
        (
            pd.concat(event_frames, ignore_index=True)
            if event_frames
            else pd.DataFrame()
        ),
    )


def event_definition_sensitivity(
    predictions: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Recompute event scores under all declared gap definitions."""

    predictions = predictions.copy()
    predictions["target_date"] = pd.to_datetime(predictions["target_date"])
    gap_days = list(
        map(
            int,
            config["evaluation"].get(
                "event_gap_sensitivity_days",
                [config["evaluation"]["event_gap_days"]],
            ),
        )
    )
    rows = []
    grouping = ["model", "horizon_days", "seed"]
    for keys, group in predictions.groupby(grouping, observed=True):
        model, horizon, seed = keys
        group = group.sort_values("target_date")
        threshold = constant_decision_threshold(group)
        for allowed_gap in gap_days:
            values = event_metrics(
                group["target_high_flow"].to_numpy(),
                group["probability"].to_numpy(),
                threshold,
                pd.DatetimeIndex(group["target_date"]),
                allowed_gap,
            )
            rows.append(
                {
                    "model": str(model),
                    "horizon_days": int(horizon),
                    "seed": int(seed),
                    "allowed_intervening_negative_days": int(allowed_gap),
                    "primary_definition": allowed_gap
                    == int(config["evaluation"]["event_gap_days"]),
                    **values,
                }
            )
    return pd.DataFrame(rows)


def leave_one_year_out_sensitivity(
    predictions: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Report metric stability after omitting each test year in turn."""

    predictions = predictions.copy()
    predictions["target_date"] = pd.to_datetime(predictions["target_date"])
    rows = []
    grouping = ["model", "horizon_days", "seed"]
    for keys, group in predictions.groupby(grouping, observed=True):
        model, horizon, seed = keys
        group = group.sort_values("target_date")
        threshold = constant_decision_threshold(group)
        for omitted_year in sorted(group["target_date"].dt.year.unique()):
            retained = group[group["target_date"].dt.year != omitted_year]
            values = binary_metrics(
                retained["target_high_flow"].to_numpy(),
                retained["probability"].to_numpy(),
                threshold,
                int(config["evaluation"]["reliability_bins"]),
            )
            values.update(
                event_metrics(
                    retained["target_high_flow"].to_numpy(),
                    retained["probability"].to_numpy(),
                    threshold,
                    pd.DatetimeIndex(retained["target_date"]),
                    int(config["evaluation"]["event_gap_days"]),
                )
            )
            rows.append(
                {
                    "model": str(model),
                    "horizon_days": int(horizon),
                    "seed": int(seed),
                    "omitted_year": int(omitted_year),
                    "retained_days": int(len(retained)),
                    "retained_positive_days": int(
                        retained["target_high_flow"].sum()
                    ),
                    **values,
                }
            )
    return pd.DataFrame(rows)


def calibration_diagnostic_table(
    predictions: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Calculate test-period calibration intercepts and slopes per fitted run."""

    predictions = predictions.copy()
    predictions["target_date"] = pd.to_datetime(predictions["target_date"])
    calibrated_models = set(map(str, config["figures"]["calibrated_models"]))
    rows = []
    grouping = ["model", "horizon_days", "seed"]
    for keys, group in predictions.groupby(grouping, observed=True):
        model, horizon, seed = keys
        if str(model) not in calibrated_models:
            continue
        group = group.sort_values("target_date")
        values = calibration_slope_intercept(
            group["target_high_flow"].to_numpy(),
            group["probability"].to_numpy(),
            pd.DatetimeIndex(group["target_date"]),
            float(config["evaluation"]["confidence_level"]),
        )
        rows.append(
            {
                "model": str(model),
                "horizon_days": int(horizon),
                "seed": int(seed),
                "test_days": int(len(group)),
                "test_positive_days": int(group["target_high_flow"].sum()),
                **values,
            }
        )
    return pd.DataFrame(rows)
