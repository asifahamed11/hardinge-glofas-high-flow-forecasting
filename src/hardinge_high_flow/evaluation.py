"""Rare-event calibration, metrics, and uncertainty."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class SigmoidCalibrator:
    coefficient: float
    intercept: float

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
        logits = np.log(clipped / (1 - clipped))
        calibrated_logits = self.coefficient * logits + self.intercept
        return 1 / (1 + np.exp(-calibrated_logits))


def fit_sigmoid_calibrator(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> SigmoidCalibrator:
    labels = np.asarray(labels, dtype=int)
    if np.unique(labels).size != 2:
        raise ValueError("Calibration labels must contain both classes.")
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(solver="lbfgs", max_iter=1_000)
    model.fit(logits, labels)
    return SigmoidCalibrator(
        coefficient=float(model.coef_[0, 0]),
        intercept=float(model.intercept_[0]),
    )


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    assignments = np.clip(np.digitize(probabilities, edges) - 1, 0, bins - 1)
    error = 0.0
    for bin_index in range(bins):
        mask = assignments == bin_index
        if not mask.any():
            continue
        error += mask.mean() * abs(probabilities[mask].mean() - labels[mask].mean())
    return float(error)


def reliability_points(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> pd.DataFrame:
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    assignments = np.clip(np.digitize(probabilities, edges) - 1, 0, bins - 1)
    rows = []
    for bin_index in range(bins):
        mask = assignments == bin_index
        if not mask.any():
            continue
        rows.append(
            {
                "bin": bin_index,
                "mean_probability": float(probabilities[mask].mean()),
                "observed_fraction": float(labels[mask].mean()),
                "count": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def select_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    metric: str = "f1",
) -> tuple[float, float]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    scores = []
    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(int)
        if metric == "f1":
            score = f1_score(labels, predictions, zero_division=0)
        elif metric == "csi":
            true_negative, false_positive, false_negative, true_positive = (
                confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
            )
            denominator = true_positive + false_positive + false_negative
            score = true_positive / denominator if denominator else 0.0
        else:
            raise ValueError(f"Unsupported threshold metric: {metric}")
        scores.append(float(score))
    best_score = max(scores)
    best_thresholds = candidates[np.isclose(scores, best_score)]
    return float(best_thresholds.max()), float(best_score)


def binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    reliability_bins: int = 10,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    has_both_classes = np.unique(labels).size == 2
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()
    specificity_denominator = true_negative + false_positive
    csi_denominator = true_positive + false_positive + false_negative
    warning_denominator = true_positive + false_positive
    roc_auc = (
        float(roc_auc_score(labels, probabilities))
        if np.unique(labels).size == 2
        else float("nan")
    )
    average_precision = (
        float(average_precision_score(labels, probabilities))
        if np.unique(labels).size == 2
        else float("nan")
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(labels, predictions))
            if has_both_classes
            else float("nan")
        ),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": (
            float(true_negative / specificity_denominator)
            if specificity_denominator
            else float("nan")
        ),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "average_precision": average_precision,
        "roc_auc": roc_auc,
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "ece": expected_calibration_error(
            labels,
            probabilities,
            bins=reliability_bins,
        ),
        "mcc": (
            float(matthews_corrcoef(labels, predictions))
            if has_both_classes
            else float("nan")
        ),
        "critical_success_index": (
            float(true_positive / csi_denominator) if csi_denominator else 0.0
        ),
        "false_alarm_ratio": (
            float(false_positive / warning_denominator) if warning_denominator else 0.0
        ),
        "true_negative": float(true_negative),
        "false_positive": float(false_positive),
        "false_negative": float(false_negative),
        "true_positive": float(true_positive),
        "threshold": float(threshold),
        "prevalence": float(labels.mean()),
    }


def _event_spans(
    positives: np.ndarray,
    dates: pd.DatetimeIndex,
    allowed_gap_days: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    positive_dates = dates[np.asarray(positives, dtype=bool)]
    if positive_dates.empty:
        return []
    spans = []
    start = positive_dates[0]
    previous = positive_dates[0]
    for date in positive_dates[1:]:
        if (date - previous).days > allowed_gap_days + 1:
            spans.append((start, previous))
            start = date
        previous = date
    spans.append((start, previous))
    return spans


def event_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    dates: pd.DatetimeIndex,
    allowed_gap_days: int = 1,
) -> dict[str, float]:
    predictions = np.asarray(probabilities) >= threshold
    actual_events = _event_spans(labels, dates, allowed_gap_days)
    predicted_events = _event_spans(predictions, dates, allowed_gap_days)
    detected = 0
    onset_errors = []
    for actual_start, actual_end in actual_events:
        matching_dates = dates[
            predictions & (dates >= actual_start) & (dates <= actual_end)
        ]
        if not matching_dates.empty:
            detected += 1
            onset_errors.append((matching_dates[0] - actual_start).days)

    false_alarm_events = 0
    positive_mask = np.asarray(labels, dtype=bool)
    for predicted_start, predicted_end in predicted_events:
        overlap = positive_mask[(dates >= predicted_start) & (dates <= predicted_end)]
        if not overlap.any():
            false_alarm_events += 1

    return {
        "actual_events": float(len(actual_events)),
        "detected_events": float(detected),
        "missed_events": float(len(actual_events) - detected),
        "event_detection_rate": (
            detected / len(actual_events) if actual_events else float("nan")
        ),
        "predicted_events": float(len(predicted_events)),
        "false_alarm_events": float(false_alarm_events),
        "event_false_alarm_ratio": (
            false_alarm_events / len(predicted_events) if predicted_events else 0.0
        ),
        "median_onset_delay_days": (
            float(np.median(onset_errors)) if onset_errors else float("nan")
        ),
    }


def event_catalog(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    target_dates: pd.DatetimeIndex,
    issue_dates: pd.DatetimeIndex,
    target_values: np.ndarray,
    allowed_gap_days: int = 1,
) -> pd.DataFrame:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    target_values = np.asarray(target_values, dtype=float)
    if not (
        len(labels)
        == len(probabilities)
        == len(target_dates)
        == len(issue_dates)
        == len(target_values)
    ):
        raise ValueError("Event-catalog inputs must have equal lengths.")

    predictions = probabilities >= threshold
    rows = []
    for event_number, (event_start, event_end) in enumerate(
        _event_spans(labels, target_dates, allowed_gap_days),
        start=1,
    ):
        event_mask = (target_dates >= event_start) & (target_dates <= event_end)
        event_positions = np.flatnonzero(event_mask)
        warning_positions = event_positions[predictions[event_positions]]
        peak_position = event_positions[
            int(np.nanargmax(target_values[event_positions]))
        ]
        detected = len(warning_positions) > 0
        first_position = int(warning_positions[0]) if detected else None
        first_target_date = (
            target_dates[first_position] if first_position is not None else pd.NaT
        )
        first_issue_date = (
            issue_dates[first_position] if first_position is not None else pd.NaT
        )
        rows.append(
            {
                "event_number": event_number,
                "event_start": event_start,
                "event_end": event_end,
                "duration_days": int((event_end - event_start).days + 1),
                "peak_date": target_dates[peak_position],
                "peak_target_value": float(target_values[peak_position]),
                "peak_probability": float(probabilities[peak_position]),
                "peak_detected": bool(predictions[peak_position]),
                "detected": detected,
                "first_warning_target_date": first_target_date,
                "first_warning_issue_date": first_issue_date,
                "onset_delay_target_days": (
                    int((first_target_date - event_start).days) if detected else np.nan
                ),
                "warning_lead_to_onset_days": (
                    int((event_start - first_issue_date).days) if detected else np.nan
                ),
                "warning_days_within_event": int(len(warning_positions)),
            }
        )
    return pd.DataFrame(rows)


def cost_loss_analysis(
    labels: np.ndarray,
    probabilities: np.ndarray,
    ratios: list[float],
) -> pd.DataFrame:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    rows = []
    for ratio in ratios:
        if not 0 < ratio < 1:
            raise ValueError("Cost-loss ratios must be between zero and one.")
        decisions = probabilities >= ratio
        hits = int(np.sum(decisions & (labels == 1)))
        false_alarms = int(np.sum(decisions & (labels == 0)))
        misses = int(np.sum(~decisions & (labels == 1)))
        forecast_expense = ratio * (hits + false_alarms) + misses
        climatology_expense = min(ratio * len(labels), int(labels.sum()))
        perfect_expense = ratio * int(labels.sum())
        denominator = climatology_expense - perfect_expense
        relative_value = (
            (climatology_expense - forecast_expense) / denominator
            if denominator > 0
            else float("nan")
        )
        rows.append(
            {
                "cost_loss_ratio": float(ratio),
                "decision_threshold": float(ratio),
                "forecast_expense": float(forecast_expense),
                "climatology_expense": float(climatology_expense),
                "perfect_expense": float(perfect_expense),
                "relative_economic_value": float(relative_value),
            }
        )
    return pd.DataFrame(rows)


def block_bootstrap_intervals(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    dates: pd.DatetimeIndex,
    iterations: int,
    confidence_level: float,
    seed: int,
    reliability_bins: int,
) -> dict[str, float]:
    years = np.asarray(dates.year)
    unique_years = np.unique(years)
    if len(unique_years) < 2:
        raise ValueError("At least two years are required for block bootstrap.")
    rng = np.random.default_rng(seed)
    collected: dict[str, list[float]] = {}
    for _ in range(iterations):
        sampled_years = rng.choice(
            unique_years,
            size=len(unique_years),
            replace=True,
        )
        sample_indices = np.concatenate(
            [np.flatnonzero(years == year) for year in sampled_years]
        )
        sample_metrics = binary_metrics(
            labels[sample_indices],
            probabilities[sample_indices],
            threshold,
            reliability_bins,
        )
        for name, value in sample_metrics.items():
            if name in {
                "true_negative",
                "false_positive",
                "false_negative",
                "true_positive",
                "threshold",
            }:
                continue
            if np.isfinite(value):
                collected.setdefault(name, []).append(value)

    alpha = 1 - confidence_level
    intervals = {}
    for name, values in collected.items():
        if not values:
            continue
        intervals[f"{name}_ci_low"] = float(np.quantile(values, alpha / 2))
        intervals[f"{name}_ci_high"] = float(np.quantile(values, 1 - alpha / 2))
    return intervals


def paired_block_bootstrap_difference(
    labels: np.ndarray,
    first_probabilities: np.ndarray,
    second_probabilities: np.ndarray,
    first_threshold: float,
    second_threshold: float,
    dates: pd.DatetimeIndex,
    metric: str,
    iterations: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float]:
    scorers: dict[str, Callable[[np.ndarray, np.ndarray, float], float]] = {
        "f1": lambda y, p, threshold: f1_score(
            y,
            p >= threshold,
            zero_division=0,
        ),
        "average_precision": lambda y, p, threshold: average_precision_score(y, p),
        "brier_score": lambda y, p, threshold: brier_score_loss(y, p),
    }
    if metric not in scorers:
        raise ValueError(f"Unsupported paired metric: {metric}")
    if not (
        len(labels)
        == len(first_probabilities)
        == len(second_probabilities)
        == len(dates)
    ):
        raise ValueError("Paired bootstrap inputs must have equal lengths.")
    scorer = scorers[metric]
    years = np.asarray(dates.year)
    unique_years = np.unique(years)
    if len(unique_years) < 2:
        raise ValueError("At least two years are required for paired bootstrap.")
    rng = np.random.default_rng(seed)
    differences = []
    attempts = 0
    maximum_attempts = iterations * 50
    while len(differences) < iterations and attempts < maximum_attempts:
        attempts += 1
        sampled_years = rng.choice(
            unique_years,
            size=len(unique_years),
            replace=True,
        )
        indices = np.concatenate(
            [np.flatnonzero(years == year) for year in sampled_years]
        )
        # Average precision and rare-event F1 comparisons are undefined or
        # uninformative
        # when a resample contains no positive event.  Record the number of
        # attempted draws rather than silently treating such draws as zero.
        if np.unique(labels[indices]).size < 2:
            continue
        first = scorer(
            labels[indices],
            first_probabilities[indices],
            first_threshold,
        )
        second = scorer(
            labels[indices],
            second_probabilities[indices],
            second_threshold,
        )
        differences.append(float(first - second))

    if len(differences) < iterations:
        raise ValueError(
            "Could not draw enough two-class temporal bootstrap samples."
        )

    alpha = 1 - confidence_level
    lower = float(np.quantile(differences, alpha / 2))
    upper = float(np.quantile(differences, 1 - alpha / 2))
    return {
        "metric": metric,
        "mean_difference": float(np.mean(differences)),
        "ci_low": lower,
        "ci_high": upper,
        "probability_first_better": float(np.mean(np.asarray(differences) > 0)),
        "bootstrap_iterations_valid": int(len(differences)),
        "bootstrap_iterations_attempted": int(attempts),
    }
