from __future__ import annotations

import numpy as np
import pandas as pd

from hardinge_high_flow.evaluation import (
    binary_metrics,
    block_bootstrap_intervals,
    cost_loss_analysis,
    event_catalog,
    event_metrics,
    fit_sigmoid_calibrator,
    paired_block_bootstrap_difference,
    select_threshold,
)


def test_threshold_and_metrics_are_consistent() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.05, 0.1, 0.4, 0.6, 0.8, 0.95])
    threshold, score = select_threshold(labels, probabilities, metric="f1")
    metrics = binary_metrics(labels, probabilities, threshold)
    assert score == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["true_positive"] == 3
    assert metrics["false_positive"] == 0


def test_sigmoid_calibration_stays_bounded() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.2, 0.3, 0.4, 0.6, 0.7, 0.8])
    calibrator = fit_sigmoid_calibrator(probabilities, labels)
    calibrated = calibrator.transform(probabilities)
    assert np.all((calibrated > 0) & (calibrated < 1))
    assert np.all(np.diff(calibrated) > 0)


def test_event_metrics_count_contiguous_events() -> None:
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    labels = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 0])
    probabilities = np.array([0.1, 0.8, 0.9, 0.2, 0.1, 0.7, 0.8, 0.6, 0.2, 0.1])
    metrics = event_metrics(labels, probabilities, 0.5, dates, 0)
    assert metrics["actual_events"] == 2
    assert metrics["detected_events"] == 2
    assert metrics["event_detection_rate"] == 1.0


def test_event_catalog_reports_peak_and_warning_lead() -> None:
    target_dates = pd.date_range("2020-07-01", periods=6, freq="D")
    issue_dates = target_dates - pd.Timedelta(days=2)
    labels = np.array([0, 1, 1, 1, 0, 0])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9, 0.1, 0.1])
    target_values = np.array([10, 12, 15, 14, 9, 8], dtype=float)
    catalog = event_catalog(
        labels,
        probabilities,
        0.5,
        target_dates,
        issue_dates,
        target_values,
        allowed_gap_days=0,
    )
    assert len(catalog) == 1
    event = catalog.iloc[0]
    assert event["peak_date"] == pd.Timestamp("2020-07-03")
    assert bool(event["peak_detected"])
    assert event["onset_delay_target_days"] == 1
    assert event["warning_lead_to_onset_days"] == 1


def test_year_block_bootstrap_returns_intervals() -> None:
    dates = pd.date_range("2018-01-01", "2021-12-31", freq="D")
    labels = (dates.month == 8).astype(int)
    probabilities = np.where(labels == 1, 0.7, 0.1)
    intervals = block_bootstrap_intervals(
        labels,
        probabilities,
        threshold=0.5,
        dates=dates,
        iterations=25,
        confidence_level=0.95,
        seed=42,
        reliability_bins=10,
    )
    assert "f1_ci_low" in intervals
    assert intervals["f1_ci_low"] <= intervals["f1_ci_high"]


def test_cost_loss_analysis_uses_probability_threshold() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.01, 0.2, 0.8, 0.9])
    table = cost_loss_analysis(labels, probabilities, [0.1, 0.3])
    assert list(table["decision_threshold"]) == [0.1, 0.3]
    assert np.isfinite(table["forecast_expense"]).all()


def test_paired_bootstrap_reports_valid_event_bearing_draws() -> None:
    dates = pd.date_range("2019-01-01", "2023-12-31", freq="D")
    labels = (dates.month == 8).astype(int)
    first = np.where(labels == 1, 0.8, 0.1)
    second = np.where(labels == 1, 0.6, 0.2)
    result = paired_block_bootstrap_difference(
        labels,
        first,
        second,
        first_threshold=0.5,
        second_threshold=0.5,
        dates=dates,
        metric="pr_auc",
        iterations=25,
        confidence_level=0.95,
        seed=42,
    )
    assert result["bootstrap_iterations_valid"] == 25
    assert result["bootstrap_iterations_attempted"] >= 25
