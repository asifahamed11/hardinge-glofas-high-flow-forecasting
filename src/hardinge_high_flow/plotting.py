"""Accessible, publication-quality figures for the forecasting workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_curve

from .evaluation import reliability_points

DISPLAY_NAMES = {
    "climatology": "Seasonal climatology",
    "persistence": "Persistence",
    "glofas_signal": "Current-day GloFAS signal",
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "hist_gradient_boosting": "Gradient boosting",
    "lstm": "LSTM",
    "gru": "GRU",
    "cnn_lstm": "CNN–LSTM",
    "attention_lstm": "Attention-LSTM",
}

LINE_STYLES = {
    "climatology": "--",
    "persistence": ":",
    "glofas_signal": "-.",
    "logistic_regression": "-",
    "random_forest": "--",
    "hist_gradient_boosting": "-.",
    "lstm": "-",
    "gru": "--",
    "cnn_lstm": "-.",
    "attention_lstm": ":",
}

MARKERS = {
    "climatology": "o",
    "persistence": "s",
    "glofas_signal": "^",
    "logistic_regression": "D",
    "random_forest": "v",
    "hist_gradient_boosting": "P",
    "lstm": "X",
    "gru": "<",
    "cnn_lstm": ">",
    "attention_lstm": "h",
}


def figure_colors(config: dict[str, Any]) -> dict[str, str]:
    palette = config["figures"]["palette"]
    return {
        "climatology": palette["gray"],
        "persistence": palette["black"],
        "glofas_signal": "#4D4D4D",
        "logistic_regression": palette["orange"],
        "random_forest": palette["bluish_green"],
        "hist_gradient_boosting": palette["sky_blue"],
        "lstm": palette["blue"],
        "gru": palette["vermillion"],
        "cnn_lstm": palette["reddish_purple"],
        "attention_lstm": palette["yellow"],
    }


def configure_figure_style(config: dict[str, Any]) -> None:
    figure_config = config["figures"]
    base_size = float(figure_config["base_font_size"])
    matplotlib.rcParams.update(
        {
            "font.family": figure_config["font_family"],
            "font.size": base_size,
            "axes.labelsize": base_size,
            "axes.titlesize": base_size,
            "xtick.labelsize": base_size - 0.5,
            "ytick.labelsize": base_size - 0.5,
            "legend.fontsize": base_size - 0.5,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.25,
            "lines.markersize": 4.0,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(
    figure: plt.Figure,
    stem: str,
    output_directory: Path,
    config: dict[str, Any],
) -> list[Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    figure_config = config["figures"]
    saved = []
    for file_format in figure_config["formats"]:
        destination = output_directory / f"{stem}.{file_format}"
        options: dict[str, Any] = {
            "bbox_inches": "tight",
            "pad_inches": 0.03,
        }
        if file_format == "png":
            options["dpi"] = int(figure_config["png_dpi"])
        elif file_format in {"tif", "tiff"}:
            options["dpi"] = int(figure_config["tiff_dpi"])
            options["pil_kwargs"] = {"compression": "tiff_lzw"}
        figure.savefig(destination, **options)
        if file_format in {"png", "tif", "tiff"}:
            dpi = int(figure_config["png_dpi" if file_format == "png" else "tiff_dpi"])
            with Image.open(destination) as raster:
                rgb_raster = raster.convert("RGB")
            save_options: dict[str, Any] = {"dpi": (dpi, dpi)}
            if file_format in {"tif", "tiff"}:
                save_options["compression"] = "tiff_lzw"
            rgb_raster.save(destination, **save_options)
        saved.append(destination)
    plt.close(figure)
    return saved


def _ensemble_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    grouping = [
        "model",
        "horizon_days",
        "target_date",
        "target_high_flow",
        "target_value",
    ]
    ensemble = (
        predictions.groupby(grouping, as_index=False, observed=True)
        .agg(
            probability=("probability", "mean"),
            threshold=("threshold", "mean"),
        )
        .sort_values(["model", "horizon_days", "target_date"])
    )
    return ensemble


def plot_horizon_skill(
    metrics: pd.DataFrame,
    output_directory: Path,
    config: dict[str, Any],
) -> list[Path]:
    colors = figure_colors(config)
    width = float(config["figures"]["double_column_width_inches"])
    figure, axes = plt.subplots(1, 2, figsize=(width, width * 0.5))
    grouped = (
        metrics.groupby(["model", "horizon_days"], observed=True)
        .agg(
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
        )
        .reset_index()
    )
    model_order = list(dict.fromkeys(metrics["model"]))
    for i, model in enumerate(model_order):
        subset = grouped[grouped["model"] == model].sort_values("horizon_days")
        if subset.empty:
            continue

        jitter = (i - (len(model_order) - 1) / 2.0) * 0.08
        x_values = subset["horizon_days"] + jitter

        for axis, metric, standard_deviation in (
            (axes[0], "pr_auc_mean", "pr_auc_std"),
            (axes[1], "f1_mean", "f1_std"),
        ):
            axis.errorbar(
                x_values,
                subset[metric],
                yerr=subset[standard_deviation].fillna(0),
                color=colors[model],
                linestyle=LINE_STYLES[model],
                marker=MARKERS[model],
                capsize=2,
                alpha=0.85,
                label=DISPLAY_NAMES[model],
            )
    axes[0].set_ylabel("Area under precision–recall curve")
    axes[1].set_ylabel("F1 score")
    for panel, axis in zip(("a", "b"), axes, strict=False):
        axis.set_xlabel("Forecast horizon (days)")
        axis.set_xticks(sorted(metrics["horizon_days"].unique()))
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.5)
        axis.text(
            0.01,
            0.98,
            panel,
            transform=axis.transAxes,
            fontweight="bold",
            va="top",
        )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        frameon=False,
    )
    figure.subplots_adjust(bottom=0.35, wspace=0.35)
    return save_figure(
        figure,
        "Fig1_skill_by_horizon",
        output_directory,
        config,
    )


def plot_precision_recall(
    predictions: pd.DataFrame,
    output_directory: Path,
    config: dict[str, Any],
) -> list[Path]:
    ensemble = _ensemble_predictions(predictions)
    horizon = int(ensemble["horizon_days"].min())
    subset = ensemble[ensemble["horizon_days"] == horizon]
    width = float(config["figures"]["single_column_width_inches"])
    figure, axis = plt.subplots(figsize=(width, width * 1.0))
    colors = figure_colors(config)
    for model in dict.fromkeys(subset["model"]):
        model_data = subset[subset["model"] == model]
        precision, recall, _ = precision_recall_curve(
            model_data["target_high_flow"],
            model_data["probability"],
        )
        axis.plot(
            recall,
            precision,
            color=colors[model],
            linestyle=LINE_STYLES[model],
            alpha=0.85,
            label=DISPLAY_NAMES[model],
        )
    prevalence = float(subset["target_high_flow"].mean())
    axis.axhline(
        prevalence,
        color=config["figures"]["palette"]["gray"],
        linestyle="--",
        linewidth=0.9,
        label=f"Prevalence ({prevalence:.3f})",
    )
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(color="#E1E1E1", linewidth=0.5)
    axis.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
    )
    figure.subplots_adjust(bottom=0.35)
    return save_figure(
        figure,
        f"Fig2_precision_recall_h{horizon}",
        output_directory,
        config,
    )


def plot_reliability(
    predictions: pd.DataFrame,
    output_directory: Path,
    config: dict[str, Any],
) -> list[Path]:
    ensemble = _ensemble_predictions(predictions)
    horizon = int(ensemble["horizon_days"].min())
    subset = ensemble[ensemble["horizon_days"] == horizon]
    width = float(config["figures"]["single_column_width_inches"])
    figure, axis = plt.subplots(figsize=(width, width * 1.0))
    colors = figure_colors(config)
    for model in dict.fromkeys(subset["model"]):
        model_data = subset[subset["model"] == model]
        points = reliability_points(
            model_data["target_high_flow"].to_numpy(),
            model_data["probability"].to_numpy(),
            int(config["evaluation"]["reliability_bins"]),
        )
        axis.plot(
            points["mean_probability"],
            points["observed_fraction"],
            marker=MARKERS[model],
            color=colors[model],
            linestyle=LINE_STYLES[model],
            alpha=0.85,
            label=DISPLAY_NAMES[model],
        )
    axis.plot([0, 1], [0, 1], color="#666666", linestyle="--", linewidth=0.9)
    axis.set_xlabel("Forecast probability")
    axis.set_ylabel("Observed frequency")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(color="#E1E1E1", linewidth=0.5)
    axis.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
    )
    figure.subplots_adjust(bottom=0.35)
    return save_figure(
        figure,
        f"Fig3_reliability_h{horizon}",
        output_directory,
        config,
    )


def plot_event_timeline(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    output_directory: Path,
    config: dict[str, Any],
) -> list[Path]:
    ensemble = _ensemble_predictions(predictions)
    horizon = int(ensemble["horizon_days"].min())
    best_row = (
        metrics[metrics["horizon_days"] == horizon]
        .groupby("model", observed=True)["pr_auc"]
        .mean()
        .idxmax()
    )
    subset = ensemble[
        (ensemble["horizon_days"] == horizon) & (ensemble["model"] == best_row)
    ].copy()
    positive_by_year = subset.groupby(
        subset["target_date"].dt.year,
        observed=True,
    )["target_high_flow"].sum()
    selected_year = int(positive_by_year.idxmax())
    subset = subset[subset["target_date"].dt.year == selected_year]

    width = float(config["figures"]["double_column_width_inches"])
    figure, axis = plt.subplots(figsize=(width, width * 0.42))
    color = figure_colors(config)[best_row]
    axis.plot(
        subset["target_date"],
        subset["probability"],
        color=color,
        label=f"{DISPLAY_NAMES[best_row]} probability",
    )
    axis.axhline(
        float(subset["threshold"].mean()),
        color="#444444",
        linestyle="--",
        linewidth=0.9,
        label="Validation-selected threshold",
    )
    axis.fill_between(
        subset["target_date"],
        0,
        subset["target_high_flow"],
        step="mid",
        color="#F6D7CF",
        alpha=1.0,
        label="Target high-flow days",
    )
    axis.set_ylabel("Probability")
    axis.set_xlabel("Target date")
    axis.set_ylim(0, 1)
    axis.grid(axis="y", color="#E1E1E1", linewidth=0.5)
    axis.legend(
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
    )
    figure.subplots_adjust(bottom=0.3)
    return save_figure(
        figure,
        f"Fig4_event_timeline_h{horizon}_{selected_year}",
        output_directory,
        config,
    )


def plot_confusion_matrices(
    predictions: pd.DataFrame,
    output_directory: Path,
    config: dict[str, Any],
) -> list[Path]:
    ensemble = _ensemble_predictions(predictions)
    horizon = int(ensemble["horizon_days"].min())
    subset = ensemble[ensemble["horizon_days"] == horizon]
    models = list(dict.fromkeys(subset["model"]))
    columns = 3
    rows = int(np.ceil(len(models) / columns))
    width = float(config["figures"]["double_column_width_inches"])
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(width, rows * width / columns * 0.95),
        squeeze=False,
    )
    for axis, model in zip(axes.ravel(), models, strict=False):
        model_data = subset[subset["model"] == model]
        matrix = confusion_matrix(
            model_data["target_high_flow"],
            model_data["probability"] >= model_data["threshold"],
            labels=[0, 1],
        )
        image = axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:,}",
                    ha="center",
                    va="center",
                    color=(
                        "white" if matrix[row, column] > matrix.max() / 2 else "black"
                    ),
                )
        axis.set_title(DISPLAY_NAMES[model])
        axis.set_xticks([0, 1], ["No", "Yes"])
        axis.set_yticks([0, 1], ["No", "Yes"])
        axis.set_xlabel("Predicted high flow")
        axis.set_ylabel("Target high flow")
        image.set_clim(0, matrix.max())
    for axis in axes.ravel()[len(models) :]:
        axis.set_visible(False)
    figure.subplots_adjust(wspace=0.6, hspace=0.7)
    return save_figure(
        figure,
        f"Fig5_confusion_matrices_h{horizon}",
        output_directory,
        config,
    )


def generate_publication_figures(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    output_directory: Path,
    config: dict[str, Any],
) -> list[Path]:
    configure_figure_style(config)
    saved = []
    saved.extend(plot_horizon_skill(metrics, output_directory, config))
    saved.extend(plot_precision_recall(predictions, output_directory, config))
    saved.extend(plot_reliability(predictions, output_directory, config))
    saved.extend(
        plot_event_timeline(
            predictions,
            metrics,
            output_directory,
            config,
        )
    )
    saved.extend(plot_confusion_matrices(predictions, output_directory, config))
    return saved
