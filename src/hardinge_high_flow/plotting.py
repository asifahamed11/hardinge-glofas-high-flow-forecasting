"""Accessible figures for the retrospective proxy-prediction workflow."""

from __future__ import annotations

import re
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
from .reporting import canonicalize_metric_names, constant_decision_threshold

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
            dpi_key = "png_dpi" if file_format == "png" else "tiff_dpi"
            dpi = int(figure_config[dpi_key])
            with Image.open(destination) as raster:
                rgb_raster = raster.convert("RGB")
            save_options: dict[str, Any] = {"dpi": (dpi, dpi)}
            if file_format in {"tif", "tiff"}:
                save_options["compression"] = "tiff_lzw"
            rgb_raster.save(destination, **save_options)
        saved.append(destination)
    plt.close(figure)
    return saved


def _configured_models(
    frame: pd.DataFrame,
    config: dict[str, Any],
    key: str,
) -> list[str]:
    available = set(frame["model"].astype(str))
    requested = list(config["figures"].get(key, []))
    selected = [model for model in requested if model in available]
    return selected or sorted(available)


def _representative_predictions(
    predictions: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Select one evaluated seed per model without averaging thresholds."""

    requested_seed = int(config["figures"].get("representative_seed", 42))
    frames = []
    for _, group in predictions.groupby("model", observed=True):
        seeds = sorted(group["seed"].astype(int).unique())
        seed = requested_seed if requested_seed in seeds else seeds[0]
        frames.append(group[group["seed"].astype(int) == seed])
    return pd.concat(frames, ignore_index=True).sort_values(
        ["model", "horizon_days", "target_date"]
    )


def plot_horizon_skill(
    metrics: pd.DataFrame,
    output_directory: Path,
    config: dict[str, Any],
) -> list[Path]:
    metrics = canonicalize_metric_names(metrics)
    models = _configured_models(metrics, config, "main_models")
    metrics = metrics[metrics["model"].isin(models)]
    colors = figure_colors(config)
    width = float(config["figures"]["double_column_width_inches"])
    figure, axes = plt.subplots(1, 2, figsize=(width, width * 0.43))
    grouped = (
        metrics.groupby(["model", "horizon_days"], observed=True)
        .agg(
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            average_precision_mean=("average_precision", "mean"),
            average_precision_std=("average_precision", "std"),
        )
        .reset_index()
    )
    for index, model in enumerate(models):
        subset = grouped[grouped["model"] == model].sort_values("horizon_days")
        if subset.empty:
            continue
        jitter = (index - (len(models) - 1) / 2.0) * 0.055
        x_values = subset["horizon_days"] + jitter
        for axis, metric, standard_deviation in (
            (
                axes[0],
                "average_precision_mean",
                "average_precision_std",
            ),
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
                label=DISPLAY_NAMES[model],
            )
    axes[0].set_ylabel("Average precision (AP)")
    axes[1].set_ylabel("F1 score")
    for panel, axis in zip(("a", "b"), axes, strict=True):
        axis.set_xlabel("Prediction horizon (days)")
        axis.set_xticks(sorted(metrics["horizon_days"].unique()))
        axis.set_ylim(0, 1.02)
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
        bbox_to_anchor=(0.5, -0.01),
        ncol=min(3, len(labels)),
        frameon=False,
    )
    figure.subplots_adjust(bottom=0.27, wspace=0.32)
    return save_figure(figure, "Fig1_skill_by_horizon", output_directory, config)


def plot_precision_recall(
    predictions: pd.DataFrame,
    output_directory: Path,
    config: dict[str, Any],
) -> list[Path]:
    representative = _representative_predictions(predictions, config)
    models = _configured_models(representative, config, "main_models")
    horizon = int(representative["horizon_days"].min())
    subset = representative[
        (representative["horizon_days"] == horizon)
        & representative["model"].isin(models)
    ]
    width = float(config["figures"]["single_column_width_inches"])
    figure, axis = plt.subplots(figsize=(width, width * 0.82))
    colors = figure_colors(config)
    for model in models:
        model_data = subset[subset["model"] == model]
        if model_data.empty:
            continue
        precision, recall, _ = precision_recall_curve(
            model_data["target_high_flow"],
            model_data["probability"],
        )
        axis.plot(
            recall,
            precision,
            color=colors[model],
            linestyle=LINE_STYLES[model],
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
    axis.set_ylim(0, 1.02)
    axis.grid(color="#E1E1E1", linewidth=0.5)
    axis.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
    )
    figure.subplots_adjust(bottom=0.37)
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
    representative = _representative_predictions(predictions, config)
    models = _configured_models(representative, config, "calibrated_models")
    horizon = int(representative["horizon_days"].min())
    subset = representative[
        (representative["horizon_days"] == horizon)
        & representative["model"].isin(models)
    ]
    width = float(config["figures"]["single_column_width_inches"])
    figure, axis = plt.subplots(figsize=(width, width * 0.82))
    colors = figure_colors(config)
    for model in models:
        model_data = subset[subset["model"] == model]
        if model_data.empty:
            continue
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
            label=DISPLAY_NAMES[model],
        )
    axis.plot([0, 1], [0, 1], color="#666666", linestyle="--", linewidth=0.9)
    axis.set_xlabel("Predicted probability")
    axis.set_ylabel("Observed frequency")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.grid(color="#E1E1E1", linewidth=0.5)
    axis.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
    )
    figure.subplots_adjust(bottom=0.37)
    return save_figure(
        figure,
        f"Fig3_reliability_h{horizon}",
        output_directory,
        config,
    )


def plot_event_timeline(
    predictions: pd.DataFrame,
    output_directory: Path,
    config: dict[str, Any],
) -> list[Path]:
    representative = _representative_predictions(predictions, config)
    models = _configured_models(representative, config, "event_models")
    horizon = int(representative["horizon_days"].min())
    horizon_data = representative[representative["horizon_days"] == horizon]
    reference_model = models[0]
    reference = horizon_data[horizon_data["model"] == reference_model].copy()
    positive_by_year = reference.groupby(
        reference["target_date"].dt.year,
        observed=True,
    )["target_high_flow"].sum()
    selected_year = int(positive_by_year.idxmax())
    reference = reference[reference["target_date"].dt.year == selected_year]

    width = float(config["figures"]["double_column_width_inches"])
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(width, width * 0.56),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.25]},
    )
    axes[0].plot(
        reference["target_date"],
        reference["target_value"],
        color=config["figures"]["palette"]["blue"],
        label="GloFAS-modelled discharge",
    )
    if "target_threshold" in reference and reference["target_threshold"].notna().any():
        axes[0].axhline(
            float(reference["target_threshold"].dropna().iloc[0]),
            color=config["figures"]["palette"]["vermillion"],
            linestyle="--",
            label="Training 95th-percentile threshold",
        )
    axes[0].set_ylabel("Discharge (m3 s-1)")
    axes[0].grid(axis="y", color="#E1E1E1", linewidth=0.5)
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].text(
        0.01,
        0.95,
        "a",
        transform=axes[0].transAxes,
        fontweight="bold",
        va="top",
    )

    colors = figure_colors(config)
    for model in models:
        model_data = horizon_data[
            (horizon_data["model"] == model)
            & (horizon_data["target_date"].dt.year == selected_year)
        ]
        if model_data.empty:
            continue
        axes[1].plot(
            model_data["target_date"],
            model_data["probability"],
            color=colors[model],
            linestyle=LINE_STYLES[model],
            label=DISPLAY_NAMES[model],
        )
    high_flow = reference["target_high_flow"].astype(bool).to_numpy()
    axes[1].fill_between(
        reference["target_date"],
        0,
        1,
        where=high_flow,
        step="mid",
        color="#F6D7CF",
        label="Target high-flow days",
    )
    axes[1].set_ylabel("Predicted probability")
    axes[1].set_xlabel("Target date")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(axis="y", color="#E1E1E1", linewidth=0.5)
    axes[1].text(
        0.01,
        0.95,
        "b",
        transform=axes[1].transAxes,
        fontweight="bold",
        va="top",
    )
    axes[1].legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.27),
    )
    figure.subplots_adjust(bottom=0.25, hspace=0.12)
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
    representative = _representative_predictions(predictions, config)
    models = _configured_models(representative, config, "confusion_models")
    horizon = int(representative["horizon_days"].min())
    subset = representative[
        (representative["horizon_days"] == horizon)
        & representative["model"].isin(models)
    ]
    columns = min(2, len(models))
    rows = int(np.ceil(len(models) / columns))
    width = float(config["figures"]["double_column_width_inches"])
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(width, rows * width / columns * 0.62),
        squeeze=False,
    )
    for axis, model in zip(axes.ravel(), models, strict=False):
        model_data = subset[subset["model"] == model]
        threshold = constant_decision_threshold(model_data)
        matrix = confusion_matrix(
            model_data["target_high_flow"],
            model_data["probability"] >= threshold,
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
    figure.subplots_adjust(wspace=0.35, hspace=0.45)
    return save_figure(
        figure,
        f"Fig5_confusion_matrices_h{horizon}",
        output_directory,
        config,
    )


def _feature_label(name: str) -> str:
    if name == "day_of_year_sin":
        return "Day-of-year sine"
    if name == "day_of_year_cos":
        return "Day-of-year cosine"
    if name == "glofas_discharge_m3s":
        return "GloFAS discharge"
    label = name.replace("era5_", "ERA5 ").replace("nasa_", "NASA ")
    label = re.sub(r"_mean_(\d+)d", r", \1-day mean", label)
    label = re.sub(r"_sum_(\d+)d", r", \1-day sum", label)
    label = re.sub(r"_std_(\d+)d", r", \1-day SD", label)
    label = re.sub(r"_lag_(\d+)d", r", \1-day lag", label)
    label = label.replace("_change_1d", ", 1-day change")
    for unit in ("_m3m3", "_mm", "_percent", "_c"):
        label = label.replace(unit, "")
    return " ".join(label.replace("_", " ").split())


def plot_permutation_importance(
    importance_frames: dict[str, pd.DataFrame],
    output_directory: Path,
    config: dict[str, Any],
) -> list[Path]:
    """Compare grouped feature importance with and without current discharge."""

    usable = {name: frame for name, frame in importance_frames.items() if not frame.empty}
    if not usable:
        return []
    width = float(config["figures"]["double_column_width_inches"])
    figure, axes = plt.subplots(
        1,
        len(usable),
        figsize=(width, width * 0.62),
        squeeze=False,
    )
    colors = [
        config["figures"]["palette"]["bluish_green"],
        config["figures"]["palette"]["blue"],
    ]
    for panel_index, ((name, frame), axis) in enumerate(
        zip(usable.items(), axes.ravel(), strict=True)
    ):
        grouped = (
            frame.groupby("feature", observed=True)["importance"]
            .agg(["mean", "std"])
            .fillna(0)
            .nlargest(10, "mean")
            .sort_values("mean")
        )
        positions = np.arange(len(grouped))
        axis.barh(
            positions,
            grouped["mean"],
            xerr=grouped["std"],
            color=colors[panel_index % len(colors)],
            error_kw={"elinewidth": 0.7, "capsize": 1.5},
        )
        axis.set_yticks(positions, [_feature_label(value) for value in grouped.index])
        axis.axvline(0, color="#555555", linewidth=0.7)
        axis.set_xlabel("Decrease in average precision")
        axis.grid(axis="x", color="#E1E1E1", linewidth=0.5)
        axis.text(
            0.01,
            0.99,
            chr(ord("a") + panel_index),
            transform=axis.transAxes,
            fontweight="bold",
            va="top",
        )
        axis.text(
            0.99,
            0.02,
            name,
            transform=axis.transAxes,
            ha="right",
            va="bottom",
        )
    figure.subplots_adjust(wspace=0.65, left=0.19, right=0.98, bottom=0.13)
    return save_figure(
        figure,
        "Fig6_grouped_permutation_importance",
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
    predictions = predictions.copy()
    predictions["target_date"] = pd.to_datetime(predictions["target_date"])
    predictions["issue_date"] = pd.to_datetime(predictions["issue_date"])
    saved = []
    saved.extend(plot_horizon_skill(metrics, output_directory, config))
    saved.extend(plot_precision_recall(predictions, output_directory, config))
    saved.extend(plot_reliability(predictions, output_directory, config))
    saved.extend(plot_event_timeline(predictions, output_directory, config))
    saved.extend(plot_confusion_matrices(predictions, output_directory, config))
    return saved
