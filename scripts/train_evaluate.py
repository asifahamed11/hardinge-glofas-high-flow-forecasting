"""Run leakage-safe multi-horizon forecasting experiments."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from hardinge_high_flow.config import load_config
from hardinge_high_flow.experiment import run_experiment, smoke_test_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train baselines and sequence models with an untouched test period."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "default.yaml",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a short end-to-end integration test.",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        help="Override configured lead times.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Override configured random seeds.",
    )
    parser.add_argument(
        "--imbalance-strategy",
        choices=("none", "pos_weight", "focal", "sampler"),
        help="Run one mutually exclusive neural-model imbalance strategy.",
    )
    streamflow = parser.add_mutually_exclusive_group()
    streamflow.add_argument(
        "--include-streamflow",
        action="store_true",
        help="Include current GloFAS discharge for an ablation run.",
    )
    streamflow.add_argument(
        "--exclude-streamflow",
        action="store_true",
        help="Exclude GloFAS discharge from learned models.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        help="Override the year-block bootstrap iterations.",
    )
    parser.add_argument(
        "--output-namespace",
        help="Write this run beneath a named output subdirectory.",
    )
    return parser.parse_args()


def apply_overrides(
    config: dict,
    args: argparse.Namespace,
) -> dict:
    if args.smoke_test:
        config = smoke_test_config(config)
    if args.horizons:
        if min(args.horizons) < 1:
            raise ValueError("Horizons must be positive.")
        config["experiment"]["horizons"] = sorted(set(args.horizons))
    if args.seeds:
        config["experiment"]["seeds"] = list(dict.fromkeys(args.seeds))
    if args.imbalance_strategy:
        config["experiment"]["imbalance_strategy"] = args.imbalance_strategy
    if args.include_streamflow:
        config["features"]["include_streamflow"] = True
    if args.exclude_streamflow:
        config["features"]["include_streamflow"] = False
    if args.bootstrap_iterations is not None:
        if args.bootstrap_iterations < 20:
            raise ValueError("Use at least 20 bootstrap iterations.")
        config["evaluation"]["bootstrap_iterations"] = args.bootstrap_iterations
    return config


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    outputs = run_experiment(
        config,
        output_namespace=args.output_namespace,
    )
    for name, path in outputs.items():
        logging.info("%s: %s", name, path)


if __name__ == "__main__":
    main()
