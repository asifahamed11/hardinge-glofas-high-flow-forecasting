"""Run the reproducible project stages in order."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run data acquisition, preprocessing, and experiments."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "default.yaml",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download ERA5-Land and GloFAS inputs.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Do not download a missing NASA POWER file.",
    )
    parser.add_argument(
        "--target-source",
        choices=("glofas_proxy", "observed"),
    )
    parser.add_argument(
        "--overwrite-downloads",
        action="store_true",
        help="Replace existing remote inputs; required after acquisition fixes.",
    )
    parser.add_argument(
        "--output-namespace",
        help="Keep this experiment isolated beneath each outputs directory.",
    )
    experiment = parser.add_mutually_exclusive_group()
    experiment.add_argument("--smoke-test", action="store_true")
    experiment.add_argument("--full-experiment", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = str(args.config.resolve())
    if args.download:
        for script in (
            "download_era5_daily.py",
            "download_era5_accumulations.py",
            "download_glofas.py",
        ):
            command = [
                sys.executable,
                f"scripts/{script}",
                "--config",
                config_path,
            ]
            if args.overwrite_downloads:
                command.append("--overwrite")
            run(command)

    build_command = [
        sys.executable,
        "scripts/build_dataset.py",
        "--config",
        config_path,
    ]
    if args.offline:
        build_command.append("--offline")
    run(build_command)

    label_command = [
        sys.executable,
        "scripts/create_high_flow_labels.py",
        "--config",
        config_path,
    ]
    if args.target_source:
        label_command.extend(["--target-source", args.target_source])
    run(label_command)

    if args.smoke_test or args.full_experiment:
        experiment_command = [
            sys.executable,
            "scripts/train_evaluate.py",
            "--config",
            config_path,
        ]
        if args.smoke_test:
            experiment_command.append("--smoke-test")
        output_namespace = args.output_namespace
        if output_namespace is None and args.target_source == "observed":
            output_namespace = "observed_target"
        if output_namespace:
            experiment_command.extend(["--output-namespace", output_namespace])
        run(experiment_command)


if __name__ == "__main__":
    main()
