"""Run expanding-window rolling-origin validation."""

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
from hardinge_high_flow.validation import run_rolling_origin


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run leakage-safe expanding-window validation."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "default.yaml",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use one seed, model, horizon, and two epochs.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    metrics, summary = run_rolling_origin(
        load_config(args.config),
        smoke_test=args.smoke_test,
    )
    logging.info("Rolling metrics: %s", metrics)
    logging.info("Rolling summary: %s", summary)


if __name__ == "__main__":
    main()
