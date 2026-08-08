"""Fail-fast audit for a journal-submission analysis release."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERIFIED_ACCUMULATION_METHOD = (
    "ERA5-Land 00 UTC accumulation shifted to the preceding UTC day"
)


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def run_checks(*, allow_proxy_target: bool = False) -> list[Check]:
    master = _read_json(PROJECT_ROOT / "outputs/metadata/master_dataset.json")
    labels = _read_json(PROJECT_ROOT / "outputs/metadata/high_flow_labels.json")
    experiment_path = (
        PROJECT_ROOT / "outputs/metadata/experiment.json"
        if allow_proxy_target
        else PROJECT_ROOT / "outputs/metadata/observed_target/experiment.json"
    )
    final_experiment = _read_json(experiment_path)
    required_target = "glofas_proxy" if allow_proxy_target else "observed"
    target_description = (
        "GloFAS-modelled high-flow proxy"
        if allow_proxy_target
        else "quality-controlled BWDB/FFWC observations"
    )
    credentials = PROJECT_ROOT / "cds_keys.txt"
    credential_text = (
        credentials.read_text(encoding="utf-8", errors="ignore")
        if credentials.is_file()
        else ""
    )
    checks = [
        Check(
            "Version-controlled release",
            _git_commit() is not None,
            "A Git commit is required so manuscript results map to immutable code.",
        ),
        Check(
            "Software licence",
            any((PROJECT_ROOT / name).is_file() for name in ("LICENSE", "LICENSE.txt")),
            "Choose and add the authors' intended open-source licence.",
        ),
        Check(
            "Credential hygiene",
            not credential_text.strip(),
            "Remove cds_keys.txt from the release and rotate any shared keys.",
        ),
        Check(
            "Declared target labels",
            labels is not None and labels.get("target_source") == required_target,
            f"Create labels from {target_description} and report that target exactly.",
        ),
        Check(
            "Target-matched final experiment",
            final_experiment is not None
            and final_experiment.get("target_source") == required_target
            and bool(final_experiment.get("run_fingerprint_sha256")),
            f"Run the final multi-seed {required_target} experiment at "
            f"{experiment_path.relative_to(PROJECT_ROOT).as_posix()}.",
        ),
        Check(
            "Verified ERA5-Land accumulations",
            master is not None
            and master.get("sources", {})
            .get("era5_land", {})
            .get("accumulation_method")
            == VERIFIED_ACCUMULATION_METHOD,
            "Re-run the accumulation downloader and rebuild the dataset.",
        ),
        Check(
            "Causal missing-data policy",
            master is not None
            and master.get("missing_data_policy", {}).get("method")
            == "past-only forward fill; no future-value interpolation",
            "Rebuild data with the causal missing-data implementation.",
        ),
    ]
    required_outputs = {
        "streamflow excluded": "outputs/metadata/ablations/no_streamflow/experiment.json",
        "streamflow included": "outputs/metadata/ablations/with_streamflow/experiment.json",
        "imbalance none": "outputs/metadata/ablations/imbalance_none/experiment.json",
        "imbalance pos_weight": (
            "outputs/metadata/ablations/imbalance_pos_weight/experiment.json"
        ),
        "imbalance focal": "outputs/metadata/ablations/imbalance_focal/experiment.json",
        "imbalance sampler": "outputs/metadata/ablations/imbalance_sampler/experiment.json",
        "rolling-origin metrics": "outputs/tables/rolling_origin_metrics.csv",
        "rolling-origin summary": "outputs/tables/rolling_origin_summary.csv",
        "rolling-origin thresholds": "outputs/tables/rolling_origin_thresholds.csv",
    }
    for label, relative_path in required_outputs.items():
        checks.append(
            Check(
                f"Required analysis: {label}",
                (PROJECT_ROOT / relative_path).is_file(),
                f"Missing {relative_path}.",
            )
        )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether the analysis artifacts are ready for submission."
    )
    parser.add_argument(
        "--allow-proxy-target",
        action="store_true",
        help=(
            "Audit a GloFAS-proxy paper instead of requiring observed labels. "
            "This does not permit claims about independently observed floods."
        ),
    )
    args = parser.parse_args()
    checks = run_checks(allow_proxy_target=args.allow_proxy_target)
    if args.allow_proxy_target:
        print(
            "TARGET MODE: GloFAS proxy only; independently observed flood claims "
            "are out of scope.\n"
        )
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.name}: {check.detail}")
    failures = sum(not check.passed for check in checks)
    print(f"\nPublication-readiness result: {len(checks) - failures}/{len(checks)} passed")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
