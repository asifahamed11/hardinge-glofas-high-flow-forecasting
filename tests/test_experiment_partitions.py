from __future__ import annotations

import numpy as np

from hardinge_high_flow.experiment import _validation_partitions


def test_validation_partition_requires_rare_events_in_both_blocks() -> None:
    labels = np.zeros(1_800, dtype=int)
    labels[550:620] = 1
    labels[1_250:1_320] = 1
    calibration, threshold = _validation_partitions(
        labels,
        calibration_fraction=0.65,
        minimum_positives=30,
    )
    assert labels[calibration].sum() >= 30
    assert labels[threshold].sum() >= 30
    assert calibration.max() < threshold.min()
