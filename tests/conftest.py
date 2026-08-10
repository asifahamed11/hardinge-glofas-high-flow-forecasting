from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]
