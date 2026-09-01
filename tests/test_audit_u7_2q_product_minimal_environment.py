from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_u7_2q_product_minimal_environment import (
    _canonical_distribution,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u7_2q_product_minimal_environment_v1.json"


def test_distribution_canonicalization_and_frozen_inventory() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert _canonical_distribution("OpenCV_Python.Headless") == (
        "opencv-python-headless"
    )
    assert len(config["required_distributions"]) == 14
    assert set(config["required_distributions"]).isdisjoint(
        config["forbidden_distribution_roots"]
    )
