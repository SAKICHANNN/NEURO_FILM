from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.audit_u7_2s_product_yaml_runtime_decoupling import (
    _canonical_distribution,
    _normalize_parent_report,
    _requirements,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "u7_2s_product_yaml_runtime_decoupling_v1.json"


def test_distribution_canonicalization_and_exact_manifest() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert _canonical_distribution("OpenCV_Python.Headless") == (
        "opencv-python-headless"
    )
    assert (
        _requirements(ROOT / config["requirements_path"])
        == config["required_distributions"]
    )
    assert set(config["removed_distributions"]).isdisjoint(
        config["required_distributions"]
    )


def test_parent_normalization_removes_only_execution_commit() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    terminal = config["u7_2r_terminal_report"]
    payload = (ROOT / terminal["path"]).read_bytes()
    normalized = _normalize_parent_report(payload)
    assert len(normalized) == terminal["normalized_scientific_bytes"]
    assert (
        hashlib.sha256(normalized).hexdigest()
        == terminal["normalized_scientific_sha256"]
    )
    source = json.loads(payload)
    rebuilt = json.loads(normalized)
    assert set(source) - set(rebuilt) == {"execution_commit"}
    assert rebuilt["status"] == "FAIL_CLOSED"
