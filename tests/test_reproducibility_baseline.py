from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_reproducibility_baseline import load_baseline, sha256_file, verify


ROOT = Path(__file__).resolve().parents[1]


def test_committed_reproducibility_baseline_verifies() -> None:
    baseline = load_baseline(ROOT / "configs" / "reproducibility_baseline.json")
    result = verify(baseline)
    assert result["ok"], result["errors"]


def test_checksum_drift_is_reported(tmp_path: Path) -> None:
    tracked = tmp_path / "configs" / "fixture.json"
    tracked.parent.mkdir(parents=True)
    tracked.write_text('{"version": 1}\n', encoding="utf-8")
    test_file = tmp_path / "tests" / "test_fixture.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("pass\n", encoding="utf-8")
    registry = tmp_path / "configs" / "registry.json"
    registry.write_text("{}\n", encoding="utf-8")
    baseline = {
        "schema_version": 1,
        "tracked_inputs": {"configs/fixture.json": "0" * 64},
        "required_test_files": ["tests/test_fixture.py"],
        "benchmark_registry": "configs/registry.json",
        "regression_fixture_registry": "configs/registry.json",
    }
    result = verify(baseline, root=tmp_path)
    assert not result["ok"]
    assert "checksum drift: configs/fixture.json" in result["errors"]
    assert result["input_hashes"]["configs/fixture.json"] == sha256_file(tracked)
