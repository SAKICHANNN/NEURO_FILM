from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import src.real_film.three_stock_scan_storage_preflight as target

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/sf3_a0p_three_stock_scan_storage_preflight_v1.json"


def test_current_p_backed_scan_tier_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        target.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=10_000_000_000, used=0, free=6_531_579_904),
    )
    report = target.evaluate(CONTRACT, root=ROOT)
    assert report["automatic_pass"] is True
    assert report["scan_tasks"] == 153
    assert report["scan_pixels"] == 6_000_000
    assert report["worst_case_plan_bytes"] == 5_668_432_128
    assert report["projected_remaining_bytes"] == 863_147_776
    assert report["scan_to_stimulus_pixel_ratio"] >= 3.0


def test_insufficient_capacity_fails_without_relaxing_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        target.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=6_000_000_000, used=0, free=6_000_000_000),
    )
    report = target.evaluate(CONTRACT, root=ROOT)
    assert report["automatic_pass"] is False
    assert report["gates"]["worst_case_plan_preserves_free_space"] is False
    assert report["decision"].startswith("RETAIN_SF3")


def test_contract_rejects_scan_profile_drift(tmp_path: Path) -> None:
    raw = CONTRACT.read_text(encoding="utf-8").replace('"width": 3000', '"width": 2999')
    path = tmp_path / "contract.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(target.ThreeStockScanStoragePreflightError, match="profile drift"):
        target.load_contract(path, root=ROOT)
