from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

import src.real_film.three_stock_scan_storage_preflight as target
from src.preprocess import srgb_icc_profile

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


def test_scan_file_requires_exact_srgb16_container(tmp_path: Path) -> None:
    _, contract = target.load_contract(CONTRACT, root=ROOT)
    profile = {**contract["scan_profile"], "width": 10, "height": 8}
    icc = srgb_icc_profile()
    path = tmp_path / "scan.tif"
    tifffile.imwrite(
        path,
        np.arange(8 * 10 * 3, dtype=np.uint16).reshape(8, 10, 3),
        photometric="rgb",
        planarconfig="contig",
        metadata=None,
        extratags=[(34675, "B", len(icc), icc, False)],
    )
    facts = target.validate_scan_file(path, profile)
    assert facts["shape"] == [8, 10, 3]
    assert facts["icc_profile_sha256"] == profile["required_embedded_icc_profile_sha256"]

    unprofiled = tmp_path / "unprofiled.tif"
    tifffile.imwrite(
        unprofiled,
        np.zeros((8, 10, 3), dtype=np.uint16),
        photometric="rgb",
        metadata=None,
    )
    with pytest.raises(target.ThreeStockScanStoragePreflightError, match="ICC"):
        target.validate_scan_file(unprofiled, profile)
