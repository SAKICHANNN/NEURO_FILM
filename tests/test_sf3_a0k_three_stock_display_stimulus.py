from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.real_film.three_stock_display_stimulus import (
    ThreeStockDisplayStimulusError,
    build_pack,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/sf3_a0k_three_stock_display_stimulus_v1.json"


def _inventory(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_sf3_a0k_two_clean_builds_are_exact(tmp_path: Path) -> None:
    contract = load_contract(CONTRACT)
    left = tmp_path / "left"
    right = tmp_path / "right"
    left_report = build_pack(contract, ROOT, left)
    right_report = build_pack(contract, ROOT, right)
    assert left_report == right_report
    assert _inventory(left) == _inventory(right)
    assert left_report["scene_counts"] == {"development": 8, "confirmation": 4}
    assert left_report["diagnostic_count"] == 3
    manifest = json.loads((left / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["scene_rows"]) == 12
    assert all(row["stimulus_sha256"] == row["source_decoded_sha256"] for row in manifest["scene_rows"])


def test_sf3_a0k_is_create_only(tmp_path: Path) -> None:
    contract = load_contract(CONTRACT)
    destination = tmp_path / "pack"
    build_pack(contract, ROOT, destination)
    with pytest.raises(ThreeStockDisplayStimulusError, match="must not exist"):
        build_pack(contract, ROOT, destination)


def test_sf3_a0k_rejects_rights_drift(tmp_path: Path) -> None:
    contract = load_contract(CONTRACT)
    contract["required_source_policy"]["rights_scope"] = "wrong"
    with pytest.raises(ThreeStockDisplayStimulusError, match="rights_scope drift"):
        build_pack(contract, ROOT, tmp_path / "pack")
