from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from src.eval.three_stock_autonomous_blind_salience import (
    BlindSalienceError,
    _mapping,
)


def test_mapping_is_deterministic_and_reorders_every_source() -> None:
    sources = [f"s{index}" for index in range(16)]
    arms = ["v", "p", "e"]
    first = _mapping(sources, arms, "seed")
    second = _mapping(sources, arms, "seed")
    assert first == second
    by_source = {(row["round"], row["source_id"]): row["label_to_arm"] for row in first}
    for source in sources:
        assert by_source[(1, source)] != by_source[(2, source)]


def test_contract_is_frozen_and_uses_existing_population() -> None:
    root = Path(__file__).resolve().parents[1]
    contract = json.loads(
        (
            root / "configs/rf3_d15_three_stock_autonomous_blind_salience_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert contract["status"] == "FROZEN_BEFORE_BLIND_MATERIAL_BUILD_OR_REVIEW"
    assert contract["source_count"] == 16
    assert len(contract["arm_ids"]) == 3
    assert contract["gates"]["minimum_visible_source_count_per_stock_pair"] == 12
    assert contract["blind_protocol"]["prior_labeled_exposure_disclosed"] is True


def test_sheet_fixture_can_be_decoded(tmp_path: Path) -> None:
    from src.eval.three_stock_autonomous_blind_salience import _sheet

    paths = {}
    for label, color in zip(
        ("A", "B", "C"), ((255, 0, 0), (0, 255, 0), (0, 0, 255)), strict=True
    ):
        path = tmp_path / f"{label}.png"
        Image.new("RGB", (32, 24), color).save(path)
        paths[label] = path
    destination = tmp_path / "sheet.png"
    _sheet(paths, destination)
    with Image.open(destination) as image:
        assert image.size == (2160, 596)


def test_hash_helper_detects_drift(tmp_path: Path) -> None:
    from src.eval.three_stock_autonomous_blind_salience import _sha256_file

    path = tmp_path / "x"
    path.write_bytes(b"a")
    assert _sha256_file(path) == hashlib.sha256(b"a").hexdigest()
    path.write_bytes(b"b")
    assert _sha256_file(path) != hashlib.sha256(b"a").hexdigest()


def test_error_type_is_value_error() -> None:
    assert issubclass(BlindSalienceError, ValueError)
