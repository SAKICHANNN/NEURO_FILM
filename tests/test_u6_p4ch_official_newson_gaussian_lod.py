from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from src.eval.official_newson_gaussian_lod import (
    OfficialNewsonLodError,
    ZipEntry,
    _evaluate_pair,
    _selected_rows,
    higher_order_features,
    parse_central_directory,
    validate_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4ch_official_newson_gaussian_lod_v1.json"


def load_contract() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def _central_entry(name: str, *, offset: int = 7) -> bytes:
    encoded = name.encode("utf-8")
    return struct.pack(
        "<4s6H3L5H2L",
        b"PK\x01\x02",
        20,
        20,
        0,
        8,
        0,
        0,
        zlib.crc32(b"payload"),
        7,
        7,
        len(encoded),
        0,
        0,
        0,
        0,
        0,
        offset,
    ) + encoded


def test_contract_rejects_archive_or_selection_drift() -> None:
    contract = load_contract()
    contract["archive"]["entry_count"] -= 1
    with pytest.raises(OfficialNewsonLodError, match="contract drift"):
        validate_contract(contract)

    contract = load_contract()
    contract["bounded_source_selection"]["selected_indices_by_radius"]["0.8"].pop()
    with pytest.raises(OfficialNewsonLodError, match="contract drift"):
        validate_contract(contract)


def test_central_directory_parser_rejects_duplicate_or_bad_signature() -> None:
    entry = _central_entry("root/a.png")
    parsed = parse_central_directory(entry)
    assert parsed["root/a.png"].compressed_size == 7
    with pytest.raises(OfficialNewsonLodError, match="duplicate"):
        parse_central_directory(entry + entry)
    with pytest.raises(OfficialNewsonLodError, match="signature"):
        parse_central_directory(b"NOPE")


def test_selected_rows_reproduce_frozen_metadata_algorithm() -> None:
    contract = load_contract()
    rows = [
        {"img_name": "unused.png", "seed": index, "radius": 0.05, "zoom": 1.0}
        for index in range(4000)
    ]
    entries: dict[str, ZipEntry] = {}
    for radius, indices in contract["bounded_source_selection"][
        "selected_indices_by_radius"
    ].items():
        for position, index in enumerate(indices):
            image = f"{radius}-{position}.png"
            rows[index] = {
                "img_name": image,
                "seed": index,
                "radius": float(radius),
                "zoom": 1.0,
            }
            for name in (
                f"NeuralFilmGrainRendering_Dataset/Mixed/test/{image}",
                f"NeuralFilmGrainRendering_Dataset/Mixed_grain/test/{index:06d}.png",
            ):
                entries[name] = ZipEntry(name, 8, 0, 1, 1, 0)
    # The synthetic fixture does not reproduce the official hash order and must fail closed.
    with pytest.raises(OfficialNewsonLodError, match="selected row identity drift"):
        _selected_rows(contract, rows, entries)


def test_higher_order_features_are_shape_stable() -> None:
    features = higher_order_features(
        np.random.default_rng(4).normal(size=(64, 80)), load_contract()
    )
    assert {name: value.shape for name, value in features.items()} == {
        "marginal_quantiles": (15,),
        "local_rms_quantiles": (5,),
        "excursion_topology": (16,),
    }
    assert all(np.all(np.isfinite(value)) for value in features.values())


def test_pair_evaluation_is_exact_and_periodogram_matched() -> None:
    yy, xx = np.mgrid[:65, :81]
    clean = np.asarray((31 + xx + 2 * yy) % 220, dtype=np.uint8)
    noise = np.random.default_rng(12).normal(scale=6.0, size=clean.shape)
    grain = np.asarray(np.clip(np.rint(clean.astype(float) + noise), 0, 255), dtype=np.uint8)
    row = {"index": 17, "img_name": "fixture.png", "radius": 0.2}
    first, first_ratio = _evaluate_pair(row, clean, grain, load_contract())
    second, second_ratio = _evaluate_pair(row, clean, grain, load_contract())
    assert first == second
    assert first_ratio == second_ratio
    assert first["feature_shape"] == [64, 80]
    assert first["maximum_periodogram_relative_error"] < 1e-12
