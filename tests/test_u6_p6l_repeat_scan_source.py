from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from src.eval.physical_repeat_scan_source import (
    RepeatScanSourceError,
    audit_repeat_scan_source,
    load_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p6l_uchicago_repeat_scan_source_v1.json"


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324


def _fixture(tmp_path: Path) -> tuple[dict, Path]:
    root = tmp_path / "root"
    destination = root / "data/repeats"
    destination.mkdir(parents=True)
    yy, xx = np.mgrid[:96, :112]
    base = np.asarray(
        5000 + 70 * xx + 40 * yy + 2000 * ((xx // 9 + yy // 11) % 2),
        dtype=np.uint16,
    )
    images = [
        base,
        np.roll(base, shift=(1, -2), axis=(0, 1)),
        np.roll(base, shift=(-1, 2), axis=(0, 1)),
    ]
    rows = []
    for index, image in enumerate(images, start=1):
        name = f"repeat{index}.tif"
        path = destination / name
        tifffile.imwrite(path, image)
        rows.append(
            {
                "name": name,
                "expected_bytes": path.stat().st_size,
                "expected_md5": _md5(path),
                "url": (
                    "https://knowledge.uchicago.edu/api/records/"
                    f"qw505-gq084/files/{name}/content"
                ),
                "role": f"same_plate_repeat_{index}",
            }
        )
    metadata = destination / "README.txt"
    metadata.write_text("fixture\n", encoding="utf-8")
    rows.append(
        {
            "name": metadata.name,
            "expected_bytes": metadata.stat().st_size,
            "expected_md5": _md5(metadata),
            "url": (
                "https://knowledge.uchicago.edu/api/records/"
                "qw505-gq084/files/README.txt/content"
            ),
            "role": "source_metadata",
        }
    )
    config = json.loads(CONTRACT.read_text(encoding="utf-8"))
    config["experiment_id"] = "fixture"
    config["acquisition"]["destination"] = "data/repeats"
    config["acquisition"]["files"] = rows
    config["acquisition"]["maximum_total_bytes"] = sum(
        row["expected_bytes"] for row in rows
    )
    config["source_gate"]["minimum_registered_pair_correlation"] = 0.99
    return config, root


def test_contract_is_exact_and_bounded() -> None:
    contract = load_contract(CONTRACT)
    assert contract["acquisition"]["maximum_total_bytes"] == 88_475_062
    assert len(contract["acquisition"]["files"]) == 4


def test_repeat_scan_audit_registers_and_repeats(tmp_path: Path) -> None:
    config, root = _fixture(tmp_path)
    first = audit_repeat_scan_source(config, root)
    second = audit_repeat_scan_source(config, root)
    assert first == second
    assert first["source_pass"]
    assert first["decision"] == "open_repeat_residual_analysis"
    assert len(first["registration"]) == 2
    assert first["minimum_registered_pair_correlation"] >= 0.99


def test_repeat_scan_audit_rejects_integrity_drift(tmp_path: Path) -> None:
    config, root = _fixture(tmp_path)
    target = root / "data/repeats/repeat2.tif"
    target.write_bytes(target.read_bytes() + b"x")
    with pytest.raises(
        RepeatScanSourceError, match="integrity mismatch"
    ):
        audit_repeat_scan_source(config, root)
