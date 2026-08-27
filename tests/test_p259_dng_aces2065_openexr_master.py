from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.audit_p259_dng_aces2065_openexr_master import _bound

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p259_dng_aces2065_openexr_master_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_p259_rows_are_exactly_the_existing_p98_and_p257_rows() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    p98 = json.loads(
        (ROOT / config["bindings"]["p98_config_path"]).read_text(encoding="utf-8")
    )
    p257 = json.loads(
        (ROOT / config["bindings"]["p257_config_path"]).read_text(encoding="utf-8")
    )
    actual = {row["source_id"]: row for row in config["rows"]}
    assert set(actual) == {row["source_id"] for row in p98["rows"]}
    assert set(actual) == {row["source_id"] for row in p257["rows"]}
    for row in p98["rows"]:
        frozen = actual[row["source_id"]]
        assert frozen["logical_path"] == row["logical_path"]
        assert frozen["source_bytes"] == row["source_bytes"]
        assert frozen["source_sha256"] == row["source_sha256"]
    for row in p257["rows"]:
        assert (
            actual[row["source_id"]]["p98_working_float32_sha256"]
            == row["working_float32_sha256"]
        )


def test_p259_ap1_ap0_matrix_is_the_exact_p249_matrix() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    p249 = json.loads(
        (ROOT / config["bindings"]["p249_config_path"]).read_text(encoding="utf-8")
    )
    assert np.array_equal(
        np.asarray(config["transform"]["ap1_to_ap0_matrix"], dtype=np.float64),
        np.asarray(p249["expected"]["ap1_to_ap0_matrix"], dtype=np.float64),
    )


def test_p259_bound_source_files_are_present_and_immutable_before_formal() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for row in config["rows"]:
        path = ROOT / row["logical_path"]
        assert path.stat().st_size == row["source_bytes"]
        assert _sha256(path) == row["source_sha256"]


def test_p259_report_binding_keeps_repo_relative_junction_path() -> None:
    source = ROOT / "data/external/rawpixls_bh1_v1/raw/lg_lg_h850.dng"
    assert _bound(source)["path"] == source.relative_to(ROOT).as_posix()
