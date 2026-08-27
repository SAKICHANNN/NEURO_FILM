from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p279_hdrplus_merged_dng_ingress_v1.json"


def test_p279_exact_parent_and_source_lock() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    parent = ROOT / "docs/evidence/P269_HDRPLUS_ONE_BURST_ACQUISITION_RESULT.json"
    assert (
        hashlib.sha256(parent.read_bytes()).hexdigest()
        == config["parent_p269_evidence_sha256"]
    )
    root = ROOT / config["source_root"]
    for row in config["rows"]:
        path = root / row["path"]
        assert path.stat().st_size == row["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]


def test_p279_dimensions_and_roles_are_frozen() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert (config["expected_full_width"], config["expected_full_height"]) == (
        4032,
        3024,
    )
    assert (
        config["known_metadata_page_width"],
        config["known_metadata_page_height"],
    ) == (640, 480)
    assert [row["version"] for row in config["rows"]] == ["20161014", "20171023"]


def test_p279_evidence_is_exact_when_present() -> None:
    path = ROOT / "docs/evidence/P279_HDRPLUS_MERGED_DNG_INGRESS_RESULT.json"
    if not path.is_file():
        return
    evidence = json.loads(path.read_text(encoding="utf-8"))
    assert evidence["status"] in {
        "PASS_PRIVATE_HDRPLUS_MERGED_DNG_INGRESS_SAFETY",
        "FAIL_CLOSED_HDRPLUS_MERGED_DNG_INGRESS_SAFETY",
    }
    assert evidence["execution"]["forward_reverse_report_exact"] is True
