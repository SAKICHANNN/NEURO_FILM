from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p280_hdrplus_result_pair_geometry_v1.json"


def test_p280_source_roles_are_exact() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    parent = ROOT / "docs/evidence/P279_HDRPLUS_MERGED_DNG_INGRESS_RESULT.json"
    assert (
        hashlib.sha256(parent.read_bytes()).hexdigest()
        == config["parent_p279_evidence_sha256"]
    )
    root = ROOT / config["source_root"]
    for row in config["rows"]:
        observation = root / row["observation"]
        target = root / row["target"]
        assert (
            hashlib.sha256(observation.read_bytes()).hexdigest()
            == row["observation_sha256"]
        )
        assert target.stat().st_size == row["target_bytes"]
        assert hashlib.sha256(target.read_bytes()).hexdigest() == row["target_sha256"]


def test_p280_representation_and_gates_are_frozen() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert (config["expected_width"], config["expected_height"]) == (4032, 3024)
    assert (config["downsample_width"], config["downsample_height"]) == (256, 192)
    assert config["shift_columns"] == 16
    assert config["gates"] == {
        "minimum_identity_correlation": 0.7,
        "minimum_flip_margin": 0.2,
        "minimum_shift_margin": 0.15,
        "required_row_count": 2,
    }


def test_p280_evidence_is_exact_when_present() -> None:
    path = ROOT / "docs/evidence/P280_HDRPLUS_RESULT_PAIR_GEOMETRY_RESULT.json"
    if not path.is_file():
        return
    evidence = json.loads(path.read_text(encoding="utf-8"))
    assert evidence["status"] in {
        "PASS_PRIVATE_HDRPLUS_RESULT_PAIR_GEOMETRY",
        "FAIL_CLOSED_HDRPLUS_RESULT_PAIR_GEOMETRY",
    }
    assert evidence["execution"]["forward_reverse_report_exact"] is True
