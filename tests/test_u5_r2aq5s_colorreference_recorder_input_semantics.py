from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.run_u5_r2aq5s_colorreference_recorder_input_semantics import (
    CONFIG_SHA256,
    ROOT,
    audit_tiff,
    load_config,
    run_audit,
)


CONFIG = (
    ROOT / "configs/u5_r2aq5s_colorreference_recorder_input_semantics_v1.json"
)


def test_frozen_config_and_source_semantics() -> None:
    config = load_config(CONFIG, CONFIG_SHA256)
    assert config["official_source"]["recorder_profile_or_characterization"] == (
        "unknown"
    )
    assert config["identification_gate"]["guess_srgb_or_scene_linear_allowed"] is (
        False
    )


def test_exact_sources_have_no_colour_characterization() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for source in config["source_files"]:
        facts = audit_tiff(ROOT / source["path"], source["sha256"])
        assert facts["mode"] == "RGB"
        assert facts["icc_profile_present"] is False
        assert facts["white_point_present"] is False
        assert facts["primary_chromaticities_present"] is False
        assert facts["transfer_function_present"] is False


def test_audit_closes_unidentified_mapping() -> None:
    report = run_audit(CONFIG, CONFIG_SHA256)
    assert report["all_required_tiff_facts_match"] is True
    assert report["identification_evidence_count"] == 0
    assert report["recorder_input_semantics_identified"] is False
    assert report["decision"] == (
        "close_recorder_photo_mapping_and_strength_tuning"
    )


def test_source_hash_drift_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.tif"
    source.write_bytes(b"not-a-tiff")
    wrong_hash = hashlib.sha256(b"other").hexdigest()
    with pytest.raises(ValueError, match="source hash mismatch"):
        audit_tiff(source, wrong_hash)
