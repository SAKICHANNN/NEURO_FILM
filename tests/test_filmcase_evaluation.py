from __future__ import annotations

import json
from pathlib import Path

from src.filmcase.evaluation import adjudicate, coverage_report, freeze_union_sources


def _source_index(root: Path) -> Path:
    image = root / "data" / "one.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"not-an-image-but-a-stable-fixture")
    index = root / "union.json"
    index.write_text(json.dumps([{"id": "01", "key": "one", "input": str(image)}]), encoding="utf-8")
    return index


def test_provisional_set_tracks_identity_and_reports_missing_coverage(tmp_path: Path) -> None:
    frozen = freeze_union_sources(_source_index(tmp_path), root=tmp_path, gold_ids=["01"], bucket_map={"01": ["sky"]})
    assert frozen["status"] == "provisional"
    assert frozen["samples"][0]["source_sha256"]
    coverage = coverage_report(frozen)
    assert coverage["gold_count"] == 1
    assert "face" in coverage["missing_required_gold_buckets"]


def test_provisional_or_uncertain_gold_never_promotes(tmp_path: Path) -> None:
    frozen = freeze_union_sources(_source_index(tmp_path), root=tmp_path, gold_ids=["01"], bucket_map={"01": ["sky"]})
    result = adjudicate(
        frozen,
        [{"candidate_id": "candidate", "sample_id": "01", "severity": "uncertain"}],
        candidate_id="candidate",
    )
    assert result.report["splits"]["gold"]["uncertain_count"] == 1
    assert not result.report["gold_promotion_pass"]


def test_final_complete_coverage_with_no_severe_can_promote() -> None:
    buckets = sorted(["face", "text_logo", "sky", "foliage", "high_key", "deep_shadow", "saturated_objects", "fine_detail"])
    frozen = {"status": "final", "samples": [{"id": "01", "split": "gold", "availability": "available", "buckets": buckets}, {"id": "02", "split": "stress", "availability": "available", "buckets": []}]}
    result = adjudicate(
        frozen,
        [{"candidate_id": "candidate", "sample_id": "01", "severity": "none"}, {"candidate_id": "candidate", "sample_id": "02", "severity": "moderate"}],
        candidate_id="candidate",
    )
    assert result.report["gold_promotion_pass"]
    assert result.report["splits"]["stress"]["severe_rate_wilson95"][1] > 0
