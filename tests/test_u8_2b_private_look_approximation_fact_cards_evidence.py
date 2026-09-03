from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.historical_evidence_binding import assert_historical_evidence_binding

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U8_2B_PRIVATE_LOOK_APPROXIMATION_FACT_CARDS_RESULT.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_u8_2b_evidence_binds_reports_and_published_artifact() -> None:
    from src.inference.product_fact_cards import validate_product_fact_cards

    evidence = json.loads(EVIDENCE.read_text("utf-8"))
    reports = [ROOT / row["path"] for row in evidence["formal_execution"]["reports"]]
    assert reports[0].read_bytes() == reports[1].read_bytes()
    assert all(path.stat().st_size == 3002 for path in reports)
    assert all(_sha256(path) == evidence["formal_execution"]["reports"][0]["sha256"] for path in reports)
    report = json.loads(reports[0].read_text("utf-8"))
    assert report["status"] == evidence["status"]
    assert all(report["gates"].values())
    artifact = ROOT / evidence["published_artifact"]["path"]
    assert artifact.stat().st_size == evidence["published_artifact"]["bytes"]
    assert _sha256(artifact) == evidence["published_artifact"]["sha256"]
    bundle = json.loads(artifact.read_text("utf-8"))
    config = json.loads((ROOT / "configs/u8_2b_private_look_approximation_fact_cards_v1.json").read_text("utf-8"))
    validate_product_fact_cards(bundle, config=config)


def test_u8_2b_evidence_preserves_claim_and_availability_boundaries() -> None:
    evidence = json.loads(EVIDENCE.read_text("utf-8"))
    assert all(evidence["gates"].values())
    assert evidence["facts"]["available_colour_looks"] == [
        "velvia_50",
        "portra_400",
        "ektar_100",
    ]
    assert evidence["facts"]["generic_bw_availability"] == "blocked_severe_artifact"
    assert evidence["facts"]["root_license"] == "unresolved"
    assert evidence["claim_ceiling"]["calibrated_stock_response"] is False
    assert evidence["claim_ceiling"]["physical_film_reproduction"] is False
    assert evidence["claim_ceiling"]["public_release"] is False


def test_u8_2b_evidence_binds_committed_sources() -> None:
    evidence = json.loads(EVIDENCE.read_text("utf-8"))
    for relative, binding in evidence["source_bindings"].items():
        assert_historical_evidence_binding(ROOT, {"path": relative, **binding})
