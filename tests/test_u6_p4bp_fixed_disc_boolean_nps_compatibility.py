from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4bp_fixed_disc_boolean_nps_compatibility_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_p4bp_contract_is_frozen_before_measured_spectrum_fit() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert payload["schema"] == (
        "neuro_film.u6_p4bp_fixed_disc_boolean_nps_compatibility_contract.v1"
    )
    fit = payload["fit"]
    assert fit["build_bands_lines_per_mm"] == [[5.0, 100.0], [255.0, 350.0]]
    assert fit["confirmation_bands_lines_per_mm"] == [
        [105.0, 250.0],
        [355.0, 500.0],
    ]
    assert fit["candidate_radius_micrometres_bounds"] == [0.25, 20.0]
    assert fit["candidate_coverage_probability_bounds"] == [0.02, 0.98]
    assert fit["post_score_radius_coverage_frequency_split_model_or_gate_tuning_allowed"] is False
    assert payload["primary_model_source"]["source_code_reuse_allowed"] is False


def test_p4bp_contract_binds_primary_paper_and_measured_parent() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    source = payload["primary_model_source"]
    assert _sha256(ROOT / source["article_low_resolution_pdf_path"]) == source[
        "article_low_resolution_pdf_sha256"
    ]
    assert _sha256(ROOT / source["official_source_archive_path"]) == source[
        "official_source_archive_sha256"
    ]
    parents = payload["parents"]
    for stem in ("p4bl_contract", "p4bl_decision", "p4bl_bundle", "p4bl_report"):
        assert _sha256(ROOT / parents[f"{stem}_path"]) == parents[f"{stem}_sha256"]
