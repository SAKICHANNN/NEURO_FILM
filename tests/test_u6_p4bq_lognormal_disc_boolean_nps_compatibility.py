from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4bq_lognormal_disc_boolean_nps_compatibility_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_p4bq_contract_is_frozen_before_variable_radius_fit() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert payload["schema"] == (
        "neuro_film.u6_p4bq_lognormal_disc_boolean_nps_compatibility_contract.v1"
    )
    model = payload["model"]
    assert model["median_radius_micrometres_bounds"] == [0.25, 12.0]
    assert model["log_radius_sigma_bounds"] == [0.05, 0.55]
    assert model["radius_cdf_interval"] == [0.00001, 0.99999]
    assert model["radius_distribution_quadrature_nodes"] == 32
    assert model["radial_hankel_quadrature_samples"] == 4097
    assert payload["fit"]["post_score_distribution_bound_quadrature_frequency_split_model_or_gate_tuning_allowed"] is False
    assert model["source_code_reuse_allowed"] is False


def test_p4bq_contract_binds_measured_bundle_and_closed_fixed_disc_parent() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    parents = payload["parents"]
    for stem in ("p4bl_bundle", "p4bp_contract", "p4bp_decision", "p4bp_report"):
        assert _sha256(ROOT / parents[f"{stem}_path"]) == parents[f"{stem}_sha256"]
