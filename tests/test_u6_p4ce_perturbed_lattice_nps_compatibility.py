from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.perturbed_lattice_nps_compatibility import (
    PerturbedLatticeNPSCompatibilityError,
    load_contract,
)
from src.film_physics.perturbed_lattice_nps import (
    perturbed_lattice_gaussian_mark_nps,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4ce_perturbed_lattice_nps_compatibility_v1.json"
DECISION = ROOT / "configs/u6_p4ce_perturbed_lattice_nps_compatibility_decision_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_p4ce_contract_is_frozen_before_fit() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert payload["model"]["bragg_atoms_modeled"] is False
    assert payload["model"]["jitter_sigma_micrometres_per_state_bounds"] == [
        0.05,
        50.0,
    ]
    assert payload["fit"]["build_bands_lines_per_mm"] == [
        [5.0, 100.0],
        [255.0, 350.0],
    ]
    assert payload["fit"]["confirmation_bands_lines_per_mm"] == [
        [105.0, 250.0],
        [355.0, 500.0],
    ]
    assert (
        payload["fit"][
            "post_score_scale_jitter_frequency_split_model_or_gate_tuning_allowed"
        ]
        is False
    )


def test_p4ce_contract_binds_source_bundle_and_thomas_control() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    source = payload["primary_model_source"]
    assert _sha256(ROOT / source["paper_path"]) == source["paper_sha256"]
    parents = payload["parents"]
    for stem in ("p4bl_bundle", "p4br_report"):
        assert _sha256(ROOT / parents[f"{stem}_path"]) == parents[f"{stem}_sha256"]


def test_perturbed_lattice_diffuse_spectrum_has_expected_limits() -> None:
    frequency = np.asarray([0.0, 10.0, 50.0, 500.0])
    spectrum = perturbed_lattice_gaussian_mark_nps(frequency, 0.001, 0.004)
    assert spectrum[0] == 0.0
    assert np.all(spectrum[1:] > 0.0)
    assert spectrum[1] < spectrum[2]
    assert spectrum[-1] < spectrum[2]


def test_p4ce_rejects_dependent_field_rescue(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["model"]["jitter_sigma_micrometres_per_state_bounds"] = [0.01, 100.0]
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PerturbedLatticeNPSCompatibilityError, match="contract drift"):
        load_contract(path)


def test_p4ce_decision_binds_exact_reports() -> None:
    payload = json.loads(DECISION.read_text(encoding="utf-8"))
    assert payload["report_sha256"] == payload["repeat_report_sha256"]
    assert payload["decision"] in {
        "retain_perturbed_lattice_diffuse_historical_nps_baseline",
        "close_independent_perturbed_lattice_diffuse_nps_family",
    }
