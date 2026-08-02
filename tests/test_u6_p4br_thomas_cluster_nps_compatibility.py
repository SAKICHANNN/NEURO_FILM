from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.eval.thomas_cluster_nps_compatibility import (
    ThomasClusterNPSCompatibilityError,
    _optimizer_result_is_usable,
    load_contract,
)
from src.film_physics.thomas_cluster_nps import (
    thomas_cluster_gaussian_mark_nps,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4br_thomas_cluster_nps_compatibility_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_p4br_contract_is_frozen_before_cluster_fit() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert payload["schema"] == (
        "neuro_film.u6_p4br_thomas_cluster_nps_compatibility_contract.v1"
    )
    assert payload["model"][
        "particle_sigma_micrometres_shared_per_material_bounds"
    ] == [0.1, 12.0]
    assert payload["model"]["cluster_sigma_micrometres_per_state_bounds"] == [0.1, 30.0]
    assert payload["model"]["mean_offspring_per_cluster_per_state_bounds"] == [
        0.01,
        100.0,
    ]
    assert payload["fit"]["build_bands_lines_per_mm"] == [[5.0, 100.0], [255.0, 350.0]]
    assert payload["fit"]["confirmation_bands_lines_per_mm"] == [
        [105.0, 250.0],
        [355.0, 500.0],
    ]
    assert (
        payload["fit"][
            "post_score_scale_multiplicity_frequency_split_model_or_gate_tuning_allowed"
        ]
        is False
    )


def test_p4br_contract_binds_primary_spectrum_and_closed_disc_family() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    source = payload["primary_model_source"]
    assert (
        _sha256(ROOT / source["official_vor_pdf_path"])
        == source["official_vor_pdf_sha256"]
    )
    parents = payload["parents"]
    for stem in ("p4bl_bundle", "p4bq_contract", "p4bq_decision", "p4bq_report"):
        assert _sha256(ROOT / parents[f"{stem}_path"]) == parents[f"{stem}_sha256"]


def test_thomas_cluster_spectrum_matches_structure_factor_endpoints() -> None:
    frequency = np.asarray([0.0, 100.0, 500.0])
    spectrum = thomas_cluster_gaussian_mark_nps(frequency, 0.001, 0.004, 3.0)
    assert spectrum[0] == 4.0
    assert np.all(np.diff(spectrum) < 0.0)
    assert np.all(spectrum > 0.0)


def test_p4br_contract_rejects_cluster_capacity_rescue(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["model"]["mean_offspring_per_cluster_per_state_bounds"] = [0.001, 1000.0]
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ThomasClusterNPSCompatibilityError, match="contract drift"):
        load_contract(path)


def test_p4br_accepts_only_exact_frozen_budget_exhaustion() -> None:
    exhausted = SimpleNamespace(
        x=np.asarray([1.0]),
        fun=0.1,
        success=False,
        nit=160,
        message="Maximum number of iterations has been exceeded.",
    )
    assert _optimizer_result_is_usable(exhausted, 160)
    exhausted.nit = 159
    assert not _optimizer_result_is_usable(exhausted, 160)
