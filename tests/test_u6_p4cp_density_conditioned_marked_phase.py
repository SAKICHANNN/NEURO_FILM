from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval import density_conditioned_marked_phase as p4cp

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4cp_density_conditioned_marked_phase_v1.json"
EVIDENCE = ROOT / "docs/evidence/U6_P4CP_DENSITY_CONDITIONED_MARKED_PHASE_RESULT.json"


def test_contract_freezes_source_observable_two_bin_conditioning() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert set(contract["development"]["source_names"]).isdisjoint(
        contract["confirmation"]["source_names"]
    )
    assert contract["patch_observation"]["density_split"] == 0.5
    assert contract["candidate"]["expected_parent_count_grid"] == [4, 8, 16, 32, 64, 128]
    assert contract["candidate"]["cluster_sigma_pixels"] == 1.0
    assert contract["confirmation"]["metadata_lock_sha256"] == (
        "3ee90ac134a40b271b1c69ca2b51dcdcfbca537fbea56a8377a9940f09d759ea"
    )


def test_contract_retains_tail_and_exact_spectrum_gates() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    gates = contract["confirmation_gates"]
    assert gates["minimum_median_source_improvement_over_fixed_p4co"] == 0.2
    assert gates["maximum_worst_source_distance_ratio_to_fixed_p4co"] == 1.0
    assert gates["require_exact_power_projection"] is True
    assert gates["require_exact_acf_projection"] is True


def test_density_patch_observation_is_repeat_exact() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    rng = np.random.default_rng(17)
    scalar = np.clip(0.4 + 0.01 * rng.standard_normal((384, 384)), 0.0, 1.0)
    first = p4cp._select_patch_observations(scalar, contract)
    second = p4cp._select_patch_observations(scalar, contract)
    assert [row["mean"] for row in first] == [row["mean"] for row in second]
    assert all(
        np.array_equal(left["residual"], right["residual"])
        for left, right in zip(first, second, strict=True)
    )


def test_scoring_uses_only_current_mean_for_bin_selection() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    rng = np.random.default_rng(23)
    residuals = []
    for _ in range(4):
        field = rng.standard_normal((96, 96))
        field -= field.mean()
        residuals.append(field / np.sqrt(np.mean(np.square(field))))
    observations = {
        "source": [
            {"mean": 0.2, "residual": residuals[0]},
            {"mean": 0.4, "residual": residuals[1]},
            {"mean": 0.6, "residual": residuals[2]},
            {"mean": 0.8, "residual": residuals[3]},
        ]
    }
    medians = {4: np.zeros(4), 64: np.ones(4), 128: np.full(4, 2.0)}
    rows, support = p4cp._score_confirmation(
        contract, observations, {"low": 4, "high": 128}, medians, np.ones(4)
    )
    assert support == {"low": 2, "high": 2}
    assert rows[0]["patch_count"] == 4


def test_evidence_closes_archive_brightness_conditioning() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["two_full_runs_byte_identical"] is True
    assert evidence["confirmation"]["sources_beating_fixed_p4co"] == 1
    assert evidence["gates"]["development_and_confirmation_bin_support"] is True
    assert evidence["gates"]["minimum_20_percent_median_source_improvement"] is False
    assert evidence["gates"]["worst_source_distance"] is False
    assert evidence["decision"] == "FAIL_CLOSED_DENSITY_CONDITIONED_MARKED_PHASE_TRANSFER"
