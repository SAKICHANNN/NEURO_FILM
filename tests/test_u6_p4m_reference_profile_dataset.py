from __future__ import annotations

from pathlib import Path

from scripts.run_u6_p4m_reference_profile_dataset import (
    CONFIG_SHA256,
    ROOT,
)
from src.eval.physical_reference_profile_dataset import (
    evaluate_reference_datasets,
    generate_reference_dataset,
    load_contract,
)


CONFIG = ROOT / "configs/u6_p4m_reference_profile_dataset_v1.json"


def test_contract_is_exact_group_split_and_synthetic_only() -> None:
    contract, _ = load_contract(ROOT, CONFIG, CONFIG_SHA256)
    assert contract["dataset"]["photograph_or_scan_inputs_allowed"] is False
    assert contract["dataset"]["measured_film_statistics_allowed"] is False
    assert set(contract["dataset"]["splits"]) == {
        "development",
        "confirmation",
        "stress",
    }


def test_small_generation_is_repeat_exact_and_leakage_free(
    tmp_path: Path,
) -> None:
    contract, parent = load_contract(ROOT, CONFIG, CONFIG_SHA256)
    reduced = {
        **contract,
        "dataset": {
            **contract["dataset"],
            "coarse_shape": [12, 16],
            "field_families": ["flat", "checker"],
            "splits": {
                "development": [101],
                "confirmation": [503],
                "stress": [701],
            },
        },
        "automatic_gates": {
            **contract["automatic_gates"],
            "expected_record_count": 6,
            "expected_development_records": 2,
            "expected_confirmation_records": 2,
            "expected_stress_records": 2,
        },
    }
    first = generate_reference_dataset(reduced, parent, tmp_path / "a")
    second = generate_reference_dataset(reduced, parent, tmp_path / "b")
    report = evaluate_reference_datasets(
        reduced, [first, second], owned_temp_residue_count=0
    )
    assert report["automatic_pass"] is True
    assert all(report["checks"].values())


def test_group_overlap_is_rejected(tmp_path: Path) -> None:
    contract, parent = load_contract(ROOT, CONFIG, CONFIG_SHA256)
    reduced = {
        **contract,
        "dataset": {
            **contract["dataset"],
            "coarse_shape": [8, 10],
            "field_families": ["flat"],
            "splits": {
                "development": [101],
                "confirmation": [101],
                "stress": [701],
            },
        },
        "automatic_gates": {
            **contract["automatic_gates"],
            "expected_record_count": 3,
            "expected_development_records": 1,
            "expected_confirmation_records": 1,
            "expected_stress_records": 1,
        },
    }
    first = generate_reference_dataset(reduced, parent, tmp_path / "a")
    second = generate_reference_dataset(reduced, parent, tmp_path / "b")
    report = evaluate_reference_datasets(
        reduced, [first, second], owned_temp_residue_count=0
    )
    assert report["automatic_pass"] is False
    assert report["checks"]["group_split_disjoint"] is False
    assert report["checks"]["input_hash_split_disjoint"] is False
