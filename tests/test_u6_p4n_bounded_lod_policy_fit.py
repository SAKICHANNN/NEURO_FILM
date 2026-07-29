from __future__ import annotations

import numpy as np

from scripts.run_u6_p4n_bounded_lod_policy_fit import (
    CONFIG_SHA256,
    ROOT,
)
from src.eval.physical_bounded_lod_policy_fit import (
    load_contract,
    summarize_candidate,
)


CONFIG = ROOT / "configs/u6_p4n_bounded_lod_policy_fit_v1.json"


def test_contract_binds_group_split_dataset_and_hard_policy() -> None:
    contract, _, manifest, _ = load_contract(
        ROOT, CONFIG, CONFIG_SHA256
    )
    assert contract["policy"]["selection_split"] == "development"
    assert contract["policy"]["sealed_splits"] == [
        "confirmation",
        "stress",
    ]
    assert contract["policy"]["continuous_or_dense_mixture_allowed"] is False
    assert len(manifest["records"]) == 48


def test_summary_is_exact_for_identity_and_detects_bias() -> None:
    reference = np.linspace(
        0.1, 1.5, 16 * 16 * 3, dtype=np.float32
    ).reshape(16, 16, 3)
    identity = summarize_candidate(
        reference,
        reference.copy(),
        fallback_fraction=0.0,
        block_size=8,
    )
    assert identity["global_mean_absolute_error"] == 0.0
    assert identity["block_mean_absolute_error"] == 0.0
    assert identity["minimum_variance_ratio"] == 1.0
    shifted = summarize_candidate(
        reference,
        reference + np.float32(0.1),
        fallback_fraction=0.25,
        block_size=8,
    )
    assert shifted["global_mean_absolute_error"] > 0.099
    assert shifted["block_mean_absolute_error"] > 0.099
    assert shifted["exact_fallback_fraction"] == 0.25
