from __future__ import annotations

import numpy as np

from src.eval.fivek_source_hard_retrieval import (
    evaluate_development_family,
    source_descriptor,
)


def _descriptor_spec() -> dict:
    return {
        "spatial_grid": 4,
        "luma_quantiles": 17,
        "gradient_magnitude_quantiles": 9,
    }


def test_source_descriptors_are_finite_and_have_fixed_family_shapes() -> None:
    yy, xx = np.mgrid[:32, :48]
    rgb = np.stack(
        [xx / 47, yy / 31, ((xx + yy) % 17) / 16], axis=-1
    )
    global_features = source_descriptor(
        rgb, "global_photometric", _descriptor_spec()
    )
    spatial_features = source_descriptor(
        rgb, "spatial_photometric", _descriptor_spec()
    )
    tone_features = source_descriptor(rgb, "tone_layout", _descriptor_spec())
    assert np.all(np.isfinite(global_features))
    assert np.all(np.isfinite(spatial_features))
    assert np.all(np.isfinite(tone_features))
    assert len(spatial_features) > len(global_features)
    assert len(tone_features) == 17 + 2 * 16 + 9


def test_whole_group_hard_retrieval_recovers_repeated_source_types() -> None:
    rows = []
    ids = []
    groups = []
    styles = []
    for group_index in range(3):
        for style in range(2):
            pair_id = f"g{group_index}-s{style}"
            value = 0.2 if style == 0 else 0.8
            image = np.full((16, 16, 3), value, dtype=np.float64)
            image[:, :, 1] += np.linspace(0.0, 0.05, 16)[:, None]
            rows.append(
                {
                    "pair_id": pair_id,
                    "group": f"g{group_index}",
                    "source": image,
                }
            )
            ids.append(pair_id)
            groups.append(f"g{group_index}")
            styles.append(style)
    error_matrix = np.asarray(
        [
            [0.1 if styles[q] == styles[c] else 0.5 for c in range(6)]
            for q in range(6)
        ],
        dtype=np.float64,
    )
    prepared = {
        "ids": ids,
        "groups": np.asarray(groups, dtype=object),
        "error_matrix": error_matrix,
        "global_errors": np.full(6, 0.4),
        "oracle_errors": np.full(6, 0.1),
        "random_errors": np.full(6, 0.5),
    }
    gates = {
        "minimum_mean_improvement_over_global": 0.1,
        "minimum_win_fraction_over_global": 0.8,
        "maximum_p95_ratio_to_global": 1.0,
        "maximum_worst_ratio_to_global": 1.0,
        "minimum_oracle_gap_closure": 0.5,
        "minimum_mean_improvement_over_random_case": 0.1,
        "minimum_win_fraction_over_random_case": 0.8,
        "minimum_bootstrap_lower_improvement": 0.0,
        "maximum_fallback_fraction": 1.0,
        "minimum_distinct_selected_cases": 2,
        "maximum_selected_case_share": 0.5,
    }
    report = evaluate_development_family(
        rows=rows,
        prepared=prepared,
        family="global_photometric",
        descriptor_spec=_descriptor_spec(),
        selector_spec={
            "ood_distance_quantile": 1.0,
            "bootstrap_seed": 7,
            "bootstrap_repetitions": 1000,
        },
        gates=gates,
    )
    assert report["automatic_pass"] is True
    assert report["metrics"]["mean_improvement_over_global"] > 0.7
    assert all(not row["fallback"] for row in report["rows"])
    assert all(row["pair_id"] != row["selected_case_id"] for row in report["rows"])
