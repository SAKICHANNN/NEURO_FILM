from __future__ import annotations

import copy
import math

import pytest

from scripts.analyze_dpct_invocation_failure_signatures import (
    EXPECTED_KNOWN_IDS,
    FailureSignatureError,
    SAMPLE_IDS,
    analyze_progress,
)


def _invocation(seed: str, clipping: float) -> dict[str, object]:
    return {
        "clipping_fraction": clipping,
        "consumer_receipt_id": f"receipt-{seed}",
        "consumer_transform_id": f"transform-{seed}",
        "out_of_gamut_fraction": clipping,
        "producer_bundle_id": f"bundle-{seed}",
        "producer_diagnostics_id": f"diagnostics-{seed}",
        "producer_result_id": f"result-{seed}",
        "projected_fraction": 0.0,
        "request_id": f"request-{seed}",
        "response_id": f"response-{seed}",
    }


def _add_output_id(value: dict[str, object]) -> None:
    for group in ("known_rows", "photographic_rows"):
        rows = value[group]
        assert isinstance(rows, dict)
        for row in rows.values():
            row["invocation"]["output_pixel_sha256"] = "a" * 64
    rows = value["context_rows"]
    assert isinstance(rows, dict)
    for row in rows.values():
        for invocation in row["invocations"]:
            invocation["output_pixel_sha256"] = "b" * 64


def _progress() -> dict[str, object]:
    known: dict[str, object] = {}
    for index, row_id in enumerate(EXPECTED_KNOWN_IDS):
        source_error = 10.0 + index / 10.0
        candidate_error = source_error * (1.5 + index / 100.0)
        known[row_id] = {
            "invocation": _invocation(row_id, 0.001 + index / 10000.0),
            "metrics": {
                "candidate_new_boundary_fraction": 0.01 + index / 10000.0,
                "candidate_target_delta_e76_median": candidate_error,
                "candidate_target_delta_e76_p95": candidate_error + 2.0,
                "median_improvement_fraction": (
                    source_error - candidate_error
                )
                / source_error,
                "sample_id": row_id,
                "source_target_delta_e76_median": source_error,
                "source_target_delta_e76_p95": source_error + 1.0,
            },
        }
    photo: dict[str, object] = {}
    context: dict[str, object] = {}
    for index, sample_id in enumerate(SAMPLE_IDS):
        reasons = ["neutral-axis-chroma"]
        if index % 2:
            reasons.append("semantic-hue-rotation")
        if index == 0:
            reasons.append("new-boundary-fraction")
        photo[sample_id] = {
            "invocation": _invocation(f"photo-{sample_id}", 0.002),
            "metrics": {
                "neutral_chroma_p95": 20.0 + index,
                "new_boundary_fraction": 0.06 if index == 0 else 0.01,
                "passed": False,
                "reasons": reasons,
                "semantic_hue_rotation_p95_degrees": (
                    80.0 if index % 2 else 20.0
                ),
                "tone_largest_reversal_delta_l": 0.0,
                "tone_plateau_fraction": 0.0,
                "tone_reversal_fraction": 0.0,
            },
        }
        context[sample_id] = {
            "invocations": [
                _invocation(f"context-a-{sample_id}", 0.001),
                _invocation(f"context-b-{sample_id}", 0.002),
            ],
            "metrics": {
                "delta_e76_maximum": 30.0 + index,
                "delta_e76_median": 20.0 + index,
                "delta_e76_p95": 25.0 + index,
                "passed": False,
                "reasons": [
                    "shared-colour-median-drift",
                    "shared-colour-p95-drift",
                    "shared-colour-maximum-drift",
                ],
            },
        }
    return {
        "protocol": (
            "neuro-film.dpct-invocation-promotion-evaluation.v1"
        ),
        "contract_id": "contract-v1",
        "known_rows": known,
        "photographic_rows": photo,
        "context_rows": context,
    }


def test_analysis_is_deterministic_and_excludes_timing_bound_ids() -> None:
    first_input = _progress()
    second_input = copy.deepcopy(first_input)
    for group in ("known_rows", "photographic_rows"):
        rows = second_input[group]
        assert isinstance(rows, dict)
        for row in rows.values():
            row["invocation"]["producer_diagnostics_id"] += "-timing"
            row["invocation"]["producer_result_id"] += "-timing"
            row["invocation"]["consumer_receipt_id"] += "-timing"
            row["invocation"]["response_id"] += "-timing"
    rows = second_input["context_rows"]
    assert isinstance(rows, dict)
    for row in rows.values():
        for invocation in row["invocations"]:
            invocation["producer_diagnostics_id"] += "-timing"
            invocation["producer_result_id"] += "-timing"
            invocation["consumer_receipt_id"] += "-timing"
            invocation["response_id"] += "-timing"

    first = analyze_progress(first_input)
    second = analyze_progress(second_input)
    assert first == second
    assert first["stable_evidence_id"] == second["stable_evidence_id"]


def test_analysis_identifies_discriminating_failure_signatures() -> None:
    report = analyze_progress(_progress())
    known = report["known_operator"]
    assert known["row_count"] == 30
    assert known["regressed_count"] == 30
    assert known["candidate_error_exceeds_source_count"] == 30
    assert known["regressed_with_clipping_at_most_5pct_count"] == 30
    assert known["unique_bundle_count"] == 30
    assert set(known["by_source"]) == set(SAMPLE_IDS)
    assert set(known["by_reference"]) == set(SAMPLE_IDS)
    assert all(
        row["row_count"] == 5 for row in known["by_source"].values()
    )
    assert report["photographic"]["failed_count"] == 6
    assert report["context"]["changed_bundle_pair_count"] == 6
    assert report["failure_signatures"] == {
        "universal_global_overcorrection": True,
        "regression_predominantly_not_explained_by_large_clipping": True,
        "source_context_changes_every_context_bundle": True,
        "neutral_axis_corruption_present": True,
        "semantic_hue_corruption_present": True,
        "boundary_corruption_present": True,
        "all_context_probes_fail": True,
    }


def test_current_output_pixel_identity_is_validated_but_not_hashed() -> None:
    legacy = _progress()
    current = copy.deepcopy(legacy)
    _add_output_id(current)

    assert analyze_progress(current) == analyze_progress(legacy)

    current["known_rows"][EXPECTED_KNOWN_IDS[0]]["invocation"][
        "output_pixel_sha256"
    ] = "not-a-sha"
    with pytest.raises(FailureSignatureError):
        analyze_progress(current)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-row",
        "sample-mismatch",
        "improvement-mismatch",
        "nonfinite",
        "extra-field",
        "context-count",
        "pass-reason-mismatch",
    ],
)
def test_incomplete_or_inconsistent_progress_fails_closed(
    mutation: str,
) -> None:
    value = _progress()
    if mutation == "missing-row":
        value["known_rows"].pop(EXPECTED_KNOWN_IDS[0])
    elif mutation == "sample-mismatch":
        value["known_rows"][EXPECTED_KNOWN_IDS[0]]["metrics"][
            "sample_id"
        ] = "foreign"
    elif mutation == "improvement-mismatch":
        value["known_rows"][EXPECTED_KNOWN_IDS[0]]["metrics"][
            "median_improvement_fraction"
        ] = 0.5
    elif mutation == "nonfinite":
        value["known_rows"][EXPECTED_KNOWN_IDS[0]]["metrics"][
            "candidate_target_delta_e76_median"
        ] = math.nan
    elif mutation == "extra-field":
        value["known_rows"][EXPECTED_KNOWN_IDS[0]]["extra"] = True
    elif mutation == "context-count":
        value["context_rows"][SAMPLE_IDS[0]]["invocations"].pop()
    elif mutation == "pass-reason-mismatch":
        value["photographic_rows"][SAMPLE_IDS[0]]["metrics"][
            "passed"
        ] = True
    with pytest.raises(FailureSignatureError):
        analyze_progress(value)
