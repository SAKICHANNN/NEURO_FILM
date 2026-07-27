from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import numpy as np
import pytest

from src.color_match import (
    CoreNumericGuardPolicyV1,
    MATCH_PROFILE_DISPLAY_SRGB,
    PreparedMatchViewV1,
    ReferenceMatchContractError,
    make_match_view,
)
from src.color_match.shared_numeric_guard import (
    guard_shared_numeric_batch_v1,
    make_shared_apply_numeric_facts_v1,
    shared_numeric_batch_guard_from_json,
    shared_numeric_batch_guard_to_json,
    validate_shared_numeric_batch_guard_v1,
)
from src.color_match.shared_operator_batch import (
    make_shared_reference_operator_v1,
    prepare_shared_operator_apply_v1,
    resolve_shared_operator_batch_v1,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_shared_numeric_batch_guard_v1.schema.json"
)


def _prepared(seed: float) -> PreparedMatchViewV1:
    pixels = np.full((2, 3, 3), seed, dtype=np.float32)
    pixels = np.ascontiguousarray(pixels)
    wire = pixels.astype(">f4", copy=False).tobytes()
    descriptor = make_match_view(
        profile_id=MATCH_PROFILE_DISPLAY_SRGB,
        pixel_sha256=hashlib.sha256(wire).hexdigest(),
        shape=pixels.shape,
        render_bridge_id="neuro-film.shared-numeric-test.v1",
        provenance_fingerprint=hashlib.sha256(
            str(seed).encode()
        ).hexdigest(),
    )
    pixels.flags.writeable = False
    return PreparedMatchViewV1(descriptor=descriptor, pixels=pixels)


def _fixture(*, second_clip: float = 0.01, second_output: float = 0.41):
    reference = _prepared(0.2)
    sources = (_prepared(0.3), _prepared(0.4))
    operator = make_shared_reference_operator_v1(
        reference=reference,
        compatibility_profile_id="neuro-film.future-shared.v1",
        capability_id="zhuise.rgin.cpu-reference.v1",
        producer_commit="a" * 40,
        producer_bundle_id="sha256:" + "b" * 64,
        producer_reference_view_id="sha256:" + "c" * 64,
        model_fingerprint="d" * 64,
        options_sha256="e" * 64,
    )
    first_pixels = np.full((2, 3, 3), 0.31, dtype=np.float32)
    second_pixels = np.full(
        (2, 3, 3), second_output, dtype=np.float32
    )
    applies = (
        prepare_shared_operator_apply_v1(
            operator=operator,
            source_index=0,
            source=sources[0],
            producer_source_view_id="sha256:" + "1" * 64,
            producer_apply_result_id="sha256:" + "2" * 64,
            diagnostics_id="sha256:" + "3" * 64,
            output_pixels=first_pixels,
        ),
        prepare_shared_operator_apply_v1(
            operator=operator,
            source_index=1,
            source=sources[1],
            producer_source_view_id="sha256:" + "4" * 64,
            producer_apply_result_id="sha256:" + "5" * 64,
            diagnostics_id="sha256:" + "6" * 64,
            output_pixels=second_pixels,
        ),
    )
    batch = resolve_shared_operator_batch_v1(
        operator=operator,
        reference=reference,
        sources=sources,
        applies=applies,
    )
    facts = (
        make_shared_apply_numeric_facts_v1(
            prepared=applies[0],
            producer_diagnostics_id=applies[0].receipt.diagnostics_id,
            all_finite=True,
            output_minimum=float(np.min(applies[0].pixels)),
            output_maximum=float(np.max(applies[0].pixels)),
            out_of_gamut_fraction=0.01,
            clipping_fraction=0.01,
            projected_fraction=0.0,
        ),
        make_shared_apply_numeric_facts_v1(
            prepared=applies[1],
            producer_diagnostics_id=applies[1].receipt.diagnostics_id,
            all_finite=True,
            output_minimum=float(np.min(applies[1].pixels)),
            output_maximum=float(np.max(applies[1].pixels)),
            out_of_gamut_fraction=second_clip,
            clipping_fraction=second_clip,
            projected_fraction=0.0,
        ),
    )
    return sources, applies, batch, facts


def test_all_sources_pass_one_policy_atomically() -> None:
    sources, applies, batch, facts = _fixture()
    result = guard_shared_numeric_batch_v1(
        batch=batch,
        sources=sources,
        applies=applies,
        facts=facts,
    )
    assert result.atomic_state == "eligible-for-transaction"
    assert [item.source_index for item in result.decisions] == [0, 1]
    assert all(item.accepted_for_transaction for item in result.decisions)
    assert all(
        item.action == "eligible-for-transaction"
        for item in result.decisions
    )
    assert "applied" not in shared_numeric_batch_guard_to_json(result)


def test_one_source_failure_forces_whole_batch_fallback() -> None:
    sources, applies, batch, facts = _fixture(second_clip=0.06)
    result = guard_shared_numeric_batch_v1(
        batch=batch,
        sources=sources,
        applies=applies,
        facts=facts,
    )
    assert result.atomic_state == "identity-fallback"
    assert result.decisions[0].accepted_for_transaction
    assert not result.decisions[1].accepted_for_transaction
    assert result.decisions[1].reasons == ("clipping-fraction",)


def test_new_boundary_is_measured_from_exact_pixels() -> None:
    sources, applies, batch, facts = _fixture(second_output=1.0)
    result = guard_shared_numeric_batch_v1(
        batch=batch,
        sources=sources,
        applies=applies,
        facts=facts,
    )
    assert result.atomic_state == "identity-fallback"
    assert result.decisions[1].new_boundary_fraction == 1.0
    assert "new-boundary-fraction" in result.decisions[1].reasons


def test_roundtrip_and_repeat_are_exact() -> None:
    sources, applies, batch, facts = _fixture()
    kwargs = {
        "batch": batch,
        "sources": sources,
        "applies": applies,
        "facts": facts,
    }
    first = guard_shared_numeric_batch_v1(**kwargs)
    second = guard_shared_numeric_batch_v1(**kwargs)
    assert first == second
    encoded = shared_numeric_batch_guard_to_json(first)
    assert shared_numeric_batch_guard_from_json(encoded) == first
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(json.loads(encoded))


@pytest.mark.parametrize(
    "mutation",
    [
        "diagnostics",
        "receipt",
        "extrema",
        "fraction",
        "nonfinite",
        "missing",
        "reordered",
    ],
)
def test_facts_and_inventory_mutations_fail_closed(mutation: str) -> None:
    sources, applies, batch, facts = _fixture()
    if mutation == "diagnostics":
        facts = (
            replace(
                facts[0],
                producer_diagnostics_id="sha256:" + "9" * 64,
            ),
            facts[1],
        )
    elif mutation == "receipt":
        facts = (replace(facts[0], receipt_id="0" * 64), facts[1])
    elif mutation == "extrema":
        facts = (replace(facts[0], output_maximum=0.9), facts[1])
    elif mutation == "fraction":
        facts = (replace(facts[0], clipping_fraction=1.1), facts[1])
    elif mutation == "nonfinite":
        facts = (replace(facts[0], output_minimum=np.nan), facts[1])
    elif mutation == "missing":
        facts = (facts[0],)
    elif mutation == "reordered":
        facts = (facts[1], facts[0])
    with pytest.raises(ReferenceMatchContractError):
        guard_shared_numeric_batch_v1(
            batch=batch,
            sources=sources,
            applies=applies,
            facts=facts,
        )


def test_policy_and_decision_identity_mutation_fail_closed() -> None:
    sources, applies, batch, facts = _fixture()
    result = guard_shared_numeric_batch_v1(
        batch=batch,
        sources=sources,
        applies=applies,
        facts=facts,
    )
    changed_policy = replace(
        result.policy, max_clipping_fraction=0.02
    )
    with pytest.raises(ReferenceMatchContractError):
        validate_shared_numeric_batch_guard_v1(
            replace(result, policy=changed_policy)
        )
    with pytest.raises(
        ReferenceMatchContractError,
        match="identity mismatch",
    ):
        validate_shared_numeric_batch_guard_v1(
            replace(result, guard_batch_id="0" * 64)
        )


def test_unknown_json_field_and_mixed_policy_fail_closed() -> None:
    sources, applies, batch, facts = _fixture()
    result = guard_shared_numeric_batch_v1(
        batch=batch,
        sources=sources,
        applies=applies,
        facts=facts,
        policy=CoreNumericGuardPolicyV1(),
    )
    payload = json.loads(shared_numeric_batch_guard_to_json(result))
    payload["unknown"] = True
    with pytest.raises(ReferenceMatchContractError, match="fields differ"):
        shared_numeric_batch_guard_from_json(json.dumps(payload))
