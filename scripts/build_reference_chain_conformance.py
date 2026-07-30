"""Build exact P28-P30 consumer-chain conformance vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.color_match import (
    MATCH_PROFILE_DISPLAY_SRGB,
    CoreNumericGuardPolicyV1,
    PreparedMatchViewV1,
    PromotionDecision,
    adapt_dpct_candidate_v2,
    admit_core_apply_receipt,
    adjudicate_core_acceptance,
    authorize_core_product_staging_v1,
    guard_core_candidate_numeric_v1,
    guard_core_numeric_batch_v1,
    make_match_view,
    resolve_dpct_batch_v1,
)
from src.color_match.canonical import canonical_bytes


PRODUCER_FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "zhuise_producer_contract_exact_bits_v2.json"
)
PROTOCOL = "neuro-film.reference-product-chain-conformance.v1"


def _prepared(
    encoded_hex: str,
    provenance: str,
) -> PreparedMatchViewV1:
    wire = bytes.fromhex(encoded_hex)
    pixels = np.frombuffer(wire, dtype=">f4").astype(np.float32, copy=True)
    pixels = np.ascontiguousarray(pixels.reshape(2, 2, 3))
    pixels.flags.writeable = False
    descriptor = make_match_view(
        profile_id=MATCH_PROFILE_DISPLAY_SRGB,
        pixel_sha256=hashlib.sha256(wire).hexdigest(),
        shape=(2, 2, 3),
        render_bridge_id="neuro-film.chain-conformance.v1",
        provenance_fingerprint=provenance * 64,
    )
    return PreparedMatchViewV1(descriptor=descriptor, pixels=pixels)


def _case(
    fixture: dict[str, Any],
    *,
    case_id: str,
    second_research_override: bool,
) -> dict[str, Any]:
    reference = _prepared(fixture["reference_pixel_f32be_hex"], "a")
    items = []
    for index, provenance in enumerate(("b", "c")):
        research = index == 1 and second_research_override
        source = _prepared(
            fixture["source_pixel_f32be_hex"],
            provenance,
        )
        candidate = adapt_dpct_candidate_v2(
            source=source,
            reference=reference,
            producer_source=fixture["source"],
            producer_reference=fixture["reference"],
            producer_transform=fixture["transform"],
            producer_transform_payload=bytes.fromhex(
                fixture["payload_hex"]
            ),
            producer_diagnostics=fixture["diagnostics"],
            producer_apply_result=fixture["apply_result"],
            output_pixel_f32be=bytes.fromhex(
                fixture["output_pixel_f32be_hex"]
            ),
            intent_id=str(index + 1) * 64,
        )
        promotion = (
            PromotionDecision("rejected", ("research-only",))
            if research
            else PromotionDecision("promoted", ())
        )
        acceptance = adjudicate_core_acceptance(
            source=source.descriptor,
            reference=reference.descriptor,
            transform=candidate.transform,
            capabilities=candidate.capabilities,
            diagnostics=candidate.diagnostics,
            promotion=promotion,
            allow_research_baseline=research,
        )
        admission = admit_core_apply_receipt(
            prepared=candidate.prepared_output,
            acceptance=acceptance,
            source=source.descriptor,
            reference=reference.descriptor,
            transform=candidate.transform,
            capabilities=candidate.capabilities,
            diagnostics=candidate.diagnostics,
        )
        items.append((source, candidate, acceptance, admission))
    batch = resolve_dpct_batch_v1(
        reference=reference,
        sources=tuple(item[0] for item in items),
        outcomes=tuple(item[1] for item in items),
        adjudications=tuple((item[2], item[3]) for item in items),
    )
    policy = CoreNumericGuardPolicyV1(
        max_out_of_gamut_fraction=1.0,
        max_clipping_fraction=1.0,
        max_projected_fraction=1.0,
        max_new_boundary_fraction=1.0,
    )
    decisions = tuple(
        guard_core_candidate_numeric_v1(
            source=item[0],
            reference=reference,
            candidate=item[1],
            acceptance=item[2],
            admission=item[3],
            policy=policy,
        )
        for item in items
    )
    numeric_batch = guard_core_numeric_batch_v1(
        batch=batch,
        decisions=decisions,
    )
    authorization = authorize_core_product_staging_v1(
        batch=batch,
        numeric_guard=numeric_batch,
        acceptances=tuple(item[2] for item in items),
    )
    records = [
        ("batch", batch.batch_id, batch.to_dict(), "batch_id"),
        *[
            (
                f"numeric-decision-{index}",
                decision.decision_id,
                decision.to_dict(),
                "decision_id",
            )
            for index, decision in enumerate(decisions)
        ],
        (
            "numeric-batch",
            numeric_batch.guard_batch_id,
            numeric_batch.to_dict(),
            "guard_batch_id",
        ),
        (
            "authorization",
            authorization.authorization_id,
            authorization.to_dict(),
            "authorization_id",
        ),
    ]
    identities = []
    for label, expected, payload, identity_key in records:
        identity_payload = dict(payload)
        identity_payload.pop(identity_key)
        encoded = canonical_bytes(identity_payload)
        identities.append(
            {
                "label": label,
                "canonical_hex": encoded.hex(),
                "sha256": expected,
            }
        )
    return {
        "case_id": case_id,
        "expected_numeric_state": numeric_batch.atomic_state,
        "expected_authorization_state": authorization.state,
        "identities": identities,
    }


def build_fixture() -> dict[str, Any]:
    fixture_bytes = PRODUCER_FIXTURE.read_bytes()
    producer = json.loads(fixture_bytes.decode("utf-8"))
    return {
        "protocol": PROTOCOL,
        "producer_fixture_sha256": hashlib.sha256(
            fixture_bytes
        ).hexdigest(),
        "cases": [
            _case(
                producer,
                case_id="promoted-product",
                second_research_override=False,
            ),
            _case(
                producer,
                case_id="research-override",
                second_research_override=True,
            ),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    payload = build_fixture()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
