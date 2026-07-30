"""Unified read-only discovery contract for the reference-match product shell."""

from __future__ import annotations

from typing import Any

from .batch_limits import MAX_REFERENCE_MATCH_BATCH_SOURCES
from .contracts import REFERENCE_LOOK_RECIPE_SCHEMA_ID
from .files import (
    REFERENCE_FILE_INPUT_INSPECTION_BATCH_SCHEMA_ID,
    REFERENCE_FILE_INPUT_INSPECTION_CLAIM_CEILING,
    REFERENCE_FILE_INPUT_INSPECTION_SCHEMA_ID,
    reference_file_output_capabilities_payload,
    reference_file_supported_input_rails,
)
from .output_metadata_policy import reference_file_output_metadata_policy_payload
from .safety import REFERENCE_RENDER_GUARD_POLICY_ID

REFERENCE_MATCH_PRODUCT_CAPABILITIES_ID = (
    "neuro-film.reference-match-product-capabilities.v1"
)
REFERENCE_MATCH_PRODUCT_CAPABILITIES_CLAIM_CEILING = (
    "discovery-only-relative-sdr-local-product-shell-"
    "not-render-authorization-or-algorithm-promotion"
)


def reference_match_product_capabilities_payload() -> dict[str, Any]:
    """Return one versioned UI/IPC discovery payload without probing files."""

    return {
        "schema_id": REFERENCE_MATCH_PRODUCT_CAPABILITIES_ID,
        "claim_ceiling": REFERENCE_MATCH_PRODUCT_CAPABILITIES_CLAIM_CEILING,
        "operations": [
            "fit-and-render",
            "input-inspection",
            "recipe-replay",
        ],
        "batch": {
            "minimum_sources": 1,
            "maximum_sources": MAX_REFERENCE_MATCH_BATCH_SOURCES,
            "order_semantics": "caller-order-preserved",
            "transaction_semantics": "all-artifacts-or-no-new-artifacts",
        },
        "input": {
            "inspection_schema_id": (
                REFERENCE_FILE_INPUT_INSPECTION_SCHEMA_ID
            ),
            "inspection_batch_schema_id": (
                REFERENCE_FILE_INPUT_INSPECTION_BATCH_SCHEMA_ID
            ),
            "inspection_claim_ceiling": (
                REFERENCE_FILE_INPUT_INSPECTION_CLAIM_CEILING
            ),
            "accepted_decoded_rails": list(
                reference_file_supported_input_rails()
            ),
            "render_revalidates_inputs": True,
        },
        "recipe": {
            "schema_id": REFERENCE_LOOK_RECIPE_SCHEMA_ID,
            "reference_required_for_fit": True,
            "reference_required_for_replay": False,
        },
        "output": reference_file_output_capabilities_payload(),
        "output_metadata": (
            reference_file_output_metadata_policy_payload()
        ),
        "delivery": {
            "guard_policy_id": REFERENCE_RENDER_GUARD_POLICY_ID,
            "algorithm_status": "not-promoted",
            "default_action": "identity-fallback",
            "research_override": "explicit-opt-in-only-not-product",
        },
    }


__all__ = [
    "REFERENCE_MATCH_PRODUCT_CAPABILITIES_CLAIM_CEILING",
    "REFERENCE_MATCH_PRODUCT_CAPABILITIES_ID",
    "reference_match_product_capabilities_payload",
]
