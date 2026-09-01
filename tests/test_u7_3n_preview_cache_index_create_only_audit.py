from __future__ import annotations

from scripts.audit_u7_3n_preview_cache_index_create_only import run_audit


def test_u7_3n_formal_audit_passes_both_orders() -> None:
    forward = run_audit("forward")
    reverse = run_audit("reverse")

    assert forward == reverse
    assert forward["status"] == (
        "PASS_PRIVATE_U7_3N_PREVIEW_CACHE_INDEX_CREATE_ONLY_REPAIR"
    )
    assert all(forward["gates"].values())
    assert {row["case_id"] for row in forward["rows"]} == {
        "success",
        "existing-destination",
        "late-foreign-destination",
        "clean-publication-failure",
        "foreign-stage-replacement",
    }
