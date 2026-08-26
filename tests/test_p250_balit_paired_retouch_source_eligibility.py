from __future__ import annotations

from scripts.audit_p250_balit_paired_retouch_source_eligibility import (
    _extract_facts,
    _plain_text,
)

EXPECTED = {
    "dataset_name": "BALit: Dataset for Backlit Low-Light Image Enhancement",
    "doi": "10.21227/cdqg-q549",
}


def test_extracts_positive_pair_facts_without_inventing_rights_or_groups() -> None:
    text = """
    BALit: Dataset for Backlit Low-Light Image Enhancement
    DOI 10.21227/cdqg-q549
    1,000 original BALit CR2 inputs (21.5 GB)
    1,000 expert-retouched references (26.84 GB)
    LOGIN TO ACCESS DATASET FILES
    """
    facts = _extract_facts(text, EXPECTED)
    assert facts["name_present"]
    assert facts["doi_present"]
    assert facts["raw_count_present"]
    assert facts["reference_count_present"]
    assert facts["cr2_present"]
    assert facts["raw_size_present"]
    assert facts["reference_size_present"]
    assert facts["login_required"]
    assert not facts["commercial_compatible_license"]
    assert not facts["exact_inventory"]
    assert not facts["group_identity_and_roles"]


def test_explicit_cc_by_manifest_and_group_roles_are_detected() -> None:
    text = """
    Creative Commons Attribution 4.0. SHA-256 file manifest with filename.
    scene_id and train validation test roles.
    """
    facts = _extract_facts(text, EXPECTED)
    assert facts["commercial_compatible_license"]
    assert facts["exact_inventory"]
    assert facts["group_identity_and_roles"]


def test_html_to_text_is_deterministic() -> None:
    assert _plain_text(b"<p>BALit &amp; RAW</p><p>paired</p>") == "BALit & RAW paired"
