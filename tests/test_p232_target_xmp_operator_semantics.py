from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_p232_target_xmp_operator_semantics.py"
CONFIG = ROOT / "configs/p232_target_xmp_operator_semantics_v1.json"
XMP = ROOT / "outputs/source_recon/mmart_ppr10k_public_metadata/config.xmp"
SPEC = importlib.util.spec_from_file_location("p232_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_exact_public_xmp_facts_are_stable() -> None:
    facts = module.inspect_xmp(XMP.read_bytes())
    assert facts["camera_raw_attribute_count"] == 123
    assert facts["direct_curve_sequence_count"] == 12
    assert facts["process_version"] == "11.0"
    assert facts["camera_profile"] == "Adobe Standard"
    assert facts["observed_colour_fields"]["Exposure2012"] == "-0.05"
    assert facts["renderer_equations_present"] is False
    assert facts["processing_order_present"] is False
    assert facts["profile_bytes_present"] is False


def test_semantic_admission_fails_before_operator_or_pixels() -> None:
    result = module.run(CONFIG)
    assert result["status"].startswith("FAIL_CLOSED_BEFORE_OPERATOR_COMPILATION")
    assert result["gates"]["minimum_independent_recipes"] is False
    assert result["gates"]["bulk_recipe_access_authorized"] is False
    assert result["gates"]["renderer_equations_present"] is False
    assert result["gates"]["referenced_profile_bytes_present"] is False
    assert set(result["execution"].values()) == {0}
    assert result["consumer_mapping"] is False


def test_forward_reverse_enumeration_is_scientifically_exact() -> None:
    assert module.run(CONFIG) == module.run(CONFIG, reverse=True)


def test_nonfinite_numeric_colour_field_rejects() -> None:
    mutated = XMP.read_text(encoding="utf-8").replace(
        'crs:Exposure2012="-0.05"', 'crs:Exposure2012="NaN"'
    )
    with pytest.raises(ValueError, match="non-finite"):
        module.inspect_xmp(mutated.encode("utf-8"))
