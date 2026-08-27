from __future__ import annotations

from scripts.audit_p301_mono_hdr_3d_real_exposure_source import _classify


def test_material_observation_does_not_imply_external_dataset_rights() -> None:
    readme = """
    Download and organize the dataset. It contains 4 real scenes captured at
    35 different poses with 5 different exposure time values. Dataset here.
    real/flower real/computer ...
    """
    license_text = "Apache License Version 2.0"
    facts = _classify(readme, license_text, ["README.md", "LICENSE"])
    assert facts["material_capture_observation_explicit"]
    assert facts["apache_license_present"]
    assert not facts["external_dataset_rights_explicit"]
    assert not facts["exact_asset_manifest_with_sizes_and_checksums"]
    assert not facts["group_identities_ready"]


def test_explicit_dataset_rights_manifest_and_groups_can_pass() -> None:
    readme = """
    Download and organize the dataset. It contains 4 real scenes captured at
    35 different poses with 5 different exposure time values. The dataset is
    licensed under Apache-2.0. SHA256 checksum inventory. Real scenes are
    real/flower real/computer real/desk real/kitchen.
    """
    facts = _classify(readme, "Apache License", ["dataset_manifest_sha256.json"])
    assert facts["material_capture_observation_explicit"]
    assert facts["external_dataset_rights_explicit"]
    assert facts["exact_asset_manifest_with_sizes_and_checksums"]
    assert facts["group_identities_ready"]
