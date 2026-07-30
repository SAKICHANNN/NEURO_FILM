from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from scripts.run_android_srgb_quantizer_testlab_gcp_v1 import (
    CloudRuntimeError,
    _find_matrix_id,
    _matrix_failure_codes,
    _package,
    _runtime_tokens,
    _submission_matrix_id,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ({"matrixId": "matrix-abc123"}, "matrix-abc123"),
        (
            {"testMatrix": {"testMatrixId": "matrix-nested"}},
            "matrix-nested",
        ),
        ([{"other": 1}, {"matrixId": "matrix-list"}], "matrix-list"),
        ({"matrixId": "../escape"}, None),
        ({"other": "matrix-not-an-identity-field"}, None),
    ],
)
def test_find_matrix_id_is_strict(value, expected) -> None:
    assert _find_matrix_id(value) == expected


def test_submission_matrix_id_survives_validation_nonzero() -> None:
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=1,
        stdout="",
        stderr=(
            "Matrix [matrix-validation123] failed during validation."
        ),
    )
    assert _submission_matrix_id(completed) == "matrix-validation123"


def test_matrix_failure_codes_are_enum_only() -> None:
    assert _matrix_failure_codes(
        {
            "invalidMatrixDetails": "SERVICE_NOT_ACTIVATED",
            "extendedInvalidMatrixDetails": [
                {
                    "reason": "SERVICE_NOT_ACTIVATED",
                    "message": "contains project and console URL",
                },
                {"reason": "../../unsafe"},
            ],
        }
    ) == ["SERVICE_NOT_ACTIVATED"]


def test_local_package_report_rehashes_exact_apks() -> None:
    package = _package()
    assert package["status"] == "PASS"
    assert package["package_prefix"] == "nf-019f9f37-p90"


def test_runtime_tokens_require_complete_device_result(
    tmp_path: Path,
) -> None:
    complete = tmp_path / "instrumentation.results"
    complete.write_text(
        """
        {"schema":"neuro-film.android-srgb-quantizer-runtime.v1",
        "status":"PASS","sample_count":4096,
        "threshold_identity":"fae645ef1aad04fcd1233631a32f820cf7696e3ca65d31939acf60d7f123674c",
        "vector_sha256":"3d4205e51de80392ea7a4e5eccaf05ab6d322a48475603764c28e47d61aa7628",
        "q8_exact":true,"q16_exact":true,
        "inner_replay_exact":true,"failure_atomic":true,
        "icc_exact":true,"eotf_q8_roundtrip_exact":true,
        "eotf_q16_roundtrip_exact":true,"eotf_failure_atomic":true,
        "product_chain_vector_count":10,
        "product_chain_canonical_bytes":14254,
        "product_chain_hashes_exact":true,
        "product_chain_sha_failure_atomic":true,
        "staging_truth_table_exact":true}
        """,
        encoding="utf-8",
    )
    evidence = _runtime_tokens([complete])
    assert evidence["required_token_count"] == 18
    assert evidence["matched_file_names"] == ["instrumentation.results"]
    incomplete = tmp_path / "incomplete.txt"
    incomplete.write_text('"status":"PASS"', encoding="utf-8")
    with pytest.raises(CloudRuntimeError, match="lacks required"):
        _runtime_tokens([incomplete])


def test_runtime_tokens_accept_test_lab_escaped_json(
    tmp_path: Path,
) -> None:
    escaped = tmp_path / "instrumentation.results"
    escaped.write_text(
        r"""
        test_result={
        \"schema\":\"neuro-film.android-srgb-quantizer-runtime.v1\",
        \"status\":\"PASS\",\"sample_count\":4096,
        \"threshold_identity\":\"fae645ef1aad04fcd1233631a32f820cf7696e3ca65d31939acf60d7f123674c\",
        \"vector_sha256\":\"3d4205e51de80392ea7a4e5eccaf05ab6d322a48475603764c28e47d61aa7628\",
        \"q8_exact\":true,\"q16_exact\":true,
        \"inner_replay_exact\":true,\"failure_atomic\":true,
        \"icc_exact\":true,\"eotf_q8_roundtrip_exact\":true,
        \"eotf_q16_roundtrip_exact\":true,
        \"eotf_failure_atomic\":true,
        \"product_chain_vector_count\":10,
        \"product_chain_canonical_bytes\":14254,
        \"product_chain_hashes_exact\":true,
        \"product_chain_sha_failure_atomic\":true,
        \"staging_truth_table_exact\":true}
        """,
        encoding="utf-8",
    )
    evidence = _runtime_tokens([escaped])
    assert evidence["required_token_count"] == 18
    assert evidence["matched_file_names"] == ["instrumentation.results"]
