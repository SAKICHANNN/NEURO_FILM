from __future__ import annotations

import json
import struct
from pathlib import Path

from scripts.run_u5_r2spcp0_pairwise_preference_source_lock import (
    _parse_central_directory,
    _report_schema,
    _stable_id,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2spcp0_pairwise_preference_source_lock_v1.json"
DECISION = ROOT / "configs/u5_r2spcp0_pairwise_preference_source_lock_decision_v1.json"
EVIDENCE = ROOT / "docs/evidence/U5_R2SPCP0_PAIRWISE_PREFERENCE_SOURCE_LOCK_RESULT.json"
CORRECTED_CONTRACT = (
    ROOT / "configs/u5_r2spcp1_corrected_pairwise_preference_source_lock_v1.json"
)
CORRECTED_DECISION = (
    ROOT / "configs/u5_r2spcp1_corrected_pairwise_preference_source_lock_decision_v1.json"
)
CORRECTED_EVIDENCE = (
    ROOT / "docs/evidence/U5_R2SPCP1_CORRECTED_PAIRWISE_PREFERENCE_SOURCE_LOCK_RESULT.json"
)


def test_spcp0_contract_freezes_metadata_only_source() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert payload["experiment_id"] == "U5.R2SPCP0"
    assert payload["source"]["revision"] == (
        "068af97eed82969f15278db3af4bd450176cf6f3"
    )
    assert payload["range_protocol"]["full_zip_download_forbidden"] is True
    assert payload["range_protocol"]["image_member_payload_read_forbidden"] is True
    assert payload["frozen_gates"]["image_payload_bytes_read_exact"] == 0


def test_spcp0_contract_freezes_complete_preference_graph() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    gates = payload["frozen_gates"]
    assert gates["scene_count_exact"] == 1000
    assert gates["canonical_image_count_exact"] == 12000
    assert gates["pair_rows_exact"] == 45000
    assert gates["pairs_per_scene_exact"] == 45
    assert gates["score_rows_exact"] == 12000
    assert gates["subject_columns_exact"] == 20
    assert payload["rights"]["image_source_and_redistribution_rights_inferred"] is False


def test_spcp0_successor_is_one_explicit_source_free_operator() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    successor = payload["prospective_successor"]
    assert "one shared source-free bounded" in successor["mechanism"]
    assert "per-image or per-scene routing" in successor["forbidden"]
    assert "direct RGB generation" in successor["forbidden"]


def test_spcp0_central_directory_parser_rejects_noncentral_bytes() -> None:
    try:
        _parse_central_directory(b"PK\x03\x04")
    except ValueError as error:
        assert "signature drift" in str(error)
    else:
        raise AssertionError("non-central ZIP bytes were accepted")


def test_spcp0_central_directory_parser_reads_one_row() -> None:
    name = b"SPCP_dataset/example.txt"
    header = struct.pack(
        "<IHHHHHHIIIHHHHHII",
        0x02014B50,
        20,
        20,
        0,
        8,
        0,
        0,
        123,
        9,
        11,
        len(name),
        0,
        0,
        0,
        0,
        0,
        75,
    )
    rows = _parse_central_directory(header + name)
    assert rows == [
        {
            "name": name.decode(),
            "method": 8,
            "crc32": 123,
            "compressed_size": 9,
            "uncompressed_size": 11,
            "local_offset": 75,
        }
    ]


def test_spcp0_stable_id_ignores_existing_identity() -> None:
    left = {"a": 1}
    right = {"a": 1, "stable_evidence_id": "old"}
    assert _stable_id(left) == _stable_id(right)


def test_spcp_report_schema_tracks_numbered_protocol() -> None:
    assert _report_schema("U5.R2SPCP1") == (
        "neuro-film.u5-r2spcp1-pairwise-preference-source-lock-report.v1"
    )


def test_spcp0_decision_preserves_hash_failure_and_zero_pixels() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    assert decision["status"] == "failed_closed_before_image_acquisition"
    assert decision["failed_gates"] == [
        "pair_annotation_sha256_exact",
        "score_annotation_sha256_exact",
    ]
    assert decision["execution_integrity"]["image_payload_bytes_read"] == 0
    assert "silently replacing" in decision["forbidden"][0]


def test_spcp0_evidence_binds_exact_replay_and_observed_hashes() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["formal_report_sha256"] == evidence["second_report_sha256"]
    assert evidence["execution"]["two_report_replay_byte_exact"] is True
    assert evidence["execution"]["image_payload_bytes_read"] == 0
    assert evidence["structure"]["connected_pair_graphs"] == 1000
    assert evidence["observed_annotation_hashes"]["order_trans.xlsx"] == (
        "8c42140ae0f90f37f32706911ab86cca9f377077bbd18ac301262d952bf5f58c"
    )


def test_spcp1_is_prospective_and_binds_spcp0_failure() -> None:
    corrected = json.loads(CORRECTED_CONTRACT.read_text(encoding="utf-8"))
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert corrected["experiment_id"] == "U5.R2SPCP1"
    assert corrected["parent"]["required_decision"] == (
        "close_exact_spcp0_protocol_on_preregistered_annotation_hash_mismatch"
    )
    assert corrected["annotations"]["pairwise"]["sha256"] == (
        evidence["observed_annotation_hashes"]["order_trans.xlsx"]
    )
    assert corrected["annotations"]["scores"]["sha256"] == (
        evidence["observed_annotation_hashes"]["score_trans2.xlsx"]
    )
    assert corrected["range_protocol"]["image_member_payload_read_forbidden"] is True


def test_spcp1_pass_opens_only_separate_explicit_operator_d0() -> None:
    decision = json.loads(CORRECTED_DECISION.read_text(encoding="utf-8"))
    evidence = json.loads(CORRECTED_EVIDENCE.read_text(encoding="utf-8"))
    assert decision["status"] == "metadata_only_source_lock_pass"
    assert decision["execution_integrity"]["image_payload_bytes_read"] == 0
    assert "global_explicit_operator_d0" in decision["decision"]
    assert evidence["formal_report_sha256"] == evidence["second_report_sha256"]
    assert evidence["gates"]["failed"] == 0
    assert evidence["execution"]["full_zip_downloaded"] is False
