from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.roll2film.filmset_reference_preflight import (
    build_filmset_reference_preflight,
)
from src.roll2film.manifests import (
    FILMSET_ARCHIVE_VERSION,
    FILMSET_DATASET_ID,
    FILMSET_MANIFEST_SCHEMA,
    FilmSetManifestError,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(
    role: str, access: str, content_id: str, cluster: str, domain: str
) -> dict[str, object]:
    return {
        "schema_version": FILMSET_MANIFEST_SCHEMA,
        "dataset_id": FILMSET_DATASET_ID,
        "archive_version": FILMSET_ARCHIVE_VERSION,
        "research_pool": role,
        "payload_access": access,
        "allowed_use": "internal_research_only",
        "redistributable": False,
        "content_id": content_id,
        "duplicate_cluster_id": cluster,
        "domain": domain,
        "path": f"train/{domain}/{content_id}.png",
        "sha256": "0" * 64,
    }


def _write(path: Path, rows: list[dict[str, object]]) -> str:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    return _sha(path)


def _fixture(tmp_path: Path) -> dict[str, object]:
    (tmp_path / "FilmSet").mkdir()
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    source = [
        _row("source_train", "training_source_only", f"s{i}", f"cs{i}", "input")
        for i in range(4)
    ]
    target = [
        _row(
            "target_train",
            "training_target_only",
            f"t{i}",
            f"ct{i}",
            domain,
        )
        for i in range(4)
        for domain in ("cinema", "classneg", "velvia")
    ]
    internal = [
        _row(
            "internal_dev_lockbox",
            "internal_evaluator_only",
            f"d{i}",
            f"cd{i}",
            domain,
        )
        for i in range(5)
        for domain in ("input", "cinema", "classneg", "velvia")
    ]
    hashes = {
        "source": _write(evidence / "source.jsonl", source),
        "target": _write(evidence / "target.jsonl", target),
        "internal": _write(evidence / "internal.jsonl", internal),
    }
    final = evidence / "final.jsonl"
    final.write_text("must not be parsed\n", encoding="utf-8", newline="\n")
    hashes["final"] = _sha(final)
    return {
        "schema_version": 1,
        "experiment_id": "test",
        "node": "U5.R2W2F",
        "status": "queued",
        "dataset": {
            "dataset_id": FILMSET_DATASET_ID,
            "archive_version": FILMSET_ARCHIVE_VERSION,
            "root": "FilmSet",
            "evidence_dir": "evidence",
            "source_manifest": "source.jsonl",
            "source_manifest_sha256": hashes["source"],
            "target_manifest": "target.jsonl",
            "target_manifest_sha256": hashes["target"],
            "internal_dev_manifest": "internal.jsonl",
            "internal_dev_manifest_sha256": hashes["internal"],
            "forbidden_final_manifest": "final.jsonl",
            "forbidden_final_manifest_sha256": hashes["final"],
            "domains": ["cinema", "classneg", "velvia"],
            "expected_source_identities": 4,
            "expected_target_identities": 4,
            "expected_internal_identities": 5,
            "expected_source_rows": 4,
            "expected_target_rows": 12,
            "expected_internal_rows": 20,
        },
        "partition": {
            "algorithm": "sha256_seed_pool_content_id_ascending",
            "seed": 7,
            "target_reference_bank_identities": 2,
            "source_probe_identities": 2,
            "internal_development_identities": 2,
            "internal_confirmatory_identities": 2,
            "internal_stress_identities": 1,
        },
        "activation_gate": {},
        "allowed_use": "internal_research_only",
        "claim_ceiling": "test",
    }


def test_preflight_is_repeat_exact_and_does_not_parse_final(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    first = build_filmset_reference_preflight(
        config,
        root=tmp_path,
        config_sha256="config",
        software_commit="commit",
    )
    second = build_filmset_reference_preflight(
        config,
        root=tmp_path,
        config_sha256="config",
        software_commit="commit",
    )
    assert first == second
    assert first["config_sha256"] == "config"
    assert first["software_commit"] == "commit"
    assert first["observed"]["target_rows"] == 12
    assert first["partition"]["counts"]["internal_stress"] == 1
    assert first["final_manifest_payload_rows_parsed"] == 0
    assert first["image_payloads_read"] == 0


def test_preflight_rejects_manifest_drift_before_partition(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    source = tmp_path / "evidence" / "source.jsonl"
    source.write_text(
        source.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(FilmSetManifestError, match="manifest hash mismatch"):
        build_filmset_reference_preflight(config, root=tmp_path)


def test_preflight_rejects_cross_pool_cluster_leakage(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    target = tmp_path / "evidence" / "target.jsonl"
    rows = [
        json.loads(line)
        for line in target.read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        if row["content_id"] == "t0":
            row["duplicate_cluster_id"] = "cs0"
    config["dataset"]["target_manifest_sha256"] = _write(target, rows)
    with pytest.raises(FilmSetManifestError, match="pool leakage"):
        build_filmset_reference_preflight(config, root=tmp_path)


def test_preflight_rejects_unimplemented_partition_label(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    config["partition"]["algorithm"] = "claimed_but_not_implemented"
    with pytest.raises(FilmSetManifestError, match="partition algorithm"):
        build_filmset_reference_preflight(config, root=tmp_path)


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("dataset_id", "dataset ID mismatch"),
        ("archive_version", "archive version mismatch"),
    ),
)
def test_preflight_rejects_config_lineage_drift(
    tmp_path: Path, field: str, message: str
) -> None:
    config = _fixture(tmp_path)
    config["dataset"][field] = "drifted"
    with pytest.raises(FilmSetManifestError, match=message):
        build_filmset_reference_preflight(config, root=tmp_path)
