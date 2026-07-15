from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.roll2film.ct5_data import (
    CT5DataContract,
    load_ct5_training_rows,
    sample_ct5_internal_dev,
    sample_ct5_training,
)
from src.roll2film.manifests import FILMSET_MANIFEST_SCHEMA, FilmSetManifestError


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> str:
    path.write_bytes("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode())
    return _sha(path)


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    dataset = tmp_path / "FilmSet"
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    rows_by_role = {role: [] for role in ("source_train", "target_train", "internal_dev_lockbox")}
    access = {
        "source_train": "training_source_only",
        "target_train": "training_target_only",
        "internal_dev_lockbox": "internal_evaluator_only",
    }
    specs = (
        ("source_train", "source", ("input",)),
        ("target_train", "target", ("cinema", "classneg", "velvia")),
        ("internal_dev_lockbox", "dev_a", ("input", "cinema", "classneg", "velvia")),
        ("internal_dev_lockbox", "dev_b", ("input", "cinema", "classneg", "velvia")),
    )
    for role, content, domains in specs:
        source = np.random.default_rng(len(content)).integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
        for domain_index, domain in enumerate(domains):
            relative = Path("train") / domain / f"{content}.png"
            path = dataset / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.roll(source, domain_index, axis=2)).save(path)
            rows_by_role[role].append(
                {
                    "schema_version": FILMSET_MANIFEST_SCHEMA,
                    "research_pool": role,
                    "payload_access": access[role],
                    "content_id": f"train:{content}.png",
                    "duplicate_cluster_id": f"cluster:{content}",
                    "domain": domain,
                    "path": relative.as_posix(),
                    "sha256": _sha(path),
                }
            )
    hashes = {
        role: _write_manifest(evidence / f"{role}.jsonl", rows)
        for role, rows in rows_by_role.items()
    }
    final_path = evidence / "final_628_lockbox.jsonl"
    final_path.write_text("sealed and never parsed\n", encoding="utf-8")
    hashes["final_628_lockbox"] = _sha(final_path)
    config = {
        "experiment_id": "test",
        "seed": 7,
        "dataset": {
            "root": "FilmSet",
            "evidence_dir": "evidence",
            "source_manifest_sha256": hashes["source_train"],
            "target_manifest_sha256": hashes["target_train"],
            "internal_dev_manifest_sha256": hashes["internal_dev_lockbox"],
            "forbidden_final_manifest_sha256": hashes["final_628_lockbox"],
            "domains": ["cinema", "classneg", "velvia"],
        },
        "sampling": {
            "pixels_per_training_image": 16,
            "pixels_per_dev_image": 16,
            "pilot_fraction": 0.5,
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path, tmp_path


def test_ct5_training_and_dev_sampling_are_deterministic_and_role_isolated(tmp_path: Path) -> None:
    config_path, root = _fixture(tmp_path)
    contract = CT5DataContract.from_config(config_path, root)
    first = sample_ct5_training(contract)
    second = sample_ct5_training(contract)
    dev = sample_ct5_internal_dev(contract)

    assert np.array_equal(first.source_pixels, second.source_pixels)
    assert first.source_pixels.shape == (1, 16, 3)
    assert all(values.shape == (1, 16, 3) for values in first.target_pixels.values())
    assert dev.input_pixels.shape == (2, 16, 3)
    assert set(dev.folds) <= {"pilot", "confirmatory"}
    assert all(values.shape == dev.input_pixels.shape for values in dev.target_pixels.values())


def test_ct5_manifest_hash_and_role_mismatch_fail_closed(tmp_path: Path) -> None:
    config_path, root = _fixture(tmp_path)
    contract = CT5DataContract.from_config(config_path, root)
    source_path = contract.evidence_dir / "source_train.jsonl"
    source_path.write_text(source_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(FilmSetManifestError, match="manifest hash mismatch"):
        load_ct5_training_rows(contract)


def test_ct5_payload_hash_mismatch_fails_before_decode(tmp_path: Path) -> None:
    config_path, root = _fixture(tmp_path)
    contract = CT5DataContract.from_config(config_path, root)
    source_rows, _ = load_ct5_training_rows(contract)
    path = contract.dataset_root / str(source_rows[0]["path"])
    path.write_bytes(b"corrupt")
    with pytest.raises(FilmSetManifestError, match="payload hash mismatch"):
        sample_ct5_training(contract)
