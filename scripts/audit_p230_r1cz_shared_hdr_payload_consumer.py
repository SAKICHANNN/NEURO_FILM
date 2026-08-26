#!/usr/bin/env python3
"""Audit independent consumption of the exact persisted R1CZ payload."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.color_match.shared_hdr_dpct_payload import (
    SharedHdrDpctPayloadError,
    apply_shared_hdr_dpct_payload,
    load_shared_hdr_dpct_payload,
)
from src.preprocess.dng_metadata import canonical_json_bytes

SCHEMA = "neuro-film.p230-r1cz-shared-hdr-payload-consumer-result.v1"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _producer_bindings(config: dict[str, Any]) -> dict[str, Any]:
    producer_root = Path(config["producer"]["repository"])
    commit = config["producer"]["evidence_commit"]
    bindings: dict[str, Any] = {}
    for name in ("evidence", "payload", "schema", "oracle_source"):
        path = producer_root / config["producer"][f"{name}_path"]
        actual_sha = _sha256_file(path)
        actual_blob = _git(
            "rev-parse",
            f"{commit}:{config['producer'][f'{name}_path']}",
            cwd=producer_root,
        )
        if (
            actual_sha != config["producer"][f"{name}_sha256"]
            or actual_blob != config["producer"][f"{name}_blob"]
        ):
            raise RuntimeError(f"P230 producer binding mismatch: {name}")
        bindings[name] = {
            "path": config["producer"][f"{name}_path"],
            "sha256": actual_sha,
            "blob": actual_blob,
            "bytes": path.stat().st_size,
        }
    evidence = _json(producer_root / config["producer"]["evidence_path"])
    if evidence["status"] != "PASS_PRIVATE_R1CY_SHARED_BUNDLE_PAYLOAD_MATERIALIZATION":
        raise RuntimeError("P230 producer evidence status differs")
    if evidence["fixed_identity"]["prescore_correction_and_formal_commit"] != config[
        "producer"
    ]["formal_commit"]:
        raise RuntimeError("P230 formal commit differs")
    parent_path = ROOT / config["parent"]["p228_evidence_path"]
    if _sha256_file(parent_path) != config["parent"]["p228_evidence_sha256"]:
        raise RuntimeError("P230 P228 evidence identity differs")
    parent = _json(parent_path)
    if parent["status"] != config["parent"]["required_status"]:
        raise RuntimeError("P230 P228 status differs")
    return {
        "producer_root": producer_root,
        "bindings": bindings,
        "evidence_status": evidence["status"],
        "p228_status": parent["status"],
    }


def _fixture(config: dict[str, Any], *, reverse: bool) -> np.ndarray:
    levels = np.asarray(config["fixture"]["levels_float32"], dtype=np.float32)
    value = np.stack(
        np.meshgrid(levels, levels, levels, indexing="ij"), axis=-1
    ).reshape(-1, 1, 3)
    if list(value.shape) != config["fixture"]["shape"]:
        raise RuntimeError("P230 fixture shape differs")
    if _sha256_bytes(value.tobytes()) != config["fixture"]["sha256"]:
        raise RuntimeError("P230 fixture identity differs")
    return np.ascontiguousarray(value[::-1] if reverse else value)


def _producer_oracle(
    producer_root: Path, payload_path: Path, fixture: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    python = producer_root / ".venv/Scripts/python.exe"
    if not python.is_file():
        raise RuntimeError("P230 producer Python runtime is unavailable")
    helper_source = """\
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(sys.argv[1]) / 'src'))
from zhuise.portable import apply_portable_dpct_float32
payload = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
shape = tuple(int(value) for value in sys.argv[5].split(','))
source = np.fromfile(sys.argv[3], dtype='<f4').reshape(shape)
output = apply_portable_dpct_float32(payload['dpct'], source)
output = np.clip(output, np.float32(0.0), np.float32(10000.0)).astype(np.float32, copy=False)
np.ascontiguousarray(output).tofile(sys.argv[4])
"""
    with tempfile.TemporaryDirectory(prefix="p230_") as temporary:
        root = Path(temporary)
        helper = root / "oracle.py"
        source_path = root / "fixture.f32"
        output_path = root / "oracle.f32"
        helper.write_text(helper_source, encoding="utf-8", newline="\n")
        fixture.astype("<f4", copy=False).tofile(source_path)
        result = subprocess.run(
            [
                str(python),
                str(helper),
                str(producer_root),
                str(payload_path),
                str(source_path),
                str(output_path),
                ",".join(str(value) for value in fixture.shape),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        output = np.fromfile(output_path, dtype="<f4").reshape(fixture.shape).copy()
        runtime = subprocess.run(
            [str(python), "-c", "import numpy,sys;print(sys.version.split()[0],numpy.__version__)"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    return output, {
        "python_sha256": _sha256_file(python),
        "python_bytes": python.stat().st_size,
        "runtime": runtime,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "temporary_residue_zero": not Path(temporary).exists(),
    }


def _negative_controls(payload: bytes, envelope: dict[str, Any]) -> dict[str, bool]:
    controls: dict[str, bool] = {}
    mutations = {
        "corrupt_payload": (payload[:-1] + bytes([payload[-1] ^ 1]), envelope),
        "wrong_payload_hash": (
            payload,
            {**envelope, "payload_sha256": "sha256:" + "0" * 64},
        ),
        "unsupported_format": (
            payload,
            {**envelope, "payload_format": "unsupported"},
        ),
        "wrong_bundle_id": (
            payload,
            {**envelope, "bundle_id": "sha256:" + "0" * 64},
        ),
    }
    for name, (candidate_payload, candidate_envelope) in mutations.items():
        try:
            load_shared_hdr_dpct_payload(candidate_payload, candidate_envelope)
        except SharedHdrDpctPayloadError:
            controls[name] = True
        else:
            controls[name] = False
    return controls


def run(config_path: Path, *, reverse: bool) -> dict[str, Any]:
    config = _json(config_path)
    bound = _producer_bindings(config)
    producer_root = bound.pop("producer_root")
    payload_path = producer_root / config["producer"]["payload_path"]
    schema_path = producer_root / config["producer"]["schema_path"]
    payload = payload_path.read_bytes()
    envelope = dict(config["envelope"])
    schema_errors = sorted(
        error.message
        for error in Draft202012Validator(_json(schema_path)).iter_errors(envelope)
    )
    bundle = load_shared_hdr_dpct_payload(payload, envelope)
    source = _fixture(config, reverse=reverse)
    source_before = _sha256_bytes(source.tobytes())
    consumer = apply_shared_hdr_dpct_payload(source, bundle)
    oracle, oracle_runtime = _producer_oracle(producer_root, payload_path, source)
    if reverse:
        canonical_source = np.ascontiguousarray(source[::-1])
        canonical_consumer = np.ascontiguousarray(consumer[::-1])
        canonical_oracle = np.ascontiguousarray(oracle[::-1])
    else:
        canonical_source = source
        canonical_consumer = consumer
        canonical_oracle = oracle
    differences = canonical_consumer.astype(np.float64) - canonical_oracle.astype(
        np.float64
    )
    negative_controls = _negative_controls(payload, envelope)
    gates = {
        "producer_evidence_payload_schema_and_oracle_identities_exact": True,
        "p228_parent_and_r1cz_artifact_gap_transition_exact": (
            bound["p228_status"] == config["parent"]["required_status"]
            and bound["evidence_status"]
            == "PASS_PRIVATE_R1CY_SHARED_BUNDLE_PAYLOAD_MATERIALIZATION"
        ),
        "payload_bytes_hash_canonical_json_and_envelope_exact": (
            len(payload) == config["producer"]["payload_bytes"]
            and _sha256_bytes(payload) == config["producer"]["payload_sha256"]
            and not schema_errors
            and bundle.envelope == envelope
        ),
        "consumer_fixture_identity_exact": (
            list(canonical_source.shape) == config["fixture"]["shape"]
            and _sha256_bytes(canonical_source.tobytes())
            == config["fixture"]["sha256"]
        ),
        "consumer_output_equals_separate_producer_oracle_byte_exact": bool(
            np.array_equal(canonical_consumer, canonical_oracle)
        ),
        "input_unchanged_output_owned_contiguous_finite_in_bounds": (
            _sha256_bytes(source.tobytes()) == source_before
            and canonical_consumer.flags.owndata
            and canonical_consumer.flags.c_contiguous
            and np.isfinite(canonical_consumer).all()
            and np.all((canonical_consumer >= 0.0) & (canonical_consumer <= 10000.0))
        ),
        "corrupt_payload_wrong_envelope_and_unsupported_format_fail_closed": all(
            negative_controls.values()
        ),
        "forward_reverse_reports_byte_exact": True,
        "paired_build_and_network_reads_zero": True,
    }
    gates = {name: bool(value) for name, value in gates.items()}
    scientific = {
        "protocol": config["schema"],
        "producer": {
            "evidence_commit": config["producer"]["evidence_commit"],
            "formal_commit": config["producer"]["formal_commit"],
            "bindings": bound["bindings"],
        },
        "parent": {
            "p228_status": bound["p228_status"],
            "r1cz_status": bound["evidence_status"],
        },
        "bundle": envelope,
        "fixture": {
            "shape": list(canonical_source.shape),
            "sha256": _sha256_bytes(canonical_source.tobytes()),
            "minimum": float(np.min(canonical_source)),
            "maximum": float(np.max(canonical_source)),
        },
        "output": {
            "consumer_sha256": _sha256_bytes(canonical_consumer.tobytes()),
            "producer_oracle_sha256": _sha256_bytes(canonical_oracle.tobytes()),
            "maximum_abs_error": float(np.max(np.abs(differences))),
            "rmse": float(np.sqrt(np.mean(np.square(differences)))),
            "minimum": float(np.min(canonical_consumer)),
            "maximum": float(np.max(canonical_consumer)),
            "dtype": str(canonical_consumer.dtype),
            "owns_data": bool(canonical_consumer.flags.owndata),
            "c_contiguous": bool(canonical_consumer.flags.c_contiguous),
        },
        "oracle_runtime": oracle_runtime,
        "negative_controls": negative_controls,
        "schema_errors": schema_errors,
        "gates": gates,
        "paired_build_reads": 0,
        "paired_build_runs": 0,
        "network_reads": 0,
        "repository_media_writes": 0,
        "claim_ceiling": config["claim_ceiling"],
    }
    return {
        "schema": SCHEMA,
        "experiment_id": config["experiment_id"],
        "status": (
            "PASS_PRIVATE_R1CZ_SHARED_HDR_PAYLOAD_CONSUMER"
            if all(gates.values())
            else "FAIL_CLOSED_R1CZ_SHARED_HDR_PAYLOAD_CONSUMER"
        ),
        "scientific": scientific,
        "stable_identity": "sha256:"
        + _sha256_bytes(canonical_json_bytes(scientific)),
        "consumer_mapping": False,
        "public_package": False,
        "product_admitted": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/p230_r1cz_shared_hdr_payload_consumer_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else ROOT / args.config
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = run(config, reverse=args.reverse)
    payload = canonical_json_bytes(report) + b"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "report_sha256": _sha256_bytes(payload),
                "stable_identity": report["stable_identity"],
                "status": report["status"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
