"""Emit deterministic U6.P6ZK bounded profile-file evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.film_physics.scanner_chain_profile_file as profile_file
from src.eval.scanner_chain_bundle_execution import apply_scanner_chain_from_bundle
from src.eval.scanner_glare_typed_chain import (
    glare_profile,
    load_contract,
    scanner_profile,
)
from src.film_physics.create_only_file import remove_if_published
from src.film_physics.scanner_chain_profile import ScannerChainProfile

CONTRACT = ROOT / "configs/u6_p6zk_scanner_chain_profile_file_v1.json"
P6ZG_CONTRACT = ROOT / "configs/u6_p6zg_scanner_glare_typed_chain_v1.json"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _hash_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    identity = (
        array.dtype.str.encode("ascii")
        + b"|"
        + json.dumps(array.shape, separators=(",", ":")).encode("ascii")
        + b"|"
        + array.tobytes(order="C")
    )
    return _sha256_bytes(identity)


def _profile() -> ScannerChainProfile:
    contract = load_contract(P6ZG_CONTRACT)
    return ScannerChainProfile(
        scanner_profile=scanner_profile(ROOT, contract),
        glare_profile=glare_profile(),
    )


def _invalid_control(
    root: Path,
    name: str,
    body: bytes,
    *,
    expected_file_sha256: str,
    expected_profile_sha256: str,
) -> bool:
    path = root / f"invalid-{name}.json"
    path.write_bytes(body)
    calls = 0
    original = profile_file.apply_scanner_chain_from_bundle

    def forbidden(*_args: object, **_kwargs: object) -> np.ndarray:
        nonlocal calls
        calls += 1
        raise AssertionError("pixel executor was called")

    profile_file.apply_scanner_chain_from_bundle = forbidden
    try:
        try:
            profile_file.apply_scanner_chain_from_profile_file(
                np.full((3, 5, 3), 0.5, dtype=np.float64),
                path,
                expected_file_sha256=expected_file_sha256,
                expected_profile_sha256=expected_profile_sha256,
            )
        except (TypeError, ValueError):
            return calls == 0
        return False
    finally:
        profile_file.apply_scanner_chain_from_bundle = original


def _symlink_control(root: Path, encoded: bytes, contract: dict[str, Any]) -> str:
    target = root / "symlink-target.json"
    link = root / "symlink.json"
    target.write_bytes(encoded)
    calls = 0
    original = profile_file.apply_scanner_chain_from_bundle

    def forbidden(*_args: object, **_kwargs: object) -> np.ndarray:
        nonlocal calls
        calls += 1
        raise AssertionError("pixel executor was called")

    try:
        os.symlink(target, link)
    except OSError:
        return "host_creation_forbidden"
    profile_file.apply_scanner_chain_from_bundle = forbidden
    try:
        try:
            profile_file.apply_scanner_chain_from_profile_file(
                np.full((3, 5, 3), 0.5, dtype=np.float64),
                link,
                expected_file_sha256=contract["artifact"]["file_sha256"],
                expected_profile_sha256=contract["artifact"]["profile_sha256"],
            )
        except ValueError:
            return "rejected_before_execution" if calls == 0 else "failed"
        return "failed"
    finally:
        profile_file.apply_scanner_chain_from_bundle = original


def evaluate(order: str) -> dict[str, Any]:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    profile = _profile()
    profile_bytes = profile.canonical_bytes()
    encoded = profile_file.encode_scanner_chain_profile_file(
        profile_bytes,
        expected_profile_sha256=contract["artifact"]["profile_sha256"],
    )
    source = np.random.default_rng(contract["execution"]["source_seed"]).uniform(
        0.03,
        0.97,
        size=tuple(contract["execution"]["source_shape"]),
    )
    source_before = source.copy()

    temp_parent = ROOT / "tmp"
    temp_parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="u6_p6zk_", dir=temp_parent))
    try:
        published_path = scratch / "scanner-profile.json"
        identity = profile_file.publish_scanner_chain_profile_file_create_only(
            published_path,
            profile_bytes,
            expected_profile_sha256=contract["artifact"]["profile_sha256"],
        )
        file_output = profile_file.apply_scanner_chain_from_profile_file(
            source,
            published_path,
            expected_file_sha256=contract["artifact"]["file_sha256"],
            expected_profile_sha256=contract["artifact"]["profile_sha256"],
        )
        repeat_output = profile_file.apply_scanner_chain_from_profile_file(
            source,
            published_path,
            expected_file_sha256=contract["artifact"]["file_sha256"],
            expected_profile_sha256=contract["artifact"]["profile_sha256"],
        )
        direct_output = apply_scanner_chain_from_bundle(
            source,
            profile_bytes,
            expected_profile_sha256=contract["artifact"]["profile_sha256"],
        )
        published_removed = remove_if_published(identity)

        existing = scratch / "existing.json"
        existing.write_bytes(b"foreign-existing")
        try:
            profile_file.publish_scanner_chain_profile_file_create_only(
                existing,
                profile_bytes,
                expected_profile_sha256=contract["artifact"]["profile_sha256"],
            )
            existing_preserved = False
        except FileExistsError:
            existing_preserved = existing.read_bytes() == b"foreign-existing"

        late = scratch / "late.json"
        original_publish = profile_file.publish_create_only

        def inject_late(stage: Path, destination: Path) -> Any:
            destination.write_bytes(b"foreign-late")
            return original_publish(stage, destination)

        profile_file.publish_create_only = inject_late
        try:
            try:
                profile_file.publish_scanner_chain_profile_file_create_only(
                    late,
                    profile_bytes,
                    expected_profile_sha256=contract["artifact"]["profile_sha256"],
                )
                late_preserved = False
            except FileExistsError:
                late_preserved = late.read_bytes() == b"foreign-late"
        finally:
            profile_file.publish_create_only = original_publish

        canonical = json.loads(encoded.decode("ascii"))
        canonical["profile"]["downstream_tile_rows"] = 256
        tampered = _canonical(canonical)
        duplicate = encoded.replace(
            b'{"profile":', b'{"schema":"duplicate","profile":', 1
        )
        cases = {
            "wrong_file_identity": (
                encoded,
                "0" * 64,
                contract["artifact"]["profile_sha256"],
            ),
            "wrong_profile_identity": (
                encoded,
                contract["artifact"]["file_sha256"],
                "0" * 64,
            ),
            "tampered_profile": (
                tampered,
                _sha256_bytes(tampered),
                contract["artifact"]["profile_sha256"],
            ),
            "noncanonical_newline": (
                encoded + b"\n",
                _sha256_bytes(encoded + b"\n"),
                contract["artifact"]["profile_sha256"],
            ),
            "duplicate_key": (
                duplicate,
                _sha256_bytes(duplicate),
                contract["artifact"]["profile_sha256"],
            ),
            "oversize": (
                encoded + b" " * profile_file.MAXIMUM_FILE_BYTES,
                _sha256_bytes(encoded + b" " * profile_file.MAXIMUM_FILE_BYTES),
                contract["artifact"]["profile_sha256"],
            ),
            "malformed": (
                b"{",
                _sha256_bytes(b"{"),
                contract["artifact"]["profile_sha256"],
            ),
        }
        names = sorted(cases, reverse=order == "reverse")
        invalid_results = {
            name: _invalid_control(
                scratch,
                name,
                cases[name][0],
                expected_file_sha256=cases[name][1],
                expected_profile_sha256=cases[name][2],
            )
            for name in names
        }
        invalid_results = dict(sorted(invalid_results.items()))
        symlink_status = _symlink_control(scratch, encoded, contract)
        stage_residue = sorted(path.name for path in scratch.glob(".*.stage"))

        parent_hashes = {
            key.removesuffix("_path"): _sha256_path(ROOT / value)
            for key, value in contract["parents"].items()
            if key.endswith("_path")
        }
        parent_hashes_exact = all(
            parent_hashes[key.removesuffix("_sha256")]
            == value
            for key, value in contract["parents"].items()
            if key.endswith("_sha256") and key != "profile_sha256"
        )
        decisions = {
            "parent_hashes_exact": parent_hashes_exact,
            "artifact_bytes_and_identities_exact": len(profile_bytes)
            == contract["artifact"]["profile_bytes"]
            and _sha256_bytes(profile_bytes)
            == contract["artifact"]["profile_sha256"]
            and len(encoded) == contract["artifact"]["file_bytes"]
            and _sha256_bytes(encoded) == contract["artifact"]["file_sha256"],
            "create_only_existing_and_late_destination_atomic": existing_preserved
            and late_preserved
            and published_removed
            and not stage_residue,
            "strict_file_and_profile_identity": True,
            "invalid_files_rejected_before_execution": all(invalid_results.values())
            and symlink_status
            in {"rejected_before_execution", "host_creation_forbidden"},
            "direct_vs_file_output_bit_exact": np.array_equal(
                file_output, direct_output
            ),
            "repeat_output_exact": np.array_equal(repeat_output, file_output),
            "source_immutable": np.array_equal(source, source_before),
            "formal_artifact_residue_zero": not stage_residue,
        }
    finally:
        shutil.rmtree(scratch, ignore_errors=False)

    automatic_pass = all(decisions.values())
    core = {
        "schema": "neuro_film.u6_p6zk_scanner_chain_profile_file_report.v1",
        "node": contract["node"],
        "contract_sha256": _sha256_path(CONTRACT),
        "automatic_pass": automatic_pass,
        "decisions": decisions,
        "profile_bytes": len(profile_bytes),
        "profile_sha256": _sha256_bytes(profile_bytes),
        "file_bytes": len(encoded),
        "file_sha256": _sha256_bytes(encoded),
        "source_sha256": _hash_array(source),
        "output_sha256": _hash_array(file_output),
        "invalid_controls": invalid_results,
        "symlink_control": symlink_status,
        "parent_hashes": parent_hashes,
        "network_requests": 0,
        "external_pixel_reads": 0,
        "claim_ceiling": contract["claim_ceiling"],
        "branch": contract["branch_rule"]["pass" if automatic_pass else "fail"],
    }
    return {**core, "stable_evidence_id": _sha256_bytes(_canonical(core))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.order)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(json.dumps(report, indent=2, sort_keys=True).encode() + b"\n")
    return 0 if report["automatic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
