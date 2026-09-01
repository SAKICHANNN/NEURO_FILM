#!/usr/bin/env python3
"""Run the frozen U4.5E receipt-bound preview-session audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import three_stock_preview as preview_module
from src.inference.render_contract import sha256_file
from src.inference.three_stock_preview_cache import (
    CACHE_INDEX_NAME,
    ThreeStockPreviewCacheError,
    publish_three_stock_preview_cache_index,
)
from src.inference.three_stock_preview_session import (
    ReceiptBoundVerifiedThreeStockPreviewSnapshot,
    admit_receipt_bound_three_stock_preview_session,
    lookup_receipt_bound_three_stock_preview_session,
)

CONTRACT_COMMIT = "0d1c6569"
IMPLEMENTATION_COMMIT = "4f9d0456"
IMPLEMENTATION_PATHS = (
    "src/inference/three_stock_preview_cache.py",
    "src/inference/three_stock_preview_session.py",
    "tests/test_u4_5e_receipt_bound_preview_session.py",
)
U7_3N_CACHE_CORE_GIT_OBJECT = "27029a585f67f1edc0cc8d2647eda502ff7b56f9"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_blob(commit: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)


def _git_blob_sha256(commit: str, path: str) -> str:
    return _sha256_bytes(_git_blob(commit, path))


def _git_object(commit: str, path: str) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", f"{commit}:{path}"], cwd=ROOT, text=True
    ).strip()


def _implementation_git_objects_exact() -> bool:
    for path in IMPLEMENTATION_PATHS:
        expected = _git_object(IMPLEMENTATION_COMMIT, path)
        observed = _git_object("HEAD", path)
        if path == "src/inference/three_stock_preview_cache.py":
            if observed not in {expected, U7_3N_CACHE_CORE_GIT_OBJECT}:
                return False
        elif observed != expected:
            return False
    return True


def _stable_id(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(encoded)


def _render_preview(destination: Path, source: Path) -> dict[str, Any]:
    parent = json.loads(
        (ROOT / "configs/u7_3g_three_stock_direct_preview_v1.json").read_text(
            encoding="utf-8"
        )
    )
    render = parent["render"]
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/render_three_stock_preview.py"),
            str(source),
            str(destination),
            "--max-preview-pixels",
            str(render["max_preview_pixels"]),
            "--look-amount",
            str(render["look_amount"]),
            "--seed",
            str(render["seed"]),
            "--tile-size",
            str(render["tile_size"]),
            "--tile-workers",
            str(render["tile_workers"]),
            "--png-compression",
            str(render["png_compression"]),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _snapshot_facts(
    snapshot: ReceiptBoundVerifiedThreeStockPreviewSnapshot,
) -> dict[str, Any]:
    preview = snapshot.preview
    return {
        "cache_index_sha256": snapshot.cache_index_sha256,
        "input_sha256": preview.input_sha256,
        "profile_sha256": preview.profile_sha256,
        "preview_width": preview.preview_width,
        "preview_height": preview.preview_height,
        "preview_pixels": preview.preview_pixels,
        "look_amount": preview.look_amount,
        "rows": [
            {
                "style_id": row.style_id,
                "filename": row.filename,
                "output_sha256": row.output_sha256,
                "payload_bytes": len(row.payload),
                "payload_sha256": _sha256_bytes(row.payload),
            }
            for row in preview.rows
        ],
    }


def _rejects_bound_admission(
    preview: Path,
    source: Path,
    profile: Path,
    receipt: object,
    *,
    message: str,
) -> bool:
    try:
        admit_receipt_bound_three_stock_preview_session(
            preview,
            input_path=source,
            profile_path=profile,
            cache_index_sha256=receipt,  # type: ignore[arg-type]
        )
    except ThreeStockPreviewCacheError as exc:
        return message in str(exc)
    return False


def _write_mutation(index_path: Path, original: bytes, mutation: str) -> None:
    index = json.loads(original.decode("utf-8"))
    if mutation == "rows[1].output_sha256":
        index["rows"][1]["output_sha256"] = "b" * 64
    else:
        values: dict[str, object] = {
            "parent_contract_sha256": "b" * 64,
            "preview_width": int(index["preview_width"]) + 1,
            "preview_pixels": int(index["preview_pixels"]) + 1,
            "look_amount": 0.5,
            "claim_ceiling": "foreign cache authority",
        }
        index[mutation] = values[mutation]
    index_path.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _metadata_tamper_gates(
    preview: Path,
    source: Path,
    profile: Path,
    receipt: str,
    fields: list[str],
) -> dict[str, bool]:
    index_path = preview / CACHE_INDEX_NAME
    original = index_path.read_bytes()
    results: dict[str, bool] = {}
    for field in fields:
        _write_mutation(index_path, original, field)
        results[field] = _rejects_bound_admission(
            preview,
            source,
            profile,
            receipt,
            message="receipt drift",
        )
        index_path.write_bytes(original)
    return results


def _media_tamper_gates(
    preview: Path,
    source: Path,
    profile: Path,
    receipt: str,
) -> dict[str, bool]:
    cases = (
        (source, "input drift"),
        (profile, "profile drift"),
        (preview / "portra_400.preview.png", "output drift"),
    )
    results: dict[str, bool] = {}
    for target, message in cases:
        original = target.read_bytes()
        target.write_bytes(original + b"foreign-mutation")
        results[message] = _rejects_bound_admission(
            preview,
            source,
            profile,
            receipt,
            message=message,
        )
        target.write_bytes(original)
    return results


def _warm_lookup_no_io(
    session: object,
) -> tuple[
    ReceiptBoundVerifiedThreeStockPreviewSnapshot,
    float,
    dict[str, int],
]:
    attempts = {
        "path_read_bytes": 0,
        "path_read_text": 0,
        "path_open": 0,
        "path_stat": 0,
        "path_lstat": 0,
        "path_write_bytes": 0,
        "path_write_text": 0,
        "path_touch": 0,
        "path_mkdir": 0,
        "path_unlink": 0,
        "path_rename": 0,
        "path_replace": 0,
        "pixel_decode": 0,
        "render": 0,
    }

    def _forbid(category: str):
        def _forbidden(*_args: object, **_kwargs: object) -> object:
            attempts[category] += 1
            raise AssertionError(f"warm lookup attempted {category}")

        return _forbidden

    started = time.perf_counter()
    with (
        patch.object(Path, "read_bytes", _forbid("path_read_bytes")),
        patch.object(Path, "read_text", _forbid("path_read_text")),
        patch.object(Path, "open", _forbid("path_open")),
        patch.object(Path, "stat", _forbid("path_stat")),
        patch.object(Path, "lstat", _forbid("path_lstat")),
        patch.object(Path, "write_bytes", _forbid("path_write_bytes")),
        patch.object(Path, "write_text", _forbid("path_write_text")),
        patch.object(Path, "touch", _forbid("path_touch")),
        patch.object(Path, "mkdir", _forbid("path_mkdir")),
        patch.object(Path, "unlink", _forbid("path_unlink")),
        patch.object(Path, "rename", _forbid("path_rename")),
        patch.object(Path, "replace", _forbid("path_replace")),
        patch.object(Image, "open", _forbid("pixel_decode")),
        patch.object(
            preview_module,
            "render_three_stock_previews_to_directory",
            _forbid("render"),
        ),
    ):
        snapshot = lookup_receipt_bound_three_stock_preview_session(
            session  # type: ignore[arg-type]
        )
    duration = time.perf_counter() - started
    return snapshot, duration, attempts


def build_report(*, config_path: Path, order: str) -> dict[str, Any]:
    if os.name != "nt":
        raise RuntimeError("U4.5E formal runtime must be Windows")
    if (
        subprocess.run(["git", "diff", "--quiet"], cwd=ROOT, check=False).returncode
        != 0
    ):
        raise RuntimeError("tracked working tree must be clean before U4.5E formal")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_original = ROOT / config["source"]["path"]
    profile_original = ROOT / config["profile"]["path"]
    source_original_sha = sha256_file(source_original)
    profile_original_sha = sha256_file(profile_original)
    if source_original_sha != config["source"]["sha256"]:
        raise RuntimeError("source SHA-256 mismatch")
    if profile_original_sha != config["profile"]["sha256"]:
        raise RuntimeError("profile SHA-256 mismatch")

    scratch_root = ROOT / "tmp"
    scratch_root.mkdir(exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="u4_5e_", dir=scratch_root))
    raw_scratch = str(scratch)
    try:
        source = scratch / source_original.name
        profile = scratch / profile_original.name
        shutil.copyfile(source_original, source)
        shutil.copyfile(profile_original, profile)
        preview = scratch / "preview"
        rendered = _render_preview(preview, source)
        actual_rows = {
            row["style_id"]: row["output_sha256"] for row in rendered["rows"]
        }
        if actual_rows != config["preview"]["rows"]:
            raise RuntimeError("parent preview output identity drift")
        publish_three_stock_preview_cache_index(
            preview,
            input_path=source,
            profile_path=profile,
            parent_contract_sha256=config["preview"]["parent_contract_sha256"],
        )
        index_path = preview / CACHE_INDEX_NAME
        receipt = sha256_file(index_path)

        session_facts: list[dict[str, Any]] = []
        lookup_durations: list[float] = []
        warm_operation_counts = {
            "path_read_bytes": 0,
            "path_read_text": 0,
            "path_open": 0,
            "path_stat": 0,
            "path_lstat": 0,
            "path_write_bytes": 0,
            "path_write_text": 0,
            "path_touch": 0,
            "path_mkdir": 0,
            "path_unlink": 0,
            "path_rename": 0,
            "path_replace": 0,
            "pixel_decode": 0,
            "render": 0,
        }
        admissions = list(
            range(config["formal"]["fresh_session_admissions_per_process"])
        )
        if order == "reverse":
            admissions.reverse()
        final_session = None
        for _admission in admissions:
            session = admit_receipt_bound_three_stock_preview_session(
                preview,
                input_path=source,
                profile_path=profile,
                cache_index_sha256=receipt,
            )
            final_session = session
            snapshots = []
            for _lookup in range(config["formal"]["warm_lookups_per_admission"]):
                snapshot, duration, attempts = _warm_lookup_no_io(session)
                lookup_durations.append(duration)
                for category, count in attempts.items():
                    warm_operation_counts[category] += count
                snapshots.append(_snapshot_facts(snapshot))
            if any(item != snapshots[0] for item in snapshots[1:]):
                raise RuntimeError("warm lookup identity drift")
            session_facts.append(snapshots[0])
        if final_session is None:
            raise RuntimeError("no session admissions requested")

        invalid_receipts = {
            "wrong-lowercase-64hex": _rejects_bound_admission(
                preview, source, profile, "b" * 64, message="receipt drift"
            ),
            "uppercase": _rejects_bound_admission(
                preview, source, profile, "A" * 64, message="lowercase 64-hex"
            ),
            "short": _rejects_bound_admission(
                preview, source, profile, "a" * 63, message="lowercase 64-hex"
            ),
            "non-hex": _rejects_bound_admission(
                preview, source, profile, "g" * 64, message="lowercase 64-hex"
            ),
            "non-string": _rejects_bound_admission(
                preview, source, profile, 7, message="lowercase 64-hex"
            ),
        }
        metadata_tamper = _metadata_tamper_gates(
            preview,
            source,
            profile,
            receipt,
            config["formal"]["required_tamper_fields"],
        )
        media_tamper = _media_tamper_gates(preview, source, profile, receipt)

        before_replacement = _snapshot_facts(
            lookup_receipt_bound_three_stock_preview_session(final_session)
        )
        original_index = index_path.read_bytes()
        index_path.write_bytes(original_index + b"\n")
        after_replacement = _snapshot_facts(
            lookup_receipt_bound_three_stock_preview_session(final_session)
        )
        replacement_readmission_rejected = _rejects_bound_admission(
            preview,
            source,
            profile,
            receipt,
            message="receipt drift",
        )
        index_path.write_bytes(original_index)

        canonical = session_facts[0]
        canonical_rows = {
            row["style_id"]: row["payload_sha256"] for row in canonical["rows"]
        }
        maximum_lookup = max(lookup_durations)
        gates = {
            "implementation_git_objects_exact": _implementation_git_objects_exact(),
            "receipt_is_exact_index_bytes": receipt == canonical["cache_index_sha256"],
            "canonical_preview_payloads_exact": canonical_rows
            == config["preview"]["rows"],
            "fresh_session_identity": all(
                facts == canonical for facts in session_facts[1:]
            ),
            "warm_lookup_latency": maximum_lookup
            <= config["formal"]["maximum_warm_lookup_seconds"],
            "zero_warm_filesystem_reads": all(
                warm_operation_counts[key] == 0
                for key in (
                    "path_read_bytes",
                    "path_read_text",
                    "path_open",
                    "path_stat",
                    "path_lstat",
                )
            ),
            "zero_warm_filesystem_writes": all(
                warm_operation_counts[key] == 0
                for key in (
                    "path_write_bytes",
                    "path_write_text",
                    "path_touch",
                    "path_mkdir",
                    "path_unlink",
                    "path_rename",
                    "path_replace",
                )
            ),
            "zero_warm_pixel_decode": warm_operation_counts["pixel_decode"] == 0,
            "zero_warm_render": warm_operation_counts["render"] == 0,
            "invalid_receipts_reject": all(invalid_receipts.values()),
            "metadata_and_expected_hash_mutations_reject": all(
                metadata_tamper.values()
            ),
            "current_session_survives_index_replacement": before_replacement
            == after_replacement,
            "replacement_readmission_rejects": replacement_readmission_rejected,
            "media_mutations_retain_rejection": all(media_tamper.values()),
            "source_profile_immutable": sha256_file(source_original)
            == source_original_sha
            and sha256_file(profile_original) == profile_original_sha,
        }
        scientific = {
            "schema": "kmcfm.u4-5e-receipt-bound-preview-session-scientific.v1",
            "contract_config_git_sha256": _git_blob_sha256(
                CONTRACT_COMMIT,
                "configs/u4_5e_receipt_bound_preview_session_v1.json",
            ),
            "implementation_commit": subprocess.check_output(
                ["git", "rev-parse", IMPLEMENTATION_COMMIT], cwd=ROOT, text=True
            ).strip(),
            "snapshot": canonical,
            "session_admissions": len(session_facts),
            "warm_lookups": len(lookup_durations),
            "warm_operation_counts": warm_operation_counts,
            "invalid_receipts": invalid_receipts,
            "metadata_tamper": metadata_tamper,
            "media_tamper": media_tamper,
            "gates": gates,
            "claim_ceiling": config["claim_ceiling"],
        }
        report = {
            "schema": "kmcfm.u4-5e-receipt-bound-preview-session-result.v1",
            "status": (
                "PASS_PRIVATE_RECEIPT_BOUND_PREVIEW_SESSION"
                if all(gates.values())
                else "FAIL_CLOSED_RECEIPT_BOUND_PREVIEW_SESSION"
            ),
            "order": order,
            "execution_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "scientific": scientific,
            "scientific_stable_id": _stable_id(scientific),
            "timing": {
                "warm_lookup_wall_seconds": lookup_durations,
                "maximum_warm_lookup_seconds": maximum_lookup,
                "frozen_maximum_warm_lookup_seconds": config["formal"][
                    "maximum_warm_lookup_seconds"
                ],
            },
        }
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    report["owned_runtime_residue_zero"] = not Path(raw_scratch).exists()
    if not report["owned_runtime_residue_zero"]:
        report["status"] = "FAIL_CLOSED_RECEIPT_BOUND_PREVIEW_SESSION"
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u4_5e_receipt_bound_preview_session_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), default="forward")
    args = parser.parse_args()
    report = build_report(config_path=args.config, order=args.order)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if report["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
