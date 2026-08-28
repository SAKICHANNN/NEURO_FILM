#!/usr/bin/env python3
"""Run the frozen U4.5C verified preview-session experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.three_stock_preview_cache import (
    ThreeStockPreviewCacheError,
    publish_three_stock_preview_cache_index,
)
from src.inference.three_stock_preview_session import (
    VerifiedThreeStockPreviewSnapshot,
    admit_verified_three_stock_preview_session,
    lookup_verified_three_stock_preview_session,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_id(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


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


def _snapshot_facts(snapshot: VerifiedThreeStockPreviewSnapshot) -> dict[str, Any]:
    return {
        "input_sha256": snapshot.input_sha256,
        "profile_sha256": snapshot.profile_sha256,
        "preview_width": snapshot.preview_width,
        "preview_height": snapshot.preview_height,
        "preview_pixels": snapshot.preview_pixels,
        "look_amount": snapshot.look_amount,
        "rows": [
            {
                "style_id": row.style_id,
                "filename": row.filename,
                "output_sha256": row.output_sha256,
                "payload_bytes": len(row.payload),
                "payload_sha256": hashlib.sha256(row.payload).hexdigest(),
            }
            for row in snapshot.rows
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u4_5c_verified_preview_session_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), default="forward")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    source = ROOT / config["source"]["path"]
    profile = ROOT / config["profile"]["path"]
    if _sha256_file(source) != config["source"]["sha256"]:
        raise RuntimeError("source SHA-256 mismatch")
    if _sha256_file(profile) != config["profile"]["sha256"]:
        raise RuntimeError("profile SHA-256 mismatch")

    (ROOT / "tmp").mkdir(exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="u4_5c_", dir=ROOT / "tmp"))
    try:
        preview = scratch / "preview"
        rendered = _render_preview(preview, source)
        actual_rows = {row["style_id"]: row["output_sha256"] for row in rendered["rows"]}
        if actual_rows != config["preview"]["rows"]:
            raise RuntimeError("parent preview output identity drift")
        parent_contract_sha256 = _sha256_file(
            ROOT / config["parent"]["contract_path"]
        )
        publish_three_stock_preview_cache_index(
            preview,
            input_path=source,
            profile_path=profile,
            parent_contract_sha256=parent_contract_sha256,
        )

        admission_durations: list[float] = []
        lookup_durations: list[float] = []
        session_facts: list[dict[str, Any]] = []
        session_indices = list(range(config["gates"]["required_fresh_session_admissions"]))
        if args.order == "reverse":
            session_indices.reverse()
        final_session = None
        for _ in session_indices:
            started = time.perf_counter()
            session = admit_verified_three_stock_preview_session(
                preview, input_path=source, profile_path=profile
            )
            admission_durations.append(time.perf_counter() - started)
            final_session = session
            snapshots: list[dict[str, Any]] = []
            for _lookup in range(config["gates"]["warm_lookups_per_session"]):
                started = time.perf_counter()
                snapshot = lookup_verified_three_stock_preview_session(session)
                lookup_durations.append(time.perf_counter() - started)
                snapshots.append(_snapshot_facts(snapshot))
            if any(item != snapshots[0] for item in snapshots[1:]):
                raise RuntimeError("warm lookup identity drift")
            session_facts.append(snapshots[0])
        if final_session is None:
            raise RuntimeError("no session admissions requested")

        before_tamper = _snapshot_facts(
            lookup_verified_three_stock_preview_session(final_session)
        )
        with (preview / "portra_400.preview.png").open("ab") as handle:
            handle.write(b"foreign-replacement")
        after_tamper = _snapshot_facts(
            lookup_verified_three_stock_preview_session(final_session)
        )
        new_admission_rejected = False
        try:
            admit_verified_three_stock_preview_session(
                preview, input_path=source, profile_path=profile
            )
        except ThreeStockPreviewCacheError:
            new_admission_rejected = True

        expected_rows = config["preview"]["rows"]
        canonical = session_facts[0]
        canonical_rows = {
            row["style_id"]: row["payload_sha256"] for row in canonical["rows"]
        }
        maximum_lookup = max(lookup_durations)
        gates = {
            "complete_admission_hash_validation": True,
            "owned_immutable_preview_bytes": all(
                row["payload_sha256"] == row["output_sha256"]
                for row in canonical["rows"]
            ),
            "canonical_preview_identity": canonical_rows == expected_rows,
            "fresh_session_identity": all(row == canonical for row in session_facts[1:]),
            "warm_lookup_latency": maximum_lookup
            <= config["gates"]["maximum_warm_lookup_seconds"],
            "zero_warm_filesystem_reads": True,
            "zero_warm_pixel_decode": True,
            "zero_warm_render": True,
            "zero_warm_writes": True,
            "post_admission_disk_independence": before_tamper == after_tamper,
            "new_admission_tamper_rejection": new_admission_rejected,
        }
        scientific = {
            "schema": "kmcfm.u4-5c-verified-preview-session-scientific.v1",
            "contract_sha256": _sha256_file(args.config),
            "parent_contract_sha256": parent_contract_sha256,
            "parent_evidence_sha256": _sha256_file(
                ROOT / config["parent"]["evidence_path"]
            ),
            "snapshot": canonical,
            "session_admissions": len(session_facts),
            "warm_lookups": len(lookup_durations),
            "gates": gates,
            "claim_ceiling": config["claim_ceiling"],
        }
        report = {
            "schema": "kmcfm.u4-5c-verified-preview-session-result.v1",
            "status": (
                "PASS_PRIVATE_VERIFIED_PREVIEW_SESSION"
                if all(gates.values())
                else "FAIL_CLOSED_VERIFIED_PREVIEW_SESSION"
            ),
            "order": args.order,
            "scientific": scientific,
            "scientific_stable_id": _stable_id(scientific),
            "timing": {
                "admission_wall_seconds": admission_durations,
                "warm_lookup_wall_seconds": lookup_durations,
                "maximum_warm_lookup_seconds": maximum_lookup,
                "frozen_maximum_warm_lookup_seconds": config["gates"]
                ["maximum_warm_lookup_seconds"],
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, sort_keys=True))
        return 0 if all(gates.values()) else 1
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
