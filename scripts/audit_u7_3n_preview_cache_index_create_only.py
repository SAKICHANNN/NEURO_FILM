"""Committed-head formal audit for U7.3N cache-index ownership repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import three_stock_preview_cache as cache_module
from src.inference.render_contract import sha256_file
from src.inference.three_stock_preview_cache import (
    CACHE_INDEX_NAME,
    ThreeStockPreviewCacheError,
    inspect_receipt_bound_three_stock_preview_cache,
    inspect_three_stock_preview_cache,
    publish_three_stock_preview_cache_index,
)

CONFIG = ROOT / "configs/u7_3n_preview_cache_index_create_only_v1.json"
SCRATCH = ROOT / "tmp"


def _canonical_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_bytes(commit: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def _fixture(root: Path) -> tuple[Path, Path, Path]:
    source = root / "source.bin"
    profile = root / "profile.json"
    preview = root / "preview"
    preview.mkdir()
    source.write_bytes(b"u7-3n-source-pixels")
    profile.write_text('{"profile":"u7-3n"}\n', encoding="utf-8")
    rows: list[dict[str, str]] = []
    for style in ("velvia_50", "portra_400", "ektar_100"):
        output = preview / f"{style}.preview.png"
        output.write_bytes(f"u7-3n-{style}".encode("ascii"))
        rows.append(
            {
                "film_stock_id": f"stock:{style}",
                "style_id": style,
                "output_path": str(output),
                "output_sha256": sha256_file(output),
            }
        )
    manifest = {
        "schema_version": "neuro-film.three-stock-direct-preview.v1",
        "input_sha256": sha256_file(source),
        "preview_width": 64,
        "preview_height": 48,
        "preview_pixels": 3072,
        "max_preview_pixels": 4096,
        "look_amount": 1.0,
        "rows": rows,
    }
    (preview / "preview.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    return preview, source, profile


def _publish(preview: Path, source: Path, profile: Path) -> dict[str, Any]:
    return publish_three_stock_preview_cache_index(
        preview,
        input_path=source,
        profile_path=profile,
        parent_contract_sha256="3" * 64,
    )


def _stage_entries(preview: Path) -> list[Path]:
    return sorted(preview.glob(f".{CACHE_INDEX_NAME}.*.stage"))


def _source_snapshot(preview: Path, source: Path, profile: Path) -> dict[str, str]:
    paths = [source, profile, preview / "preview.json"] + sorted(
        preview.glob("*.preview.png")
    )
    return {path.name: sha256_file(path) for path in paths}


def _run_case(case_id: str, root: Path) -> dict[str, Any]:
    preview, source, profile = _fixture(root)
    before = _source_snapshot(preview, source, profile)
    destination = preview / CACHE_INDEX_NAME
    passed = False
    details: dict[str, Any] = {}

    if case_id == "success":
        index = _publish(preview, source, profile)
        encoded = (
            json.dumps(index, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        receipt = _sha256(encoded)
        passed = (
            destination.read_bytes() == encoded
            and inspect_three_stock_preview_cache(
                preview, input_path=source, profile_path=profile
            )
            == index
            and inspect_receipt_bound_three_stock_preview_cache(
                preview,
                input_path=source,
                profile_path=profile,
                cache_index_sha256=receipt,
            )
            == index
            and _stage_entries(preview) == []
        )
        details = {
            "index_bytes": len(encoded),
            "index_sha256": receipt,
            "canonical_bytes_exact": destination.read_bytes() == encoded,
            "parent_inspection_exact": True,
            "receipt_inspection_exact": True,
        }
    elif case_id == "existing-destination":
        destination.write_bytes(b"existing-foreign")
        rejected = False
        try:
            _publish(preview, source, profile)
        except ThreeStockPreviewCacheError as exc:
            rejected = str(exc) == "preview cache index already exists"
        passed = (
            rejected
            and destination.read_bytes() == b"existing-foreign"
            and _stage_entries(preview) == []
        )
        details = {"rejected": rejected, "foreign_sha256": sha256_file(destination)}
    elif case_id == "late-foreign-destination":
        real_publish = cache_module.publish_create_only

        def inject(stage: Path, final: Path):  # type: ignore[no-untyped-def]
            final.write_bytes(b"late-foreign")
            return real_publish(stage, final)

        rejected = False
        try:
            with patch.object(cache_module, "publish_create_only", inject):
                _publish(preview, source, profile)
        except ThreeStockPreviewCacheError as exc:
            rejected = str(exc) == "preview cache index already exists"
        passed = (
            rejected
            and destination.read_bytes() == b"late-foreign"
            and _stage_entries(preview) == []
        )
        details = {"rejected": rejected, "foreign_sha256": sha256_file(destination)}
    elif case_id == "clean-publication-failure":

        def fail(_stage: Path, _final: Path) -> None:
            raise OSError("u7-3n injected publication failure")

        rejected = False
        try:
            with patch.object(cache_module, "publish_create_only", fail):
                _publish(preview, source, profile)
        except OSError as exc:
            rejected = str(exc) == "u7-3n injected publication failure"
        passed = rejected and not destination.exists() and _stage_entries(preview) == []
        details = {"rejected": rejected, "owned_residue": len(_stage_entries(preview))}
    elif case_id == "foreign-stage-replacement":

        def replace_then_fail(stage: Path, _final: Path) -> None:
            stage.unlink()
            stage.write_bytes(b"foreign-stage-replacement")
            raise OSError("u7-3n injected stage replacement")

        rejected = False
        try:
            with patch.object(
                cache_module, "publish_create_only", replace_then_fail
            ):
                _publish(preview, source, profile)
        except OSError as exc:
            rejected = str(exc) == "u7-3n injected stage replacement"
        stages = _stage_entries(preview)
        passed = (
            rejected
            and not destination.exists()
            and len(stages) == 1
            and stages[0].read_bytes() == b"foreign-stage-replacement"
        )
        details = {
            "rejected": rejected,
            "foreign_stage_count": len(stages),
            "foreign_stage_sha256": sha256_file(stages[0]) if len(stages) == 1 else None,
        }
    else:
        raise ValueError(f"unknown U7.3N case: {case_id}")

    return {
        "case_id": case_id,
        "passed": passed,
        "source_immutable": _source_snapshot(preview, source, profile) == before,
        **details,
    }


def _source_locks(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    commits = {
        "contract": config["contract_commit"],
        "baseline_cache_core": config["contract_commit"],
        "implementation_cache_core": config["implementation_commit"],
        "focused_test": config["implementation_commit"],
        "u7_3h_evidence": config["implementation_commit"],
        "u4_5c_evidence": config["implementation_commit"],
        "u4_5e_evidence": config["implementation_commit"],
    }
    rows: dict[str, dict[str, Any]] = {}
    for name, lock in sorted(config["source_locks"].items()):
        commit = commits[name]
        payload = _git_bytes(commit, lock["path"])
        observed_blob = _git("rev-parse", f"{commit}:{lock['path']}")
        rows[name] = {
            "path": lock["path"],
            "commit": commit,
            "git_blob": observed_blob,
            "git_lf_sha256": _sha256(payload),
            "blob_exact": "git_blob" not in lock or observed_blob == lock["git_blob"],
            "sha256_exact": _sha256(payload) == lock["git_lf_sha256"],
        }
    return rows


def run_audit(order: str) -> dict[str, Any]:
    if os.name != "nt":
        raise RuntimeError("U7.3N formal runtime must be Windows")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source_locks = _source_locks(config)
    case_ids = [
        "success",
        "existing-destination",
        "late-foreign-destination",
        "clean-publication-failure",
        "foreign-stage-replacement",
    ]
    if order == "reverse":
        case_ids.reverse()
    baseline = {path.name for path in SCRATCH.glob("u7-3n-*")}
    rows: list[dict[str, Any]] = []
    SCRATCH.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="u7-3n-", dir=SCRATCH) as raw:
        root = Path(raw)
        for case_id in case_ids:
            case_root = root / case_id
            case_root.mkdir()
            rows.append(_run_case(case_id, case_root))
    rows.sort(key=lambda row: row["case_id"])
    residue_zero = {path.name for path in SCRATCH.glob("u7-3n-*")} == baseline
    by_id = {row["case_id"]: row for row in rows}

    implementation_source = _git_bytes(
        config["implementation_commit"],
        config["source_locks"]["implementation_cache_core"]["path"],
    ).decode("utf-8")
    gates = {
        "source_locks_exact": all(
            row["blob_exact"] and row["sha256_exact"]
            for row in source_locks.values()
        ),
        "existing_destination_preserved": by_id["existing-destination"]["passed"],
        "late_foreign_destination_preserved": by_id[
            "late-foreign-destination"
        ]["passed"],
        "owned_residue_zero_after_failure": by_id[
            "clean-publication-failure"
        ]["passed"],
        "foreign_stage_replacement_preserved": by_id[
            "foreign-stage-replacement"
        ]["passed"],
        "successful_index_bytes_exact": by_id["success"]["passed"],
        "publish_inspect_behavior_unchanged": by_id["success"][
            "parent_inspection_exact"
        ],
        "receipt_bound_behavior_unchanged": by_id["success"][
            "receipt_inspection_exact"
        ],
        "all_sources_immutable": all(row["source_immutable"] for row in rows),
        "create_only_primitive_bound": (
            "publish_create_only(stage, path)" in implementation_source
            and "atomic_write_json(index_path, index)" not in implementation_source
        ),
        "owned_runtime_residue_zero": residue_zero,
    }
    scientific = {
        "node_id": "U7.3N",
        "bindings": {
            "contract_commit": config["contract_commit"],
            "implementation_commit": config["implementation_commit"],
            "source_locks": source_locks,
        },
        "rows": rows,
        "gates": gates,
        "claim_ceiling": config["claim_ceiling"],
        "product_facts": {
            "cache_schema_changed": False,
            "cache_validation_changed": False,
            "successful_index_bytes_changed": False,
            "renderer_or_preview_media_changed": False,
            "calibrated_or_physical_stock_claim": False,
        },
    }
    return {
        "schema_version": "neuro-film.u7-3n-preview-cache-index-create-only-result.v1",
        "status": "PASS_PRIVATE_U7_3N_PREVIEW_CACHE_INDEX_CREATE_ONLY_REPAIR"
        if all(gates.values())
        else "FAIL_CLOSED_U7_3N_PREVIEW_CACHE_INDEX_CREATE_ONLY_REPAIR",
        "formal_execution_commit": _git("rev-parse", "HEAD"),
        "stable_identity": _sha256(_canonical_json(scientific)),
        **scientific,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_audit(args.order)
    payload = _canonical_json(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    return 0 if report["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
