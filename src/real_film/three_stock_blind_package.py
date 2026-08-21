"""Build and adjudicate hash-bound SF3.A5 blind stock assignments."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from src.real_film.three_stock_blind_distinguishability import (
    adjudicate,
    build_mapping,
    load_contract,
)

RENDER_SCHEMA = "neuro-film.sf3-a4-three-stock-confirmation-render-report.v1"
SEVERE_SCHEMA = "neuro-film.sf3-a5-three-stock-severe-review.v1"
PACKAGE_SCHEMA = "neuro-film.sf3-a5-three-stock-blind-package.v1"
SHEET_SCHEMA = "neuro-film.sf3-a5-three-stock-blind-sheet.v1"
MAPPING_SCHEMA = "neuro-film.sf3-a5-three-stock-private-mapping.v1"
OBSERVATIONS_SCHEMA = "neuro-film.sf3-a5-three-stock-blind-observations.v1"
REVEAL_SCHEMA = "neuro-film.sf3-a5-three-stock-mapping-reveal.v1"
REPORT_SCHEMA = "neuro-film.sf3-a5-three-stock-blind-report.v1"


class ThreeStockBlindPackageError(ValueError):
    """Raised when A4 output, severe review, or blind evidence drifts."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _object(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ThreeStockBlindPackageError(f"expected JSON object: {path}")
    return raw, value


def _bound_parent(root: Path, binding: dict[str, Any]) -> None:
    relative = Path(str(binding.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ThreeStockBlindPackageError("invalid parent path")
    path = root.joinpath(*relative.parts)
    if not path.is_file() or _sha256_file(path) != binding.get("sha256"):
        raise ThreeStockBlindPackageError("parent contract identity drift")


def build_package(
    contract_path: Path,
    *,
    root: Path,
    render_report_path: Path,
    render_root: Path,
    severe_review_path: Path,
    secret: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Create anonymized byte-exact copies only after severe review passes."""

    contract_raw = contract_path.read_bytes()
    contract = load_contract(contract_path)
    _bound_parent(root, contract["parents"]["k1_baseline_contract"])
    _bound_parent(root, contract["parents"]["confirmation_render_contract"])
    render_raw, render = _object(render_report_path)
    severe_raw, severe = _object(severe_review_path)
    if (
        render.get("schema") != RENDER_SCHEMA
        or render.get("automatic_pass") is not True
        or render.get("decision")
        != "OPEN_THREE_STOCK_K1_SEVERE_ARTIFACT_REVIEW_THEN_DOMAIN_SEPARATED_BLIND_DISTINGUISHABILITY"
    ):
        raise ThreeStockBlindPackageError("A4 render did not open blind review")
    stocks = contract["required_stocks"]
    scenes = render.get("confirmation_scenes")
    outputs = render.get("outputs")
    if (
        stocks != render.get("stocks")
        or not isinstance(scenes, list)
        or len(scenes) != contract["protocol"]["confirmation_scenes"]
        or len(set(scenes)) != len(scenes)
        or not isinstance(outputs, list)
        or len(outputs) != len(scenes) * len(stocks)
    ):
        raise ThreeStockBlindPackageError("A4 output inventory drift")
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in outputs:
        key = (row.get("scene_id"), row.get("stock_id"))
        relative = Path(str(row.get("relative_path", "")))
        if (
            key in by_key
            or key[0] not in scenes
            or key[1] not in stocks
            or relative.is_absolute()
            or ".." in relative.parts
        ):
            raise ThreeStockBlindPackageError("A4 output identity drift")
        path = render_root.joinpath(*relative.parts)
        if not path.is_file() or _sha256_file(path) != row.get("png_sha256"):
            raise ThreeStockBlindPackageError("A4 output bytes drift")
        by_key[key] = {**row, "path": path}
    if set(by_key) != {(scene, stock) for scene in scenes for stock in stocks}:
        raise ThreeStockBlindPackageError("A4 output coverage drift")

    reviewed = (
        severe.get("reviewed_outputs")
        if severe.get("schema") == SEVERE_SCHEMA
        else None
    )
    if (
        severe.get("render_report_sha256") != _sha256(render_raw)
        or severe.get("confirmed_severe_count") != 0
        or not isinstance(reviewed, list)
        or len(reviewed) != len(by_key)
    ):
        raise ThreeStockBlindPackageError("severe review did not open blind review")
    reviewed_keys = set()
    for row in reviewed:
        key = (row.get("scene_id"), row.get("stock_id"))
        if (
            key in reviewed_keys
            or key not in by_key
            or row.get("png_sha256") != by_key[key]["png_sha256"]
            or row.get("confirmed_severe") is not False
        ):
            raise ThreeStockBlindPackageError("severe review identity drift")
        reviewed_keys.add(key)
    if reviewed_keys != set(by_key):
        raise ThreeStockBlindPackageError("severe review coverage drift")

    logical_outputs = (root / "outputs").resolve()
    output_dir = output_dir.resolve()
    if (
        output_dir == logical_outputs
        or not output_dir.is_relative_to(logical_outputs)
        or output_dir.exists()
    ):
        raise ThreeStockBlindPackageError(
            "blind output must be a new logical outputs subdirectory"
        )
    mapping = build_mapping(scenes, stocks, seed=secret)
    stage = output_dir.with_name(f".{output_dir.name}.stage-{uuid.uuid4().hex}")
    stage.mkdir(parents=True, exist_ok=False)
    try:
        public_rows = []
        private_rows = []
        for map_row in mapping:
            labels = []
            private = {}
            for label, stock in map_row["label_to_stock"].items():
                source = by_key[(map_row["scene_id"], stock)]
                relative = (
                    Path("blind")
                    / f"round-{map_row['round']}"
                    / map_row["scene_id"]
                    / f"{label}.png"
                )
                destination = stage / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source["path"], destination)
                if _sha256_file(destination) != source["png_sha256"]:
                    raise ThreeStockBlindPackageError("blind copy identity drift")
                labels.append(
                    {
                        "label": label,
                        "relative_path": relative.as_posix(),
                        "png_sha256": source["png_sha256"],
                    }
                )
                private[label] = {
                    "stock_id": stock,
                    "source_relative_path": source["relative_path"],
                    "png_sha256": source["png_sha256"],
                }
            public_rows.append(
                {
                    "round": map_row["round"],
                    "scene_id": map_row["scene_id"],
                    "labels": labels,
                }
            )
            private_rows.append(
                {
                    "round": map_row["round"],
                    "scene_id": map_row["scene_id"],
                    "labels": private,
                }
            )
        sheet = {
            "schema": SHEET_SCHEMA,
            "render_report_sha256": _sha256(render_raw),
            "rows": public_rows,
            "target_pixels_included": False,
        }
        private = {
            "schema": MAPPING_SCHEMA,
            "domain": "sf3-a5-v1",
            "rows": private_rows,
        }
        sheet_raw, private_raw = _canonical(sheet), _canonical(private)
        (stage / "public_sheet.json").write_bytes(sheet_raw)
        (stage / "private_mapping.json").write_bytes(private_raw)
        core = {
            "schema": PACKAGE_SCHEMA,
            "experiment_id": contract["experiment_id"],
            "contract_sha256": _sha256(contract_raw),
            "render_report_sha256": _sha256(render_raw),
            "severe_review_sha256": _sha256(severe_raw),
            "public_sheet_sha256": _sha256(sheet_raw),
            "private_mapping_sha256": _sha256(private_raw),
            "scene_count": len(scenes),
            "round_count": 3,
            "blind_png_count": len(mapping) * len(stocks),
            "target_pixel_reads": 0,
            "automatic_pass": True,
            "decision": "OPEN_FROZEN_BLIND_OBSERVATION_COLLECTION_WITH_MAPPING_HIDDEN",
            "claim_ceiling": contract["claim_ceiling"],
        }
        report = {**core, "stable_evidence_id": _sha256(_canonical(core))}
        (stage / "report.json").write_bytes(_canonical(report))
        os.rename(stage, output_dir)
        return report
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def adjudicate_package(
    contract_path: Path,
    *,
    package_report_path: Path,
    public_sheet_path: Path,
    private_mapping_path: Path,
    observations_path: Path,
    reveal_path: Path,
) -> dict[str, Any]:
    """Reveal exact mappings only after observations are externally frozen."""

    contract = load_contract(contract_path)
    package_raw, package = _object(package_report_path)
    sheet_raw, sheet = _object(public_sheet_path)
    mapping_raw, mapping = _object(private_mapping_path)
    observations_raw, observations = _object(observations_path)
    _, reveal = _object(reveal_path)
    if (
        package.get("schema") != PACKAGE_SCHEMA
        or package.get("automatic_pass") is not True
    ):
        raise ThreeStockBlindPackageError("blind package is invalid")
    if _sha256(sheet_raw) != package.get("public_sheet_sha256") or _sha256(
        mapping_raw
    ) != package.get("private_mapping_sha256"):
        raise ThreeStockBlindPackageError("blind package identity drift")
    if sheet.get("schema") != SHEET_SCHEMA or mapping.get("schema") != MAPPING_SCHEMA:
        raise ThreeStockBlindPackageError("blind sheet or mapping schema drift")
    if (
        observations.get("schema") != OBSERVATIONS_SCHEMA
        or observations.get("status") != "observations_frozen_mapping_unread"
        or observations.get("mapping_files_read") is not False
        or observations.get("package_report_sha256") != _sha256(package_raw)
        or observations.get("public_sheet_sha256") != _sha256(sheet_raw)
    ):
        raise ThreeStockBlindPackageError("blind observation boundary drift")
    if (
        reveal.get("schema") != REVEAL_SCHEMA
        or reveal.get("status") != "mapping_revealed_after_observations_commit"
        or reveal.get("observations_sha256") != _sha256(observations_raw)
        or reveal.get("private_mapping_sha256") != _sha256(mapping_raw)
        or len(str(reveal.get("observations_commit", ""))) != 40
    ):
        raise ThreeStockBlindPackageError("mapping reveal boundary drift")
    primitive_mapping = [
        {
            "round": row["round"],
            "scene_id": row["scene_id"],
            "label_to_stock": {
                label: facts["stock_id"] for label, facts in row["labels"].items()
            },
        }
        for row in mapping["rows"]
    ]
    result = adjudicate(
        primitive_mapping, observations.get("assignments", []), gates=contract["gates"]
    )
    passed = result["automatic_pass"]
    core = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "package_report_sha256": _sha256(package_raw),
        "observations_sha256": _sha256(observations_raw),
        "private_mapping_sha256": _sha256(mapping_raw),
        "measurements": result,
        "automatic_pass": passed,
        "decision": contract["decision_if_pass" if passed else "decision_if_fail"],
        "preference_claim_allowed": False,
        "calibrated_stock_claim_allowed": False,
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {**core, "stable_evidence_id": _sha256(_canonical(core))}


__all__ = ["ThreeStockBlindPackageError", "adjudicate_package", "build_package"]
