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
SINGLE_RENDER_SCHEMA = "neuro-film.sf3-a4-single-stock-confirmation-render-report.v1"
SINGLE_RENDER_SET_SCHEMA = (
    "neuro-film.sf3-a4-independent-single-stock-confirmation-render-set.v1"
)
SEVERE_SCHEMA = "neuro-film.sf3-a5-three-stock-severe-review.v1"
SINGLE_SEVERE_SET_SCHEMA = (
    "neuro-film.sf3-a5-independent-single-stock-severe-review-set.v1"
)
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


def _path_list(value: Path | list[Path] | tuple[Path, ...]) -> list[Path]:
    return [value] if isinstance(value, Path) else list(value)


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
    render_report_path: Path | list[Path] | tuple[Path, ...],
    render_root: Path | list[Path] | tuple[Path, ...],
    severe_review_path: Path | list[Path] | tuple[Path, ...],
    secret: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Create anonymized byte-exact copies only after severe review passes."""

    contract_raw = contract_path.read_bytes()
    contract = load_contract(contract_path)
    _bound_parent(root, contract["parents"]["k1_baseline_contract"])
    _bound_parent(root, contract["parents"]["confirmation_render_contract"])
    stocks = contract["required_stocks"]
    render_paths = _path_list(render_report_path)
    render_roots = _path_list(render_root)
    severe_paths = _path_list(severe_review_path)
    if not (
        len(render_paths) == len(render_roots) == len(severe_paths)
        and len(render_paths) in {1, len(stocks)}
    ):
        raise ThreeStockBlindPackageError(
            "provide either one complete A4 input or one input per required stock"
        )
    assembled_single_stock_set = len(render_paths) > 1
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    render_bindings: list[dict[str, Any]] = []
    severe_bindings: list[dict[str, Any]] = []
    scenes: list[str] | None = None
    observed_stocks: set[str] = set()
    scene_source_hashes: dict[str, str] = {}
    for current_render_path, current_root, current_severe_path in zip(
        render_paths, render_roots, severe_paths, strict=True
    ):
        current_render_raw, render = _object(current_render_path)
        current_severe_raw, severe = _object(current_severe_path)
        if assembled_single_stock_set:
            stock = render.get("stock")
            if (
                render.get("schema") != SINGLE_RENDER_SCHEMA
                or render.get("automatic_pass") is not True
                or render.get("decision")
                != "OPEN_SINGLE_STOCK_K1_SEVERE_ARTIFACT_REVIEW_ONLY_PENDING_THREE_STOCK_CONTROLS"
                or render.get("cross_stock_controls_evaluated") is not False
                or stock not in stocks
                or render.get("stocks") != [stock]
                or stock in observed_stocks
            ):
                raise ThreeStockBlindPackageError(
                    "independent single-stock A4 render did not open severe review"
                )
            observed_stocks.add(stock)
            current_stocks = [stock]
        else:
            if (
                render.get("schema") != RENDER_SCHEMA
                or render.get("automatic_pass") is not True
                or render.get("decision")
                != "OPEN_THREE_STOCK_K1_SEVERE_ARTIFACT_REVIEW_THEN_DOMAIN_SEPARATED_BLIND_DISTINGUISHABILITY"
                or stocks != render.get("stocks")
            ):
                raise ThreeStockBlindPackageError("A4 render did not open blind review")
            current_stocks = stocks
        current_scenes = render.get("confirmation_scenes")
        outputs = render.get("outputs")
        if (
            not isinstance(current_scenes, list)
            or len(current_scenes) != contract["protocol"]["confirmation_scenes"]
            or len(set(current_scenes)) != len(current_scenes)
            or not isinstance(outputs, list)
            or len(outputs) != len(current_scenes) * len(current_stocks)
        ):
            raise ThreeStockBlindPackageError("A4 output inventory drift")
        if scenes is None:
            scenes = list(current_scenes)
        elif scenes != current_scenes:
            raise ThreeStockBlindPackageError(
                "independent A4 reports do not share exact confirmation scenes"
            )
        report_keys: set[tuple[str, str]] = set()
        for row in outputs:
            key = (row.get("scene_id"), row.get("stock_id"))
            relative = Path(str(row.get("relative_path", "")))
            source_sha256 = str(row.get("digital_reference_sha256", ""))
            if (
                key in by_key
                or key[0] not in current_scenes
                or key[1] not in current_stocks
                or relative.is_absolute()
                or ".." in relative.parts
                or len(source_sha256) != 64
            ):
                raise ThreeStockBlindPackageError("A4 output identity drift")
            prior_source_sha256 = scene_source_hashes.setdefault(
                str(key[0]), source_sha256
            )
            if prior_source_sha256 != source_sha256:
                raise ThreeStockBlindPackageError(
                    "A4 stock outputs do not share the same digital source"
                )
            path = current_root.joinpath(*relative.parts)
            if not path.is_file() or _sha256_file(path) != row.get("png_sha256"):
                raise ThreeStockBlindPackageError("A4 output bytes drift")
            by_key[key] = {**row, "path": path}
            report_keys.add(key)
        reviewed = (
            severe.get("reviewed_outputs")
            if severe.get("schema") == SEVERE_SCHEMA
            else None
        )
        if (
            severe.get("render_report_sha256") != _sha256(current_render_raw)
            or severe.get("confirmed_severe_count") != 0
            or not isinstance(reviewed, list)
            or len(reviewed) != len(report_keys)
        ):
            raise ThreeStockBlindPackageError(
                "severe review did not open blind review"
            )
        reviewed_keys: set[tuple[str, str]] = set()
        for row in reviewed:
            key = (row.get("scene_id"), row.get("stock_id"))
            if (
                key in reviewed_keys
                or key not in report_keys
                or row.get("png_sha256") != by_key[key]["png_sha256"]
                or row.get("confirmed_severe") is not False
            ):
                raise ThreeStockBlindPackageError("severe review identity drift")
            reviewed_keys.add(key)
        if reviewed_keys != report_keys:
            raise ThreeStockBlindPackageError("severe review coverage drift")
        render_bindings.append(
            {
                "stock": current_stocks[0] if assembled_single_stock_set else None,
                "sha256": _sha256(current_render_raw),
            }
        )
        severe_bindings.append(
            {
                "stock": current_stocks[0] if assembled_single_stock_set else None,
                "sha256": _sha256(current_severe_raw),
            }
        )
    assert scenes is not None
    if set(by_key) != {(scene, stock) for scene in scenes for stock in stocks}:
        raise ThreeStockBlindPackageError("A4 output coverage drift")
    if assembled_single_stock_set:
        if observed_stocks != set(stocks):
            raise ThreeStockBlindPackageError("independent A4 stock coverage drift")
        render_raw = _canonical(
            {
                "schema": SINGLE_RENDER_SET_SCHEMA,
                "reports": sorted(render_bindings, key=lambda row: row["stock"]),
            }
        )
        severe_raw = _canonical(
            {
                "schema": SINGLE_SEVERE_SET_SCHEMA,
                "reviews": sorted(severe_bindings, key=lambda row: row["stock"]),
            }
        )
    else:
        render_raw = render_paths[0].read_bytes()
        severe_raw = severe_paths[0].read_bytes()

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
        if assembled_single_stock_set:
            core["assembled_from_independent_single_stock_reports"] = True
            core["cross_stock_controls_evaluated_before_blind"] = False
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
        or observations.get("render_report_read") is not False
        or observations.get("stock_labeled_output_paths_read") is not False
        or observations.get("confirmation_target_pixels_read") is not False
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
