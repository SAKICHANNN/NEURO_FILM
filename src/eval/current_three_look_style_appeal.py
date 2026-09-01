"""U4.2A blinded product-value audit for the current three Look Approximations."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from src.filmcase.vision_audit import (
    BlindAuditPlan,
    VisionAuditError,
    aggregate_reviews,
    build_blind_audit,
)
from src.inference.product_desktop import ProductDesktopWorkflow
from src.inference.render_contract import sha256_file


class CurrentThreeLookAuditError(ValueError):
    """Raised when a frozen U4.2A identity, asset, observation or gate drifts."""


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_locked_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    payload = path.read_bytes()
    if sha256_bytes(payload) != expected_sha256:
        raise CurrentThreeLookAuditError(f"hash mismatch: {path}")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise CurrentThreeLookAuditError(f"JSON object required: {path}")
    return value


def _decode_rgb8(path: Path) -> np.ndarray:
    decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if (
        decoded is None
        or decoded.ndim != 3
        or decoded.shape[2] not in (3, 4)
        or decoded.dtype not in (np.uint8, np.uint16)
    ):
        raise CurrentThreeLookAuditError(f"unsupported review raster: {path}")
    if decoded.shape[2] == 4:
        decoded = decoded[:, :, :3]
    rgb = decoded[:, :, ::-1]
    if rgb.dtype == np.uint16:
        rgb = np.rint(rgb.astype(np.float64) * (255.0 / 65535.0)).astype(np.uint8)
    return np.ascontiguousarray(rgb)


def _fit_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    panel = Image.new("RGB", size, "#1e1e1e")
    fitted = ImageOps.contain(image, size, Image.Resampling.LANCZOS)
    panel.paste(
        fitted,
        ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2),
    )
    return panel


def contact_sheet_bytes(
    *,
    round_index: int,
    sample_id: str,
    labeled_images: Sequence[tuple[str, np.ndarray]],
) -> bytes:
    """Build one anonymous 2x2 sheet without candidate identifiers."""

    if len(labeled_images) != 4 or [row[0] for row in labeled_images] != [
        "A",
        "B",
        "C",
        "D",
    ]:
        raise CurrentThreeLookAuditError("contact sheet requires ordered labels A-D")
    canvas = Image.new("RGB", (1600, 1240), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (16, 12),
        f"U4.2A blind review | round {round_index} | sample {sample_id}",
        fill="black",
    )
    draw.text(
        (16, 31),
        "For each panel record severe(no/yes/uncertain), style(1-5), appeal(1-5).",
        fill="black",
    )
    for index, (label, value) in enumerate(labeled_images):
        if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
            raise CurrentThreeLookAuditError("contact-sheet image must be RGB8")
        x = (index % 2) * 800
        y = 70 + (index // 2) * 585
        canvas.paste(_fit_panel(Image.fromarray(value, mode="RGB"), (800, 550)), (x, y))
        draw.rectangle((x, y, x + 44, y + 30), fill="white")
        draw.text((x + 14, y + 8), label, fill="black")
    stream = io.BytesIO()
    canvas.save(stream, format="PNG", compress_level=6)
    return stream.getvalue()


def _verify_bindings(config: Mapping[str, Any], root: Path) -> dict[str, str]:
    actual: dict[str, str] = {}
    for spec in config["implementation_bindings"].values():
        path = str(spec["path"])
        digest = sha256_file(root / path)
        if digest != spec["sha256"]:
            raise CurrentThreeLookAuditError(f"implementation binding drift: {path}")
        actual[path] = digest
    return actual


def load_parent_rows(
    config: Mapping[str, Any], root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the exact U4.1A report/evidence without reading the private review mapping."""

    parents = config["parents"]
    report_spec = parents["u4_1a_report"]
    report = _read_locked_json(root / report_spec["path"], report_spec["sha256"])
    if report.get("status") != report_spec["required_status"]:
        raise CurrentThreeLookAuditError("U4.1A formal status drift")
    evidence_spec = parents["u4_1a_evidence"]
    evidence = _read_locked_json(root / evidence_spec["path"], evidence_spec["sha256"])
    if evidence.get("status") != evidence_spec["required_status"]:
        raise CurrentThreeLookAuditError("U4.1A adjudication status drift")
    payload = report.get("scientific_payload")
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise CurrentThreeLookAuditError("U4.1A rows missing")
    sample_ids = list(config["population"]["sample_ids"])
    style_ids = list(config["arms"][:3])
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("source_id")), str(row.get("style_id")))
        if key in by_key:
            raise CurrentThreeLookAuditError("duplicate U4.1A row")
        by_key[key] = row
    expected = {(sample, style) for sample in sample_ids for style in style_ids}
    if set(by_key) != expected:
        raise CurrentThreeLookAuditError("U4.1A population or arm drift")
    return [
        by_key[(sample, style)] for sample in sample_ids for style in style_ids
    ], evidence


def _materialize_identity(
    *, root: Path, source_path: Path, destination: Path, scratch: Path
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    workflow = ProductDesktopWorkflow(
        root=root,
        scratch_root=scratch,
        python_executable=Path(sys.executable),
        max_preview_pixels=78_000,
        tile_size=128,
        tile_workers=1,
        png_compression=6,
    )
    try:
        state = workflow.render_previews(source_path, 0.0)
        receipt = workflow.export("velvia_50", destination, output_format_id="png16")
        if state.look_amount != 0.0 or receipt.look_amount != 0.0:
            raise CurrentThreeLookAuditError("identity look amount drift")
        return {
            "sample_source_sha256": state.input_sha256,
            "relative_path": destination.name,
            "bytes": destination.stat().st_size,
            "sha256": receipt.output_sha256,
        }
    finally:
        workflow.close()


def materialize_blind_package(
    config: Mapping[str, Any], root: Path, output_dir: Path, *, reverse: bool = False
) -> dict[str, Any]:
    """Create exact identity controls and anonymous sheets; never persist the mapping."""

    root = root.resolve(strict=True)
    output_dir = output_dir.resolve(strict=False)
    if output_dir.exists():
        raise FileExistsError("U4.2A output directory is create-only")
    output_dir.mkdir(parents=True)
    bindings = _verify_bindings(config, root)
    parent_rows, _ = load_parent_rows(config, root)
    sample_ids = list(config["population"]["sample_ids"])
    arms = list(config["arms"])
    parent_media_root = root / config["parents"]["u4_1a_report"]["media_root"]
    row_by_key = {
        (str(row["source_id"]), str(row["style_id"])): row for row in parent_rows
    }
    candidate_assets: dict[tuple[str, str], Path] = {}
    source_paths: dict[str, Path] = {}
    for sample_id in sample_ids:
        for style_id in arms[:3]:
            row = row_by_key[(sample_id, style_id)]
            path = parent_media_root / row["output"]["relative_path"]
            if sha256_file(path) != row["output"]["sha256"]:
                raise CurrentThreeLookAuditError("U4.1A output drift")
            candidate_assets[(style_id, sample_id)] = path
            source = (root / row["source_path"]).resolve(strict=True)
            if sha256_file(source) != row["source_sha256"]:
                raise CurrentThreeLookAuditError("U4.1A source drift")
            source_paths[sample_id] = source

    scratch_root = output_dir / "scratch"
    identity_root = output_dir / "identity"
    identity_rows: list[dict[str, Any]] = []
    ordered_samples = list(reversed(sample_ids)) if reverse else list(sample_ids)
    try:
        for sample_id in ordered_samples:
            destination = identity_root / f"{sample_id}.png"
            row = _materialize_identity(
                root=root,
                source_path=source_paths[sample_id],
                destination=destination,
                scratch=scratch_root / sample_id,
            )
            row["sample_id"] = sample_id
            row["relative_path"] = str(destination.relative_to(output_dir)).replace(
                "\\", "/"
            )
            if row["sample_source_sha256"] != sha256_file(source_paths[sample_id]):
                raise CurrentThreeLookAuditError("identity source binding drift")
            identity_rows.append(row)
            candidate_assets[(arms[3], sample_id)] = destination
    finally:
        if scratch_root.exists():
            shutil.rmtree(scratch_root)

    plan = build_blind_audit(
        sample_ids,
        arms,
        seed=int(config["blind_protocol"]["seed"]),
        rounds=int(config["blind_protocol"]["rounds"]),
    )
    sheet_rows: list[dict[str, Any]] = []
    mapping_rows = list(plan.mapping)
    if reverse:
        mapping_rows.reverse()
    for mapping in mapping_rows:
        round_index = int(mapping["round"])
        sample_id = str(mapping["sample_id"])
        images = [
            (label, _decode_rgb8(candidate_assets[(candidate, sample_id)]))
            for label, candidate in mapping["label_to_candidate"].items()
        ]
        payload = contact_sheet_bytes(
            round_index=round_index, sample_id=sample_id, labeled_images=images
        )
        relative = Path("review") / f"round_{round_index}" / f"{sample_id}.png"
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        sheet_rows.append(
            {
                "round": round_index,
                "sample_id": sample_id,
                "relative_path": str(relative).replace("\\", "/"),
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
                "labels": ["A", "B", "C", "D"],
            }
        )
    sheet_rows.sort(key=lambda row: (row["round"], sample_ids.index(row["sample_id"])))
    identity_rows.sort(key=lambda row: sample_ids.index(row["sample_id"]))
    public_manifest = {
        "schema": "neuro_film.u4_2a_blind_public_manifest.v1",
        "sample_ids": sample_ids,
        "rounds": 3,
        "anonymous_arm_count": 4,
        "sheets": sheet_rows,
        "claim_boundary": "anonymous autonomous visual review material only",
    }
    manifest_payload = canonical_bytes(public_manifest)
    (output_dir / "public_manifest.json").write_bytes(manifest_payload)
    scientific = {
        "schema": "neuro_film.u4_2a_blind_build.v1",
        "node_id": "U4.2A",
        "parent_report_sha256": config["parents"]["u4_1a_report"]["sha256"],
        "parent_evidence_sha256": config["parents"]["u4_1a_evidence"]["sha256"],
        "implementation_bindings": bindings,
        "identity_rows": identity_rows,
        "public_manifest_sha256": sha256_bytes(manifest_payload),
        "sheet_rows": sheet_rows,
        "mapping_persisted": False,
        "network_reads": 0,
        "status": "READY_FOR_FROZEN_BLIND_OBSERVATIONS",
    }
    return {
        "scientific_payload": scientific,
        "scientific_identity": sha256_bytes(canonical_bytes(scientific)),
        "status": scientific["status"],
    }


def _median(values: Sequence[float]) -> float:
    if not values:
        raise CurrentThreeLookAuditError("empty metric")
    return float(median(values))


def evaluate_observations(
    config: Mapping[str, Any], plan: BlindAuditPlan, observations: Mapping[str, Any]
) -> dict[str, Any]:
    """Unblind a complete frozen observation file and apply only frozen gates."""

    if observations.get("schema") != "neuro_film.u4_2a_blind_observations.v1":
        raise CurrentThreeLookAuditError("observation schema drift")
    rows = observations.get("observations")
    if not isinstance(rows, list):
        raise CurrentThreeLookAuditError("observation rows missing")
    try:
        aggregate = aggregate_reviews(plan, rows)
    except VisionAuditError as exc:
        raise CurrentThreeLookAuditError(str(exc)) from exc
    expected = int(config["blind_protocol"]["expected_observations"])
    if aggregate["reviewed_count"] != expected or aggregate["missing_review_count"]:
        raise CurrentThreeLookAuditError("blind observations are incomplete")
    cases = aggregate["cases"]
    identity_id = str(config["arms"][3])
    sample_ids = list(config["population"]["sample_ids"])
    by_key = {(row["candidate_id"], row["sample_id"]): row for row in cases}
    gates = config["gates"]
    look_results: dict[str, Any] = {}
    for look_id in config["arms"][:3]:
        look_cases = [by_key[(look_id, sample)] for sample in sample_ids]
        identity_cases = [by_key[(identity_id, sample)] for sample in sample_ids]
        style_deltas = [
            float(look["style_median"] - control["style_median"])
            for look, control in zip(look_cases, identity_cases, strict=True)
        ]
        appeal_deltas = [
            float(look["appeal_median"] - control["appeal_median"])
            for look, control in zip(look_cases, identity_cases, strict=True)
        ]
        severe_yes = sum(int(row["severe_yes"]) for row in look_cases)
        severe_uncertain = sum(int(row["uncertain"]) for row in look_cases)
        metrics = {
            "severe_yes": severe_yes,
            "severe_uncertain": severe_uncertain,
            "style_median_of_samples": _median(
                [float(row["style_median"]) for row in look_cases]
            ),
            "paired_style_delta_median_vs_identity": _median(style_deltas),
            "samples_with_style_delta_at_least_one": sum(
                value >= 1.0 for value in style_deltas
            ),
            "paired_appeal_delta_median_vs_identity": _median(appeal_deltas),
            "samples_with_appeal_at_least_identity": sum(
                value >= 0.0 for value in appeal_deltas
            ),
        }
        checks = {
            "severe_yes": severe_yes <= gates["maximum_severe_yes_per_look"],
            "severe_uncertain": severe_uncertain
            <= gates["maximum_severe_uncertain_per_look"],
            "style_median": metrics["style_median_of_samples"]
            >= gates["minimum_style_median_of_samples_per_look"],
            "style_delta_median": metrics["paired_style_delta_median_vs_identity"]
            >= gates["minimum_paired_style_delta_median_vs_identity"],
            "style_delta_count": metrics["samples_with_style_delta_at_least_one"]
            >= gates["minimum_samples_with_style_delta_at_least_one"],
            "appeal_delta_median": metrics["paired_appeal_delta_median_vs_identity"]
            >= gates["minimum_paired_appeal_delta_median_vs_identity"],
            "appeal_noninferior_count": metrics["samples_with_appeal_at_least_identity"]
            >= gates["minimum_samples_with_appeal_at_least_identity"],
        }
        look_results[str(look_id)] = {
            "metrics": metrics,
            "checks": checks,
            "pass": all(checks.values()),
        }
    passing = sum(row["pass"] for row in look_results.values())
    portfolio_pass = passing >= int(gates["required_passing_named_looks"])
    return {
        "aggregate": aggregate,
        "look_results": look_results,
        "passing_named_looks": passing,
        "portfolio_pass": portfolio_pass,
        "status": (
            "PASS_PRIVATE_CURRENT_THREE_LOOK_AUTONOMOUS_STYLE_APPEAL_VALUE"
            if portfolio_pass
            else "FAIL_CLOSED_CURRENT_THREE_LOOK_AUTONOMOUS_STYLE_APPEAL_VALUE"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "CurrentThreeLookAuditError",
    "canonical_bytes",
    "contact_sheet_bytes",
    "evaluate_observations",
    "load_parent_rows",
    "materialize_blind_package",
    "sha256_bytes",
]
