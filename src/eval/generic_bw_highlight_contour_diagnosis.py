"""Frozen single-variable diagnosis for generic B&W highlight contours."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from src.inference import (
    load_render_profile,
    render_resolved_safe_lab_rgb,
    resolve_generic_bw_look_parameters,
)
from src.preprocess.output_encode import save_srgb8


class GenericBwContourDiagnosisError(RuntimeError):
    """Raised when the frozen diagnosis inputs or invariants drift."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    data = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return _sha256_bytes(data)


def _read_bound_json(root: Path, path: str, expected_sha256: str) -> Any:
    resolved = root / path
    if not resolved.is_file() or _sha256_path(resolved) != expected_sha256:
        raise GenericBwContourDiagnosisError(f"bound artifact drift: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def _read_rgb8(path: Path, expected_sha256: str) -> np.ndarray:
    data = path.read_bytes()
    if _sha256_bytes(data) != expected_sha256:
        raise GenericBwContourDiagnosisError(f"source hash drift: {path}")
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        if image.mode != "RGB":
            raise GenericBwContourDiagnosisError(f"expected RGB8 source: {path}")
        return np.asarray(image, dtype=np.uint8)


def _fit_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    panel = Image.new("RGB", size, "white")
    fitted = ImageOps.contain(image, size, Image.Resampling.LANCZOS)
    panel.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    return panel


def _sheet_bytes(
    source_id: str,
    source: np.ndarray,
    outputs: list[tuple[str, np.ndarray]],
    crop_box: list[int],
) -> bytes:
    if len(outputs) != 5:
        raise GenericBwContourDiagnosisError("diagnosis requires five variants")
    images = [("SOURCE", Image.fromarray(source, mode="RGB"))] + [
        (variant_id, Image.fromarray(output, mode="RGB"))
        for variant_id, output in outputs
    ]
    canvas = Image.new("RGB", (1536, 1320), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), f"{source_id} | single-variable generic B&W diagnosis", fill="black")
    for index, (label, image) in enumerate(images):
        x = (index % 3) * 512
        y = 34 + (index // 3) * 350
        canvas.paste(_fit_panel(image, (512, 310)), (x, y))
        draw.text((x + 8, y + 314), label, fill="black")
    left, top, right, bottom = crop_box
    draw.text((10, 738), f"exact current-output diagnostic crop={crop_box}", fill="black")
    for index, (label, image) in enumerate(images):
        x = (index % 3) * 512 + 64
        y = 766 + (index // 3) * 270
        crop = image.crop((left, top, right, bottom))
        canvas.paste(_fit_panel(crop, (384, 224)), (x, y))
        draw.text((x, y + 228), label, fill="black")
    stream = io.BytesIO()
    canvas.save(stream, format="PNG", compress_level=6)
    return stream.getvalue()


def _write_create_only(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(data)
    except FileExistsError as exc:
        raise GenericBwContourDiagnosisError(f"refusing to overwrite: {path}") from exc


def run_diagnosis(
    config: dict[str, Any], root: Path, output_dir: Path, *, reverse: bool = False
) -> dict[str, Any]:
    """Render every frozen single-variable ablation and its review sheet."""
    if output_dir.exists():
        raise FileExistsError("BW2.D3 output directory is create-only")
    if config.get("status") != "FROZEN_BEFORE_ABLATION_RENDER_OR_REVIEW":
        raise GenericBwContourDiagnosisError("contract status drift")
    parent_spec = config["parent_evidence"]
    parent = _read_bound_json(
        root, parent_spec["path"], parent_spec["sha256"]
    )
    if parent.get("status") != parent_spec["required_status"]:
        raise GenericBwContourDiagnosisError("parent evidence status drift")
    formal = parent["formal_reports"]
    parent_report = _read_bound_json(
        root, formal["forward_path"], formal["sha256"]
    )
    if parent_report.get("scientific_identity") != formal["scientific_identity"]:
        raise GenericBwContourDiagnosisError("parent report identity drift")
    parent_rows = {
        row["source_id"]: row for row in parent_report["scientific_payload"]["rows"]
    }

    source_spec = config["source"]
    manifest_path = root / source_spec["manifest"]
    if _sha256_path(manifest_path) != source_spec["manifest_sha256"]:
        raise GenericBwContourDiagnosisError("source manifest drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_id = {row.get("id"): row for row in manifest if isinstance(row, dict)}
    source_ids = source_spec["confirmed_fail_ids"] + source_spec["frozen_pass_control_ids"]
    if len(source_ids) != 6 or len(set(source_ids)) != 6:
        raise GenericBwContourDiagnosisError("diagnostic source identities drifted")

    renderer = config["renderer"]
    profile_doc = _read_bound_json(
        root, renderer["profile"], renderer["profile_sha256"]
    )
    profile = load_render_profile(root / renderer["profile"], root=root)
    statistics_doc = _read_bound_json(
        root, renderer["style_statistics"], renderer["style_statistics_sha256"]
    )
    guardrail_doc = _read_bound_json(
        root, renderer["guardrails"], renderer["guardrails_sha256"]
    )
    if profile_doc.get("profile_id") != "safe-rich-product-v1":
        raise GenericBwContourDiagnosisError("render profile identity drift")
    statistics = statistics_doc["styles"]["hp5"]
    guardrails = dict(guardrail_doc.get("defaults", {}))
    guardrails.update(guardrail_doc.get("styles", {}).get("hp5", {}))
    execution_style, base_parameters = resolve_generic_bw_look_parameters(
        profile, look_amount=float(renderer["look_amount"])
    )
    variants = config["ordered_single_variable_variants"]
    if [row["variant_id"] for row in variants] != [
        "current",
        "dither_zero",
        "preserve_luma_detail_zero",
        "tone_rolloff_zero",
        "output_margin_zero",
    ]:
        raise GenericBwContourDiagnosisError("variant order drift")

    processing = list(reversed(source_ids)) if reverse else source_ids
    rows: list[dict[str, Any]] = []
    for source_id in processing:
        source = by_id.get(source_id)
        if source is None or source_id not in parent_rows:
            raise GenericBwContourDiagnosisError(f"missing source: {source_id}")
        source_rgb8 = _read_rgb8(root / source["decoded_path"], source["decoded_sha256"])
        source_float = np.ascontiguousarray(source_rgb8.astype(np.float32) / 255.0)
        rendered_rows: list[tuple[str, np.ndarray]] = []
        inventory: list[dict[str, Any]] = []
        current_pixels: np.ndarray | None = None
        for variant in variants:
            parameters = dict(base_parameters)
            parameters.update(variant["overrides"])
            output = render_resolved_safe_lab_rgb(
                source_float,
                style=execution_style,
                style_statistics=statistics,
                style_parameters=parameters,
                guardrails=guardrails,
                seed=int(renderer["seed"]),
            )
            output_path = output_dir / "renders" / variant["variant_id"] / f"{source_id}.png"
            save_srgb8(output, output_path)
            with Image.open(output_path) as image:
                image.load()
                rgb8 = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if rgb8.shape != source_rgb8.shape:
                raise GenericBwContourDiagnosisError("variant geometry drift")
            if not (
                np.array_equal(rgb8[..., 0], rgb8[..., 1])
                and np.array_equal(rgb8[..., 1], rgb8[..., 2])
            ):
                raise GenericBwContourDiagnosisError("variant neutral-axis drift")
            if variant["variant_id"] == "current":
                current_pixels = rgb8
                expected = parent_rows[source_id]["output"]
                if _sha256_bytes(rgb8.tobytes()) != expected["pixel_sha256"]:
                    raise GenericBwContourDiagnosisError("current output drift")
            assert current_pixels is not None
            difference = np.abs(rgb8.astype(np.int16) - current_pixels.astype(np.int16))
            inventory.append(
                {
                    "variant_id": variant["variant_id"],
                    "overrides": variant["overrides"],
                    "relative_path": f"renders/{variant['variant_id']}/{source_id}.png",
                    "png_sha256": _sha256_path(output_path),
                    "pixel_sha256": _sha256_bytes(rgb8.tobytes()),
                    "maximum_abs_code_difference_from_current": int(difference.max()),
                    "mean_abs_code_difference_from_current": float(difference.mean()),
                    "output_boundary_fraction": float(
                        np.mean((rgb8 == 0) | (rgb8 == 255))
                    ),
                }
            )
            rendered_rows.append((variant["variant_id"], rgb8))
        crop_box = parent_rows[source_id]["review_boxes"]["flat_new_high_frequency"]
        sheet = _sheet_bytes(source_id, source_rgb8, rendered_rows, crop_box)
        sheet_path = output_dir / "sheets" / f"{source_id}.png"
        _write_create_only(sheet_path, sheet)
        rows.append(
            {
                "source_id": source_id,
                "parent_decision": (
                    "FAIL_CONFIRMED_SEVERE_ARTIFACT"
                    if source_id in source_spec["confirmed_fail_ids"]
                    else "PASS_NO_CONFIRMED_SEVERE_ARTIFACT"
                ),
                "source_sha256": source["decoded_sha256"],
                "crop_box": crop_box,
                "variants": inventory,
                "sheet": {
                    "relative_path": f"sheets/{source_id}.png",
                    "bytes": len(sheet),
                    "sha256": _sha256_bytes(sheet),
                },
            }
        )
    rows.sort(key=lambda row: source_ids.index(row["source_id"]))
    payload = {
        "parent_evidence_sha256": parent_spec["sha256"],
        "parent_report_sha256": formal["sha256"],
        "source_manifest_sha256": source_spec["manifest_sha256"],
        "variant_ids": [row["variant_id"] for row in variants],
        "source_count": len(rows),
        "render_count": len(rows) * len(variants),
        "rows": rows,
        "network_reads": 0,
        "adjudications_present": False,
    }
    return {
        "schema_id": "neuro-film.generic-bw-highlight-contour-diagnosis-material.v1",
        "experiment_id": config["experiment_id"],
        "status": "DIAGNOSTIC_MATERIAL_READY",
        "scientific_payload": payload,
        "scientific_identity": _canonical_sha256(payload),
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = ["GenericBwContourDiagnosisError", "run_diagnosis"]
