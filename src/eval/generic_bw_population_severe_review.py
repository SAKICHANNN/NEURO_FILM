"""Frozen generic B&W population render and severe-review material builder."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.inference import load_render_profile, render_generic_bw_look_rgb
from src.preprocess.output_encode import save_srgb8

from .three_stock_structural_visual_review import _review_boxes, _sheet_bytes


class GenericBwPopulationReviewError(RuntimeError):
    """Raised when a frozen input, renderer, or output invariant drifts."""


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


def _read_bound_json(root: Path, spec: dict[str, Any]) -> Any:
    path = root / spec["path"]
    if not path.is_file() or _sha256_path(path) != spec["sha256"]:
        raise GenericBwPopulationReviewError(f"bound artifact drift: {spec['path']}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_rgb8(path: Path, expected_sha256: str) -> np.ndarray:
    data = path.read_bytes()
    if _sha256_bytes(data) != expected_sha256:
        raise GenericBwPopulationReviewError(f"source hash drift: {path}")
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        if image.mode != "RGB":
            raise GenericBwPopulationReviewError(f"expected RGB8 source: {path}")
        return np.asarray(image, dtype=np.uint8)


def _write_create_only(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(data)
    except FileExistsError as exc:
        raise GenericBwPopulationReviewError(f"refusing to overwrite: {path}") from exc


def _load_sources(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    source = config["source"]
    manifest_path = root / source["manifest"]
    if _sha256_path(manifest_path) != source["manifest_sha256"]:
        raise GenericBwPopulationReviewError("source manifest drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_id = {row.get("id"): row for row in manifest if isinstance(row, dict)}
    source_ids = source["source_ids"]
    if (
        len(source_ids) != int(source["required_count"])
        or len(set(source_ids)) != len(source_ids)
    ):
        raise GenericBwPopulationReviewError("source population drift")
    rows: list[dict[str, Any]] = []
    for source_id in source_ids:
        row = by_id.get(source_id)
        if row is None:
            raise GenericBwPopulationReviewError(f"missing source: {source_id}")
        required = {
            "allowed_use": source["required_allowed_use"],
            "rights_scope": source["required_rights_scope"],
            "decoded_color_state": source["required_color_state"],
        }
        if any(row.get(key) != value for key, value in required.items()):
            raise GenericBwPopulationReviewError(f"source policy drift: {source_id}")
        rows.append(row)
    return rows


def _load_renderer_inputs(
    config: dict[str, Any], root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    renderer = config["renderer"]
    profile_spec = renderer["profile"]
    _read_bound_json(root, profile_spec)
    profile = load_render_profile(root / profile_spec["path"], root=root)
    statistics_doc = _read_bound_json(root, renderer["style_statistics"])
    statistics = statistics_doc["styles"][renderer["style_statistics"]["style_id"]]
    guardrail_doc = _read_bound_json(root, renderer["guardrails"])
    guardrails = dict(guardrail_doc.get("defaults", {}))
    guardrails.update(
        guardrail_doc.get("styles", {}).get(renderer["guardrails"]["style_id"], {})
    )
    return profile, statistics, guardrails


def build_population_review(
    config: dict[str, Any], root: Path, output_dir: Path, *, reverse: bool = False
) -> dict[str, Any]:
    """Render the frozen population and create deterministic review sheets."""
    if output_dir.exists():
        raise FileExistsError("BW2.D2 output directory is create-only")
    if config.get("status") != "FROZEN_BEFORE_POPULATION_RENDER_OR_VISUAL_REVIEW":
        raise GenericBwPopulationReviewError("contract status drift")
    for spec in config["parent_evidence"].values():
        document = _read_bound_json(root, spec)
        required_status = spec.get("required_status")
        if required_status is not None and document.get("status") != required_status:
            raise GenericBwPopulationReviewError("parent evidence status drift")

    sources = _load_sources(config, root)
    profile, statistics, guardrails = _load_renderer_inputs(config, root)
    renderer = config["renderer"]
    ordered = list(reversed(sources)) if reverse else sources
    rows: list[dict[str, Any]] = []
    for source in ordered:
        source_rgb8 = _read_rgb8(root / source["decoded_path"], source["decoded_sha256"])
        source_float = np.ascontiguousarray(source_rgb8.astype(np.float32) / 255.0)
        rendered = render_generic_bw_look_rgb(
            source_float,
            profile=profile,
            look_amount=float(renderer["look_amount"]),
            style_statistics=statistics,
            guardrails=guardrails,
            seed=int(renderer["seed"]),
            tile_size=renderer["tile_size"],
            gamut_workers=int(renderer["gamut_workers"]),
            tile_workers=int(renderer["tile_workers"]),
        )
        if not np.isfinite(rendered).all() or np.any((rendered < 0.0) | (rendered > 1.0)):
            raise GenericBwPopulationReviewError(f"unbounded output: {source['id']}")
        output_path = output_dir / "renders" / f"{source['id']}.png"
        save_srgb8(rendered, output_path)
        output_bytes = output_path.read_bytes()
        with Image.open(io.BytesIO(output_bytes)) as image:
            image.load()
            output_rgb8 = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if output_rgb8.shape != source_rgb8.shape:
            raise GenericBwPopulationReviewError(f"geometry drift: {source['id']}")
        neutral = bool(
            np.array_equal(output_rgb8[..., 0], output_rgb8[..., 1])
            and np.array_equal(output_rgb8[..., 1], output_rgb8[..., 2])
        )
        boundary_fraction = float(np.mean((output_rgb8 == 0) | (output_rgb8 == 255)))
        boxes = _review_boxes(
            source_rgb8, output_rgb8, int(config["review"]["crop_size"])
        )
        sheet = _sheet_bytes(
            source_rgb8, output_rgb8, boxes, source["id"], "generic_bw_full_strength"
        )
        sheet_path = output_dir / "sheets" / f"{source['id']}.png"
        _write_create_only(sheet_path, sheet)
        rows.append(
            {
                "source_id": source["id"],
                "source_sha256": source["decoded_sha256"],
                "width": int(source["width"]),
                "height": int(source["height"]),
                "output": {
                    "relative_path": f"renders/{source['id']}.png",
                    "bytes": len(output_bytes),
                    "png_sha256": _sha256_bytes(output_bytes),
                    "pixel_sha256": _sha256_bytes(output_rgb8.tobytes()),
                    "neutral_axis_rgb8_exact": neutral,
                    "output_boundary_fraction": boundary_fraction,
                },
                "review_boxes": boxes,
                "sheet": {
                    "relative_path": f"sheets/{source['id']}.png",
                    "bytes": len(sheet),
                    "sha256": _sha256_bytes(sheet),
                },
            }
        )
    source_order = config["source"]["source_ids"]
    rows.sort(key=lambda row: source_order.index(row["source_id"]))
    gates = config["automatic_gates"]
    automatic = {
        "source_count": len(rows),
        "output_count": len(rows),
        "all_geometry_exact": all(
            row["width"] > 0 and row["height"] > 0 for row in rows
        ),
        "all_neutral_axis_rgb8_exact": all(
            row["output"]["neutral_axis_rgb8_exact"] for row in rows
        ),
        "maximum_output_boundary_fraction": max(
            row["output"]["output_boundary_fraction"] for row in rows
        ),
        "render_calls": len(rows),
        "network_reads": 0,
    }
    failed: list[str] = []
    if automatic["source_count"] != int(config["source"]["required_count"]):
        failed.append("source count")
    if not automatic["all_geometry_exact"]:
        failed.append("geometry")
    if not automatic["all_neutral_axis_rgb8_exact"]:
        failed.append("neutral axis")
    if automatic["maximum_output_boundary_fraction"] > float(
        gates["maximum_output_boundary_fraction"]
    ):
        failed.append("output boundary")
    if automatic["render_calls"] > int(gates["maximum_render_calls_per_process"]):
        failed.append("render call budget")
    automatic["failed_gates"] = failed
    automatic["automatic_pass"] = not failed
    payload = {
        "source_manifest_sha256": config["source"]["manifest_sha256"],
        "selection_rule": config["review"]["selection_rule"],
        "renderer": {
            "look_id": renderer["look_id"],
            "look_amount": renderer["look_amount"],
            "profile_sha256": renderer["profile"]["sha256"],
            "seed": renderer["seed"],
        },
        "rows": rows,
        "automatic": automatic,
        "adjudications_present": False,
    }
    return {
        "schema_id": "neuro-film.generic-bw-population-review-material.v1",
        "experiment_id": config["experiment_id"],
        "status": (
            "AUTOMATIC_PASS_REVIEW_MATERIAL_READY"
            if not failed
            else "FAIL_CLOSED_AUTOMATIC_GATES"
        ),
        "scientific_payload": payload,
        "scientific_identity": _canonical_sha256(payload),
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = ["GenericBwPopulationReviewError", "build_population_review"]
