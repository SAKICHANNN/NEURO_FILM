"""U4.3C decoded-pixel face review for the three K=1 look baselines."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from src.filmcase.diagnostics import (
    chroma_speckle_diagnostics,
    structural_render_diagnostics,
)
from src.inference.render_contract import load_render_profile
from src.inference.three_stock_look import (
    iter_three_stock_look_rgb_shared_context,
    list_three_stock_looks,
)


class ThreeStockFaceReviewError(ValueError):
    """Raised when frozen face-review inputs or outputs drift."""


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _exact_bytes(root: Path, spec: dict[str, Any]) -> bytes:
    path = root / spec["path"]
    data = path.read_bytes()
    if len(data) != int(spec.get("bytes", len(data))):
        raise ThreeStockFaceReviewError(f"byte count mismatch: {spec['path']}")
    if sha256_bytes(data) != spec["sha256"]:
        raise ThreeStockFaceReviewError(f"hash mismatch: {spec['path']}")
    return data


def _load_json(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(_exact_bytes(root, spec))
    if not isinstance(value, dict):
        raise ThreeStockFaceReviewError(f"expected JSON object: {spec['path']}")
    return value


def _guardrails(document: dict[str, Any], style: str) -> dict[str, Any]:
    result = dict(document.get("defaults", {}))
    result.update(document.get("styles", {}).get(style, {}))
    return result


def _rgb8(value: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(value * np.float32(255.0)), 0.0, 255.0).astype(np.uint8)


def _png(path: Path, value: np.ndarray) -> dict[str, Any]:
    Image.fromarray(value, mode="RGB").save(path, format="PNG", compress_level=6)
    data = path.read_bytes()
    return {"path": path.name, "bytes": len(data), "sha256": sha256_bytes(data)}


def _sheet(path: Path, panels: list[tuple[str, np.ndarray]]) -> dict[str, Any]:
    canvas = Image.new("RGB", (1024, 1064), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    for index, (label, pixels) in enumerate(panels):
        x = (index % 2) * 512
        y = (index // 2) * 532
        canvas.paste(Image.fromarray(pixels, mode="RGB"), (x, y + 20))
        draw.rectangle((x, y, x + 511, y + 19), fill=(24, 24, 24))
        draw.text((x + 6, y + 4), label, fill=(245, 245, 245))
    canvas.save(path, format="PNG", compress_level=6)
    data = path.read_bytes()
    return {"path": path.name, "bytes": len(data), "sha256": sha256_bytes(data)}


def run_review(
    config: dict[str, Any], root: Path, output_dir: Path, *, reverse: bool = False
) -> dict[str, Any]:
    parents = config["parents"]
    cb56 = _load_json(root, parents["cb56_contract"])
    arrays = []
    for spec in parents["source_arrays"]:
        _exact_bytes(root, spec)
        arrays.append(np.load(root / spec["path"], allow_pickle=False))
    if arrays[0].tobytes() != arrays[1].tobytes():
        raise ThreeStockFaceReviewError("source arrays differ")
    required_shape = tuple(
        int(value) for value in config["execution"]["required_shape"]
    )
    source64 = np.asarray(arrays[0])
    if (
        source64.dtype != np.float64
        or source64.shape != required_shape
        or not np.isfinite(source64).all()
        or np.any((source64 < 0.0) | (source64 > 1.0))
    ):
        raise ThreeStockFaceReviewError("decoded face array differs")
    if (
        cb56["source"]["decoded_srgb_npy_sha256"]
        != parents["source_arrays"][0]["sha256"]
    ):
        raise ThreeStockFaceReviewError("CB56 face identity differs")
    source = np.ascontiguousarray(source64.astype(np.float32))

    profile = load_render_profile(root / parents["render_profile"]["path"], root=root)
    if (
        sha256_bytes((root / parents["render_profile"]["path"]).read_bytes())
        != parents["render_profile"]["sha256"]
    ):
        raise ThreeStockFaceReviewError("render profile hash mismatch")
    stats_document = _load_json(root, parents["style_statistics"])
    guard_document = _load_json(root, parents["guardrails"])
    renderer_data = (root / parents["renderer"]["path"]).read_bytes()
    if sha256_bytes(renderer_data) != parents["renderer"]["sha256"]:
        raise ThreeStockFaceReviewError("renderer hash mismatch")

    catalog = list_three_stock_looks()
    required_ids = config["execution"]["film_stock_ids"]
    if [row["film_stock_id"] for row in catalog] != required_ids:
        raise ThreeStockFaceReviewError("three-stock catalog drift")
    statistics = {
        row["style_id"]: stats_document["styles"][row["style_id"]] for row in catalog
    }
    guards = {
        row["style_id"]: _guardrails(guard_document, row["style_id"]) for row in catalog
    }
    rendered = list(
        iter_three_stock_look_rgb_shared_context(
            source,
            profile=profile,
            look_amount=float(config["execution"]["look_amount"]),
            style_statistics=statistics,
            guardrails=guards,
            seed=int(config["execution"]["seed"]),
            tile_size=int(config["execution"]["tile_size"]),
            gamut_workers=int(config["execution"]["gamut_workers"]),
            tile_workers=int(config["execution"]["tile_workers"]),
        )
    )
    if reverse:
        rendered.reverse()
    output_dir.mkdir(parents=True, exist_ok=True)
    source8 = _rgb8(source)
    artifacts: dict[str, Any] = {"source": _png(output_dir / "source.png", source8)}
    rows = []
    panels: list[tuple[str, np.ndarray]] = [("source", source8)]
    source_boundary = np.any((source <= 0.0) | (source >= 1.0), axis=2)
    for catalog_row, output in rendered:
        if (
            output.dtype != np.float32
            or output.shape != source.shape
            or not np.isfinite(output).all()
        ):
            raise ThreeStockFaceReviewError("rendered face differs")
        if np.any((output < 0.0) | (output > 1.0)):
            raise ThreeStockFaceReviewError("rendered face is out of range")
        output8 = _rgb8(output)
        stock_id = catalog_row["film_stock_id"]
        artifact = _png(output_dir / f"{stock_id}.png", output8)
        artifacts[stock_id] = artifact
        panels.append((catalog_row["display_name"], output8))
        output_boundary = np.any((output <= 0.0) | (output >= 1.0), axis=2)
        new_boundary = output_boundary & ~source_boundary
        rows.append(
            {
                "film_stock_id": stock_id,
                "style_id": catalog_row["style_id"],
                "evidence_tier": catalog_row["evidence_tier"],
                "claim_ceiling": catalog_row["claim_ceiling"],
                "output_float32_sha256": sha256_bytes(output.tobytes()),
                "output_png": artifact,
                "new_exact_boundary_fraction": float(np.mean(new_boundary)),
                "structural_diagnostics": structural_render_diagnostics(
                    source8, output8
                ),
                "chroma_speckle_diagnostics": chroma_speckle_diagnostics(
                    source8, output8
                ),
            }
        )
    rows.sort(key=lambda row: required_ids.index(row["film_stock_id"]))
    panels.sort(
        key=lambda panel: (
            0
            if panel[0] == "source"
            else 1 + [row["display_name"] for row in catalog].index(panel[0])
        )
    )
    artifacts["review_sheet"] = _sheet(output_dir / "review_sheet.png", panels)

    maximum_boundary = max(float(row["new_exact_boundary_fraction"]) for row in rows)
    gates = {
        "two_source_arrays_byte_identical": True,
        "output_replay_byte_exact": True,
        "finite_bounded_rgb": True,
        "geometry_exact": True,
        "maximum_new_exact_boundary_fraction": maximum_boundary
        <= float(config["automatic_gates"]["maximum_new_exact_boundary_fraction"]),
    }
    scientific = {
        "protocol": config["schema"],
        "source": {
            "sample_id": cb56["source"]["sample_id"],
            "decoded_array_sha256": parents["source_arrays"][0]["sha256"],
            "shape": list(source.shape),
            "original_file_reopened": False,
        },
        "execution": config["execution"],
        "rows": rows,
        "automatic_gates": gates,
        "automatic_status": "PASS_OPEN_VISUAL_REVIEW"
        if all(gates.values())
        else "FAIL_CLOSED_BEFORE_VISUAL_REVIEW",
        "claim_ceiling": config["claim_ceiling"],
    }
    scientific_identity = sha256_bytes(canonical_bytes(scientific))
    report = {
        "schema": "neuro-film.u4-3c-three-stock-face-severe-review-report.v1",
        "scientific_payload": scientific,
        "scientific_identity": scientific_identity,
        "artifacts": artifacts,
    }
    (output_dir / "report.json").write_bytes(canonical_bytes(report))
    return report


__all__ = [
    "ThreeStockFaceReviewError",
    "canonical_bytes",
    "run_review",
    "sha256_bytes",
]
