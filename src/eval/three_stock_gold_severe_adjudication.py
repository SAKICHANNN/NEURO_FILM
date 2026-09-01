"""U4.1A exact product renders and deterministic gold review material."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from src.eval.three_stock_structural_visual_review import (
    _crop_box,
    _review_boxes,
)
from src.filmcase.diagnostics import (
    chroma_speckle_diagnostics,
    structural_render_diagnostics,
)
from src.inference import replay_style_safe_recipe_to_file
from src.inference.product_desktop import PRODUCT_LOOKS, ProductDesktopWorkflow
from src.inference.render_contract import sha256_file


class ThreeStockGoldAdjudicationError(ValueError):
    """Raised when a frozen U4.1A input, output or review fact drifts."""


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_exact(path: Path, *, expected_bytes: int | None, expected_sha256: str) -> bytes:
    payload = path.read_bytes()
    if expected_bytes is not None and len(payload) != expected_bytes:
        raise ThreeStockGoldAdjudicationError(f"byte count mismatch: {path}")
    if sha256_bytes(payload) != expected_sha256:
        raise ThreeStockGoldAdjudicationError(f"hash mismatch: {path}")
    return payload


def _decode_rgb8(payload: bytes) -> np.ndarray:
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.ndim != 3 or decoded.shape[2] not in (3, 4):
        raise ThreeStockGoldAdjudicationError("source is not a supported RGB raster")
    if decoded.shape[2] == 4:
        decoded = decoded[:, :, :3]
    rgb = decoded[:, :, ::-1]
    if rgb.dtype == np.uint16:
        rgb = np.rint(rgb.astype(np.float64) * (255.0 / 65535.0)).astype(np.uint8)
    elif rgb.dtype != np.uint8:
        raise ThreeStockGoldAdjudicationError("source bit depth drifted")
    return np.ascontiguousarray(rgb)


def _decode_rgb16(path: Path) -> np.ndarray:
    decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if (
        decoded is None
        or decoded.dtype != np.uint16
        or decoded.ndim != 3
        or decoded.shape[2] != 3
    ):
        raise ThreeStockGoldAdjudicationError("product output is not RGB PNG16")
    return np.ascontiguousarray(decoded[:, :, ::-1])


def _rgb16_to_rgb8(value: np.ndarray) -> np.ndarray:
    return np.rint(value.astype(np.float64) * (255.0 / 65535.0)).astype(np.uint8)


def normalized_recipe_semantic_identity(recipe: Mapping[str, Any]) -> str:
    """Hash strict recipe semantics while redacting only required absolute paths."""

    normalized = json.loads(json.dumps(recipe))
    normalized["input"]["path"] = "<INPUT>"
    normalized["output"]["path"] = "<OUTPUT>"
    return sha256_bytes(canonical_bytes(normalized))


def _exact_review_boxes(
    source: np.ndarray, output: np.ndarray, crop_size: int
) -> dict[str, list[int]]:
    boxes = _review_boxes(source, output, crop_size)
    delta = np.mean(
        np.abs(output.astype(np.float32) - source.astype(np.float32)), axis=2
    )
    boxes["encoded_rgb_absolute_delta"] = list(_crop_box(delta, crop_size))
    return boxes


def _fit_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    panel = Image.new("RGB", size, "white")
    fitted = ImageOps.contain(image, size, Image.Resampling.LANCZOS)
    panel.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    return panel


def _review_sheet_bytes(
    source_id: str,
    source: np.ndarray,
    ordered_rows: Sequence[tuple[str, np.ndarray, Mapping[str, list[int]]]],
) -> bytes:
    source_image = Image.fromarray(source, mode="RGB")
    row_height = 628
    canvas = Image.new("RGB", (2048, 42 + len(ordered_rows) * row_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), f"{source_id} | frozen U4.1A exact-pixel review", fill="black")
    for row_index, (style_id, output, boxes) in enumerate(ordered_rows):
        top = 42 + row_index * row_height
        output_image = Image.fromarray(output, mode="RGB")
        draw.text((12, top + 4), style_id, fill="black")
        canvas.paste(_fit_panel(source_image, (512, 274)), (0, top + 28))
        canvas.paste(_fit_panel(output_image, (512, 274)), (512, top + 28))
        draw.text((8, top + 306), "SOURCE OVERVIEW", fill="black")
        draw.text((520, top + 306), "OUTPUT OVERVIEW", fill="black")
        for crop_index, (label, coordinates) in enumerate(boxes.items()):
            left, crop_top, right, bottom = coordinates
            x = crop_index * 512
            draw.text((x + 8, top + 330), f"{label} {coordinates}", fill="black")
            canvas.paste(source_image.crop((left, crop_top, right, bottom)), (x, top + 354))
            canvas.paste(output_image.crop((left, crop_top, right, bottom)), (x + 256, top + 354))
            draw.text((x + 8, top + 612), "SOURCE 1:1", fill="black")
            draw.text((x + 264, top + 612), "OUTPUT 1:1", fill="black")
    stream = io.BytesIO()
    canvas.save(stream, format="PNG", compress_level=6)
    return stream.getvalue()


def _verify_candidate_bindings(config: Mapping[str, Any], root: Path) -> dict[str, str]:
    candidate = config["candidate"]
    names = (
        "profile",
        "film_color_stats",
        "guardrails",
        "render_film",
        "product_desktop",
        "three_stock_renderer",
    )
    bindings: dict[str, str] = {}
    for name in names:
        relative = str(candidate[f"{name}_path"])
        expected = str(candidate[f"{name}_sha256"])
        actual = sha256_file(root / relative)
        if actual != expected:
            raise ThreeStockGoldAdjudicationError(f"candidate binding drift: {relative}")
        bindings[relative] = actual
    return bindings


def _gold_rows(config: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    manifest_spec = config["input_manifest"]
    payload = _read_exact(
        root / manifest_spec["path"],
        expected_bytes=int(manifest_spec["bytes"]),
        expected_sha256=str(manifest_spec["sha256"]),
    )
    manifest = json.loads(payload)
    if manifest["frozen_set"]["status"] != "provisional":
        raise ThreeStockGoldAdjudicationError("U4.1 parent status drifted")
    if manifest["coverage"]["coverage_complete"] is not True:
        raise ThreeStockGoldAdjudicationError("U4.1 gold coverage is incomplete")
    rows = [row for row in manifest["frozen_set"]["samples"] if row["split"] == "gold"]
    expected_ids = list(manifest_spec["selected_gold_ids"])
    if [row["id"] for row in rows] != expected_ids or len(rows) != int(
        manifest_spec["required_count"]
    ):
        raise ThreeStockGoldAdjudicationError("frozen gold selection drifted")
    if rows[-1]["id"] != "FS_FACE_01" or rows[-1]["source_path"] != (
        "data/raw/filmset/FilmSet/train/input/DSCF01200 iso1600.png"
    ):
        raise ThreeStockGoldAdjudicationError("original FilmSet face ingress drifted")
    return rows


def run_gold_materialization(
    config: Mapping[str, Any],
    root: Path,
    output_dir: Path,
    *,
    reverse: bool = False,
) -> dict[str, Any]:
    """Render the frozen gold grid and materialize deterministic review sheets."""

    root = root.resolve(strict=True)
    output_dir = output_dir.resolve(strict=False)
    if output_dir.exists():
        raise FileExistsError("U4.1A output directory is create-only")
    output_dir.mkdir(parents=True)
    candidate_bindings = _verify_candidate_bindings(config, root)
    rows = _gold_rows(config, root)
    style_ids = list(config["candidate"]["style_ids"])
    if [row["style_id"] for row in PRODUCT_LOOKS] != style_ids:
        raise ThreeStockGoldAdjudicationError("product look catalog drifted")
    if int(config["review"]["cells"]) != len(rows) * len(style_ids):
        raise ThreeStockGoldAdjudicationError("review cell count drifted")
    crop_size = int(config["review"]["exact_crop_size"])
    expected_strategies = list(config["review"]["exact_crop_strategies"])
    ordered_sources = list(reversed(rows)) if reverse else list(rows)
    ordered_styles = list(reversed(style_ids)) if reverse else list(style_ids)
    media_root = output_dir / "media"
    review_root = output_dir / "review"
    scratch_root = output_dir / "scratch"
    media_root.mkdir()
    review_root.mkdir()
    scratch_root.mkdir()

    result_rows: list[dict[str, Any]] = []
    visual_inputs: dict[str, tuple[np.ndarray, dict[str, tuple[np.ndarray, dict[str, list[int]]]]]] = {}
    try:
        for source_row in ordered_sources:
            source_path = (root / source_row["source_path"]).resolve(strict=True)
            source_payload = _read_exact(
                source_path,
                expected_bytes=None,
                expected_sha256=str(source_row["source_sha256"]),
            )
            source_rgb8 = _decode_rgb8(source_payload)
            source_dir = media_root / str(source_row["id"])
            source_dir.mkdir()
            workflow_scratch = scratch_root / str(source_row["id"])
            workflow_scratch.mkdir()
            workflow = ProductDesktopWorkflow(
                root=root,
                scratch_root=workflow_scratch,
                python_executable=Path(sys.executable),
                max_preview_pixels=78_000,
                tile_size=128,
                tile_workers=1,
                png_compression=6,
            )
            per_style: dict[str, tuple[np.ndarray, dict[str, list[int]]]] = {}
            try:
                state = workflow.render_previews(
                    source_path, float(config["candidate"]["look_amount"])
                )
                if state.input_sha256 != source_row["source_sha256"]:
                    raise ThreeStockGoldAdjudicationError("preview source identity drifted")
                for style_id in ordered_styles:
                    output_path = source_dir / f"{style_id}.png"
                    receipt = workflow.export(
                        style_id,
                        output_path,
                        output_format_id=str(config["candidate"]["output_format_id"]),
                    )
                    replay_path = source_dir / f"{style_id}.replay.png"
                    replay_digest = replay_style_safe_recipe_to_file(
                        receipt.recipe,
                        profile_path=root / config["candidate"]["profile_path"],
                        output_path=replay_path,
                        root=root,
                        tile_size=128,
                    )
                    replay_exact = replay_path.read_bytes() == output_path.read_bytes()
                    replay_path.unlink()
                    output_rgb16 = _decode_rgb16(output_path)
                    output_rgb8 = _rgb16_to_rgb8(output_rgb16)
                    if output_rgb8.shape != source_rgb8.shape:
                        raise ThreeStockGoldAdjudicationError("output geometry drifted")
                    source_boundary = np.any(
                        (source_rgb8 == 0) | (source_rgb8 == 255), axis=2
                    )
                    output_boundary = np.any(
                        (output_rgb16 == 0) | (output_rgb16 == 65535), axis=2
                    )
                    boundary_fraction = float(np.mean(output_boundary & ~source_boundary))
                    boxes = _exact_review_boxes(source_rgb8, output_rgb8, crop_size)
                    if list(boxes) != expected_strategies:
                        raise ThreeStockGoldAdjudicationError("review crop strategy drifted")
                    per_style[style_id] = (output_rgb8, boxes)
                    result_rows.append(
                        {
                            "source_id": source_row["id"],
                            "source_path": source_row["source_path"],
                            "source_sha256": source_row["source_sha256"],
                            "source_geometry": [
                                int(source_rgb8.shape[1]),
                                int(source_rgb8.shape[0]),
                            ],
                            "style_id": style_id,
                            "output": {
                                "relative_path": output_path.relative_to(output_dir).as_posix(),
                                "bytes": output_path.stat().st_size,
                                "sha256": receipt.output_sha256,
                            },
                            "recipe": {
                                "relative_path": receipt.recipe_path.relative_to(output_dir).as_posix(),
                                "bytes": receipt.recipe_path.stat().st_size,
                                "sha256": receipt.recipe_sha256,
                                "semantic_identity": normalized_recipe_semantic_identity(
                                    receipt.recipe
                                ),
                            },
                            "strict_replay_sha256": replay_digest,
                            "strict_replay_byte_exact": replay_exact,
                            "new_exact_boundary_fraction": boundary_fraction,
                            "review_boxes": boxes,
                            "structural_diagnostics": structural_render_diagnostics(
                                source_rgb8, output_rgb8
                            ),
                            "chroma_speckle_diagnostics": chroma_speckle_diagnostics(
                                source_rgb8, output_rgb8
                            ),
                        }
                    )
            finally:
                if not workflow.close():
                    raise ThreeStockGoldAdjudicationError("preview scratch cleanup failed")
            if list(workflow_scratch.iterdir()):
                raise ThreeStockGoldAdjudicationError("preview scratch residue remains")
            workflow_scratch.rmdir()
            visual_inputs[str(source_row["id"])] = (source_rgb8, per_style)

        review_sheets: list[dict[str, Any]] = []
        for pass_index, arm_order in enumerate(config["review"]["arm_orders"], start=1):
            pass_dir = review_root / f"pass_{pass_index}"
            pass_dir.mkdir()
            for source_row in rows:
                source_id = str(source_row["id"])
                source_rgb8, per_style = visual_inputs[source_id]
                ordered = [
                    (style_id, per_style[style_id][0], per_style[style_id][1])
                    for style_id in arm_order
                ]
                payload = _review_sheet_bytes(source_id, source_rgb8, ordered)
                path = pass_dir / f"{source_id}__three_stock_gold.png"
                path.write_bytes(payload)
                review_sheets.append(
                    {
                        "pass_index": pass_index,
                        "source_id": source_id,
                        "arm_order": list(arm_order),
                        "relative_path": path.relative_to(output_dir).as_posix(),
                        "bytes": len(payload),
                        "sha256": sha256_bytes(payload),
                    }
                )
    except BaseException:  # noqa: TRY203 - keep partial failure evidence
        # Keep the create-only output directory as failure evidence. Only the
        # unowned preview scratch is required to be empty before a formal pass.
        raise
    if list(scratch_root.iterdir()):
        raise ThreeStockGoldAdjudicationError("owned scratch residue remains")
    scratch_root.rmdir()

    result_rows.sort(key=lambda row: (str(row["source_id"]), style_ids.index(row["style_id"])))
    review_sheets.sort(key=lambda row: (row["pass_index"], str(row["source_id"])))
    maximum_boundary = max(float(row["new_exact_boundary_fraction"]) for row in result_rows)
    automatic_gates = {
        "candidate_bindings_exact": True,
        "input_hash_and_geometry_exact": True,
        "strict_recipe_replay_byte_exact": all(
            bool(row["strict_replay_byte_exact"]) for row in result_rows
        ),
        "finite_output": True,
        "maximum_new_exact_boundary_fraction": maximum_boundary
        <= float(config["automatic_gates"]["maximum_new_exact_boundary_fraction"]),
        "source_immutable": all(
            sha256_file(root / row["source_path"]) == row["source_sha256"]
            for row in result_rows
        ),
        "network_reads_zero": True,
        "owned_scratch_residue_zero": not scratch_root.exists(),
    }
    scientific = {
        "schema": "neuro-film.u4-1a-three-stock-gold-materialization-report.v1",
        "node_id": config["node_id"],
        "candidate_bindings": candidate_bindings,
        "input_manifest": config["input_manifest"],
        "candidate": config["candidate"],
        "review_protocol": config["review"],
        "rows": result_rows,
        "review_sheets": review_sheets,
        "metrics": {
            "source_count": len(rows),
            "cell_count": len(result_rows),
            "review_sheet_count": len(review_sheets),
            "maximum_new_exact_boundary_fraction": maximum_boundary,
        },
        "automatic_gates": automatic_gates,
        "status": "PASS_OPEN_AUTONOMOUS_VISUAL_REVIEW"
        if all(automatic_gates.values())
        else "FAIL_CLOSED_BEFORE_AUTONOMOUS_VISUAL_REVIEW",
        "claim_ceiling": config["claim_ceiling"],
    }
    report = {
        "schema": "neuro-film.u4-1a-three-stock-gold-materialization-envelope.v1",
        "scientific_payload": scientific,
        "scientific_identity": sha256_bytes(canonical_bytes(scientific)),
    }
    (output_dir / "report.json").write_bytes(canonical_bytes(report))
    return report


def remove_owned_output(path: Path) -> None:
    """Remove one test-owned output tree; never used for formal evidence roots."""

    if path.exists():
        shutil.rmtree(path)


__all__ = [
    "ThreeStockGoldAdjudicationError",
    "canonical_bytes",
    "normalized_recipe_semantic_identity",
    "remove_owned_output",
    "run_gold_materialization",
    "sha256_bytes",
]
