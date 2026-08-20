"""Bounded full-resolution severe-artifact audit for the frozen RF3.D0 proxy outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from src.eval.spcp_global_logit_affine import gradient_p999_ratio

SCHEMA = "neuro-film.rf3-d0s-three-stock-proxy-full-resolution-severe-audit-contract.v1"
REPORT_SCHEMA = "neuro-film.rf3-d0s-three-stock-proxy-full-resolution-severe-audit-report.v1"


class ThreeStockFullResolutionAuditError(RuntimeError):
    """Raised when a bound RF3.D0 artifact or audit invariant drifts."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ThreeStockFullResolutionAuditError(f"invalid JSON: {path}") from exc


def _bound_json(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = root / str(binding["path"])
    if not path.is_file() or _sha256(path) != binding["sha256"]:
        raise ThreeStockFullResolutionAuditError(f"bound artifact drift: {binding['path']}")
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ThreeStockFullResolutionAuditError(f"bound artifact is not an object: {binding['path']}")
    return payload


def load_contract(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ThreeStockFullResolutionAuditError("unexpected RF3.D0S contract schema")
    expected = payload.get("expected", {})
    arms = expected.get("ordered_arms")
    if not isinstance(arms, list) or len(arms) != 4 or len(set(arms)) != 4:
        raise ThreeStockFullResolutionAuditError("RF3.D0S requires four unique ordered arms")
    if expected.get("ao6_role") != "velvia_50_display_proxy_look_approximation_baseline_only":
        raise ThreeStockFullResolutionAuditError("AO6 role drifted beyond Velvia display proxy")
    audit = payload.get("audit", {})
    if audit.get("crop_roles") != ["lowest_source_gradient", "center", "highest_source_gradient"]:
        raise ThreeStockFullResolutionAuditError("crop roles drifted")
    if int(audit.get("crop_size_pixels", 0)) < 64:
        raise ThreeStockFullResolutionAuditError("crop size is too small")
    if float(audit.get("maximum_new_output_boundary_fraction", -1.0)) != 0.0:
        raise ThreeStockFullResolutionAuditError("new-boundary severe gate must remain exact zero")
    return payload


def _rgb8(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise ThreeStockFullResolutionAuditError(f"cannot decode RGB image: {path}") from exc
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ThreeStockFullResolutionAuditError(f"invalid RGB shape: {path}")
    return rgb


def _crop_coordinates(source: np.ndarray, size: int, stride: int) -> dict[str, list[int]]:
    height, width = source.shape[:2]
    if min(height, width) < size:
        raise ThreeStockFullResolutionAuditError("source is smaller than frozen crop")
    luma = source.astype(np.float32) @ np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)
    gradient = np.zeros_like(luma)
    gradient[:, 1:] += np.abs(np.diff(luma, axis=1))
    gradient[1:, :] += np.abs(np.diff(luma, axis=0))
    positions: list[tuple[float, int, int]] = []
    ys = list(range(0, height - size + 1, stride))
    xs = list(range(0, width - size + 1, stride))
    if ys[-1] != height - size:
        ys.append(height - size)
    if xs[-1] != width - size:
        xs.append(width - size)
    for y in ys:
        for x in xs:
            positions.append((float(np.mean(gradient[y : y + size, x : x + size])), y, x))
    positions.sort(key=lambda row: (row[0], row[1], row[2]))
    center_y = (height - size) // 2
    center_x = (width - size) // 2
    return {
        "lowest_source_gradient": [positions[0][2], positions[0][1], size, size],
        "center": [center_x, center_y, size, size],
        "highest_source_gradient": [positions[-1][2], positions[-1][1], size, size],
    }


def _boundary_fraction(source: np.ndarray, output: np.ndarray) -> float:
    source_boundary = (source == 0) | (source == 255)
    output_boundary = (output == 0) | (output == 255)
    return float(np.mean(output_boundary & ~source_boundary))


def _crop_diagnostics(array: np.ndarray, box: list[int]) -> dict[str, float | int]:
    x, y, width, height = box
    crop = array[y : y + height, x : x + width]
    luma = np.rint(crop.astype(np.float32) @ np.asarray((0.2126, 0.7152, 0.0722))).astype(np.uint8)
    return {
        "occupied_luma_bins": int(np.unique(luma).size),
        "largest_luma_bin_fraction": float(np.max(np.bincount(luma.ravel(), minlength=256)) / luma.size),
    }


def _make_sheet(
    rows: list[dict[str, Any]],
    images: dict[tuple[str, str], np.ndarray],
    arms: list[str],
    crop_roles: list[str],
    crop_size: int,
    path: Path,
) -> str:
    columns = ["source", *arms]
    label_height = 20
    row_height = crop_size + label_height
    canvas = Image.new("RGB", (len(columns) * len(crop_roles) * crop_size, len(rows) * row_height), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, row in enumerate(rows):
        source_id = row["source_id"]
        for role_index, role in enumerate(crop_roles):
            x, y, width, height = row["crop_coordinates"][role]
            for column_index, arm in enumerate(columns):
                array = images[(source_id, arm)]
                crop = Image.fromarray(array[y : y + height, x : x + width], mode="RGB")
                left = (role_index * len(columns) + column_index) * crop_size
                top = row_index * row_height + label_height
                canvas.paste(crop, (left, top))
                label = f"{source_id[:12]} {role[:4]} {arm[:9]}"
                draw.text((left + 2, row_index * row_height + 3), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", compress_level=6)
    return _sha256(path)


def evaluate(contract_path: Path, root: Path, output_dir: Path) -> dict[str, Any]:
    contract = load_contract(contract_path)
    d0_contract = _bound_json(root, contract["parents"]["rf3_d0_contract"])
    d0_evidence = _bound_json(root, contract["parents"]["rf3_d0_evidence"])
    d0_report = _bound_json(root, contract["parents"]["rf3_d0_report"])
    if d0_evidence.get("status") != contract["parents"]["rf3_d0_evidence"]["required_status"]:
        raise ThreeStockFullResolutionAuditError("RF3.D0 evidence status drift")
    if d0_report.get("scientific_identity") != contract["parents"]["rf3_d0_report"]["required_scientific_identity"]:
        raise ThreeStockFullResolutionAuditError("RF3.D0 scientific identity drift")

    source_binding = d0_contract["source"]
    source_manifest_path = root / source_binding["manifest"]
    if _sha256(source_manifest_path) != source_binding["manifest_sha256"]:
        raise ThreeStockFullResolutionAuditError("RF3.D0 source manifest drift")
    source_manifest = _read_json(source_manifest_path)
    source_by_id = {row["id"]: row for row in source_manifest}
    arms = contract["expected"]["ordered_arms"]
    report_rows = d0_report.get("rows", [])
    if len(report_rows) != int(contract["expected"]["source_rows"]):
        raise ThreeStockFullResolutionAuditError("RF3.D0 source row count drift")

    run_root = (root / contract["parents"]["rf3_d0_report"]["path"]).parent
    crop_size = int(contract["audit"]["crop_size_pixels"])
    stride = int(contract["audit"]["crop_selection_stride_pixels"])
    images: dict[tuple[str, str], np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for report_row in report_rows:
        source_id = report_row["source_id"]
        source_row = source_by_id.get(source_id)
        if source_row is None:
            raise ThreeStockFullResolutionAuditError(f"missing source row: {source_id}")
        source_path = root / source_row["decoded_path"]
        if _sha256(source_path) != source_row["decoded_sha256"]:
            raise ThreeStockFullResolutionAuditError(f"source hash drift: {source_id}")
        source = _rgb8(source_path)
        images[(source_id, "source")] = source
        coordinates = _crop_coordinates(source, crop_size, stride)
        output_rows: dict[str, Any] = {}
        for arm in arms:
            inventory = report_row.get("outputs", {}).get(arm)
            if inventory is None:
                raise ThreeStockFullResolutionAuditError(f"missing arm: {source_id}/{arm}")
            path = run_root / inventory["relative_path"]
            png_sha = _sha256(path)
            output = _rgb8(path)
            pixel_sha = hashlib.sha256(output.tobytes()).hexdigest()
            geometry_match = output.shape == source.shape
            new_boundary = _boundary_fraction(source, output) if geometry_match else float("inf")
            if png_sha != inventory["png_sha256"]:
                failures.append(f"png_hash:{source_id}/{arm}")
            if pixel_sha != inventory["pixel_sha256"]:
                failures.append(f"pixel_hash:{source_id}/{arm}")
            if not geometry_match:
                failures.append(f"geometry:{source_id}/{arm}")
            if new_boundary > float(contract["audit"]["maximum_new_output_boundary_fraction"]):
                failures.append(f"new_boundary:{source_id}/{arm}")
            images[(source_id, arm)] = output
            output_rows[arm] = {
                "png_sha256": png_sha,
                "pixel_sha256": pixel_sha,
                "geometry_match": geometry_match,
                "new_output_boundary_fraction": new_boundary,
                "gradient_p999_ratio_vs_source": gradient_p999_ratio(source / 255.0, output / 255.0),
                "lowest_gradient_crop_diagnostics": _crop_diagnostics(
                    output, coordinates["lowest_source_gradient"]
                ),
            }
        rows.append(
            {
                "source_id": source_id,
                "source_sha256": source_row["decoded_sha256"],
                "width": int(source.shape[1]),
                "height": int(source.shape[0]),
                "crop_coordinates": coordinates,
                "outputs": output_rows,
            }
        )

    sheet_path = output_dir / "full_resolution_risk_crops.png"
    sheet_sha = _make_sheet(rows, images, arms, contract["audit"]["crop_roles"], crop_size, sheet_path)
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "config_sha256": _sha256(contract_path),
        "parent_report_sha256": contract["parents"]["rf3_d0_report"]["sha256"],
        "source_count": len(rows),
        "arm_count": len(arms),
        "full_resolution_outputs_verified": len(rows) * len(arms),
        "rows": rows,
        "integrity_failures": failures,
        "automatic_integrity_pass": not failures,
        "risk_crop_sheet": "full_resolution_risk_crops.png",
        "risk_crop_sheet_sha256": sheet_sha,
        "visual_review_required": not failures,
        "decision": contract["decision_if_integrity_pass"] if not failures else contract["decision_if_integrity_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    science = dict(report)
    science.pop("risk_crop_sheet")
    report["scientific_identity"] = _canonical_sha256(science)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    if report_path.exists():
        raise ThreeStockFullResolutionAuditError(f"refusing to overwrite output: {report_path}")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.contract, args.root.resolve(), args.output_dir)
    print(json.dumps({"decision": report["decision"], "scientific_identity": report["scientific_identity"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
