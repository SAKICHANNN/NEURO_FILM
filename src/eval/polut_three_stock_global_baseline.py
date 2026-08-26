"""Fixed PoLUT three-stock global baseline on the RF3.D0 source population."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from src.color_engine.lab import linear_rgb_to_lab
from src.color_engine.srgb_transfer import (
    encoded_srgb_to_linear,
    linear_srgb_to_encoded,
)
from src.eval.d_lut_published_assets import CubeAsset, parse_cube
from src.eval.spcp_global_logit_affine import gradient_p999_ratio

SCHEMA = "neuro-film.rf3-d11-polut-three-stock-global-baseline-contract.v1"
REPORT_SCHEMA = "neuro-film.rf3-d11-polut-three-stock-global-baseline-report.v1"


class PoLUTBaselineError(RuntimeError):
    """Raised when a frozen RF3.D11 input or execution invariant fails."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PoLUTBaselineError(f"invalid JSON: {path}") from exc


def _bound_json(root: Path, binding: Mapping[str, Any]) -> Any:
    path = root / str(binding["path"])
    if not path.is_file() or sha256_file(path) != binding["sha256"]:
        raise PoLUTBaselineError(f"bound artifact drift: {binding['path']}")
    return _read_json(path)


def load_contract(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise PoLUTBaselineError("unexpected RF3.D11 contract schema")
    baseline = payload.get("external_baseline", {})
    assets = baseline.get("assets")
    if not isinstance(assets, list) or len(assets) != 3:
        raise PoLUTBaselineError("RF3.D11 requires exactly three external LUTs")
    if baseline.get("full_repository_clone_allowed") is not False:
        raise PoLUTBaselineError("full repository clone must remain forbidden")
    if {row.get("stock_id") for row in assets} != {
        "fujifilm_velvia_50",
        "kodak_portra_400",
        "kodak_ektar_100",
    }:
        raise PoLUTBaselineError("RF3.D11 stock set drifted")
    colour = payload.get("colour_execution", {})
    if colour.get("exposure_adjustment_allowed") is not False:
        raise PoLUTBaselineError("exposure rescue must remain disabled")
    if colour.get("tone_or_gamut_rescue_allowed") is not False:
        raise PoLUTBaselineError("tone/gamut rescue must remain disabled")
    return payload


def _matrix(config: Mapping[str, Any], name: str) -> np.ndarray:
    value = np.asarray(config["colour_execution"][name], dtype=np.float64)
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise PoLUTBaselineError(f"invalid colour matrix: {name}")
    return value


def _adobe_encode(linear: np.ndarray, gamma: float) -> np.ndarray:
    if not np.all(np.isfinite(linear)):
        raise PoLUTBaselineError("non-finite Adobe RGB input")
    if np.any(linear < -1.0e-12) or np.any(linear > 1.0 + 1.0e-12):
        raise PoLUTBaselineError("Adobe RGB input escaped the cube domain")
    return np.power(np.clip(linear, 0.0, 1.0), 1.0 / gamma)


def trilinear_apply(asset: CubeAsset, encoded: np.ndarray) -> np.ndarray:
    """Apply a strict cube using float64 trilinear interpolation."""

    values = np.asarray(encoded, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 3 or not np.all(np.isfinite(values)):
        raise PoLUTBaselineError("cube input must be finite HxWx3")
    if np.any(values < asset.domain_min - 1.0e-12) or np.any(
        values > asset.domain_max + 1.0e-12
    ):
        raise PoLUTBaselineError("cube input escaped the declared domain")
    size = asset.values.shape[0]
    position = (np.clip(values, asset.domain_min, asset.domain_max) - asset.domain_min)
    position *= (size - 1) / (asset.domain_max - asset.domain_min)
    lower = np.minimum(np.floor(position).astype(np.intp), size - 2)
    fraction = position - lower
    red, green, blue = (lower[..., index] for index in range(3))
    fr, fg, fb = (fraction[..., index : index + 1] for index in range(3))
    cube = asset.values
    c000 = cube[red, green, blue]
    c100 = cube[red + 1, green, blue]
    c010 = cube[red, green + 1, blue]
    c001 = cube[red, green, blue + 1]
    c110 = cube[red + 1, green + 1, blue]
    c101 = cube[red + 1, green, blue + 1]
    c011 = cube[red, green + 1, blue + 1]
    c111 = cube[red + 1, green + 1, blue + 1]
    return (
        c000 * (1.0 - fr) * (1.0 - fg) * (1.0 - fb)
        + c100 * fr * (1.0 - fg) * (1.0 - fb)
        + c010 * (1.0 - fr) * fg * (1.0 - fb)
        + c001 * (1.0 - fr) * (1.0 - fg) * fb
        + c110 * fr * fg * (1.0 - fb)
        + c101 * fr * (1.0 - fg) * fb
        + c011 * (1.0 - fr) * fg * fb
        + c111 * fr * fg * fb
    )


def apply_polut(encoded_srgb: np.ndarray, asset: CubeAsset, contract: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, float]]:
    """Apply one frozen Adobe RGB PoLUT and return encoded sRGB plus gamut facts."""

    source = np.asarray(encoded_srgb, dtype=np.float64)
    linear_srgb = encoded_srgb_to_linear(source)
    srgb_to_xyz = _matrix(contract, "linear_srgb_to_xyz_d65")
    xyz_to_adobe = _matrix(contract, "xyz_d65_to_linear_adobe_rgb")
    adobe_to_xyz = _matrix(contract, "linear_adobe_rgb_to_xyz_d65")
    xyz_to_srgb = _matrix(contract, "xyz_d65_to_linear_srgb")
    linear_adobe = linear_srgb @ srgb_to_xyz.T @ xyz_to_adobe.T
    gamma = float(contract["colour_execution"]["adobe_rgb_1998_gamma"])
    cube_input = _adobe_encode(linear_adobe, gamma)
    cube_output = trilinear_apply(asset, cube_input)
    if np.any(cube_output < -1.0e-12):
        raise PoLUTBaselineError("cube emitted negative Adobe RGB code")
    linear_adobe_output = np.power(np.maximum(cube_output, 0.0), gamma)
    linear_srgb_output = linear_adobe_output @ adobe_to_xyz.T @ xyz_to_srgb.T
    out_of_gamut = (linear_srgb_output < -1.0e-12) | (
        linear_srgb_output > 1.0 + 1.0e-12
    )
    clipped = np.clip(linear_srgb_output, 0.0, 1.0)
    encoded_output = linear_srgb_to_encoded(clipped)
    return encoded_output, {
        "minimum_unclipped_linear_srgb": float(np.min(linear_srgb_output)),
        "maximum_unclipped_linear_srgb": float(np.max(linear_srgb_output)),
        "unclipped_out_of_gamut_fraction": float(np.mean(out_of_gamut)),
    }


def _decode_rgb8(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise PoLUTBaselineError(f"cannot decode source: {path}") from exc
    if array.ndim != 3 or array.shape[-1] != 3:
        raise PoLUTBaselineError(f"invalid source geometry: {path}")
    return array


def _publish_png(path: Path, array: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise PoLUTBaselineError(f"refusing to replace output: {path}")
    Image.fromarray(array, mode="RGB").save(path, format="PNG", compress_level=6)
    return sha256_file(path)


def _delta_e76(encoded_a: np.ndarray, encoded_b: np.ndarray) -> np.ndarray:
    lab_a = linear_rgb_to_lab(
        np.asarray(encoded_srgb_to_linear(encoded_a), dtype=np.float32),
        working_space="linear_srgb",
    )
    lab_b = linear_rgb_to_lab(
        np.asarray(encoded_srgb_to_linear(encoded_b), dtype=np.float32),
        working_space="linear_srgb",
    )
    return np.linalg.norm(lab_a.astype(np.float64) - lab_b.astype(np.float64), axis=-1)


def _boundary_fractions(source: np.ndarray, output: np.ndarray) -> tuple[float, float]:
    source_boundary = (source == 0) | (source == 255)
    output_boundary = (output == 0) | (output == 255)
    return float(np.mean(output_boundary)), float(np.mean(output_boundary & ~source_boundary))


def _contact_sheet(
    rows: list[dict[str, Any]],
    images: Mapping[tuple[str, str], np.ndarray],
    stock_ids: list[str],
    path: Path,
) -> str:
    width = 240
    label = 22
    columns = ["source", *stock_ids]
    thumbs: dict[tuple[str, str], Image.Image] = {}
    heights: list[int] = []
    for row in rows:
        source_id = row["source_id"]
        row_height = 0
        for column in columns:
            image = Image.fromarray(images[(source_id, column)], mode="RGB")
            height = max(1, round(image.height * width / image.width))
            thumbs[(source_id, column)] = image.resize((width, height), Image.Resampling.LANCZOS)
            row_height = max(row_height, height)
        heights.append(row_height + label)
    canvas = Image.new("RGB", (width * len(columns), sum(heights)), "white")
    draw = ImageDraw.Draw(canvas)
    top = 0
    for row, row_height in zip(rows, heights, strict=True):
        source_id = row["source_id"]
        for index, column in enumerate(columns):
            draw.text((index * width + 3, top + 3), f"{source_id} | {column}", fill="black")
            canvas.paste(thumbs[(source_id, column)], (index * width, top + label))
        top += row_height
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", compress_level=6)
    return sha256_file(path)


def evaluate(contract_path: Path, root: Path, output_dir: Path, *, order: str) -> dict[str, Any]:
    contract = load_contract(contract_path)
    d0_contract = _bound_json(root, contract["parents"]["rf3_d0_contract"])
    d0_evidence = _bound_json(root, contract["parents"]["rf3_d0_evidence"])
    d0_report = _bound_json(root, contract["parents"]["rf3_d0_report"])
    source_manifest = _bound_json(root, contract["parents"]["source_manifest"])
    if d0_evidence.get("status") != contract["parents"]["rf3_d0_evidence"]["required_status"]:
        raise PoLUTBaselineError("RF3.D0 evidence status drift")
    if d0_report.get("scientific_identity") != contract["parents"]["rf3_d0_report"]["required_scientific_identity"]:
        raise PoLUTBaselineError("RF3.D0 scientific identity drift")
    included_ids = list(d0_contract["source"]["included_ids"])
    source_by_id = {row["id"]: row for row in source_manifest}
    if len(included_ids) != int(contract["parents"]["source_manifest"]["expected_rows"]):
        raise PoLUTBaselineError("source population size drift")
    asset_rows = list(contract["external_baseline"]["assets"])
    if order not in {"canonical", "reverse"}:
        raise PoLUTBaselineError("order must be canonical or reverse")
    execution_ids = included_ids if order == "canonical" else list(reversed(included_ids))
    execution_assets = asset_rows if order == "canonical" else list(reversed(asset_rows))
    assets: dict[str, CubeAsset] = {}
    asset_inventory: list[dict[str, Any]] = []
    source_manifest_binding = contract["external_baseline"]["source_manifest"]
    external_manifest_path = root / source_manifest_binding["path"]
    if (
        not external_manifest_path.is_file()
        or external_manifest_path.stat().st_size != int(source_manifest_binding["bytes"])
        or sha256_file(external_manifest_path) != source_manifest_binding["sha256"]
    ):
        raise PoLUTBaselineError("external source manifest drift")
    external_manifest = _read_json(external_manifest_path)
    if (
        external_manifest.get("schema") != "neuro-film.rf3-d11-polut-source-manifest.v1"
        or external_manifest.get("exact_commit")
        != contract["external_baseline"]["exact_commit"]
        or external_manifest.get("repository_license")
        != contract["external_baseline"]["repository_license"]
        or external_manifest.get("full_repository_clone_performed") is not False
    ):
        raise PoLUTBaselineError("external source manifest policy drift")
    manifest_rows = {row["stock_id"]: row for row in external_manifest["assets"]}
    for row in asset_rows:
        path = root / row["local_path"]
        manifest_row = manifest_rows.get(row["stock_id"])
        if manifest_row is None:
            raise PoLUTBaselineError(f"external source manifest mismatch: {row['stock_id']}")
        for manifest_key, contract_key in (
            ("stock_id", "stock_id"),
            ("bytes", "bytes"),
            ("git_blob_sha1", "git_blob_sha1"),
            ("source_path", "relative_path"),
        ):
            if manifest_row.get(manifest_key) != row[contract_key]:
                raise PoLUTBaselineError(f"external source manifest mismatch: {row['stock_id']}")
        if not path.is_file() or path.stat().st_size != int(row["bytes"]):
            raise PoLUTBaselineError(f"external LUT size drift: {row['stock_id']}")
        actual_sha256 = sha256_file(path)
        actual_blob = git_blob_sha1(path)
        if actual_sha256 != manifest_row.get("sha256") or actual_blob != row["git_blob_sha1"]:
            raise PoLUTBaselineError(f"external LUT identity drift: {row['stock_id']}")
        asset = parse_cube(path)
        if asset.values.shape[:3] != (33, 33, 33):
            raise PoLUTBaselineError(f"unexpected cube dimension: {row['stock_id']}")
        assets[row["stock_id"]] = asset
        asset_inventory.append(
            {
                "stock_id": row["stock_id"],
                "bytes": path.stat().st_size,
                "sha256": actual_sha256,
                "git_blob_sha1": actual_blob,
            }
        )
    images: dict[tuple[str, str], np.ndarray] = {}
    row_results: dict[str, dict[str, Any]] = {}
    pair_names = (
        ("velvia_vs_portra", "fujifilm_velvia_50", "kodak_portra_400"),
        ("velvia_vs_ektar", "fujifilm_velvia_50", "kodak_ektar_100"),
        ("portra_vs_ektar", "kodak_portra_400", "kodak_ektar_100"),
    )
    for source_id in execution_ids:
        source_row = source_by_id.get(source_id)
        if source_row is None:
            raise PoLUTBaselineError(f"missing source row: {source_id}")
        for key, expected in (
            ("allowed_use", contract["parents"]["source_manifest"]["required_allowed_use"]),
            ("rights_scope", contract["parents"]["source_manifest"]["required_rights_scope"]),
            ("decoded_color_state", contract["parents"]["source_manifest"]["required_color_state"]),
        ):
            if source_row.get(key) != expected:
                raise PoLUTBaselineError(f"source policy drift: {source_id}/{key}")
        source_path = root / source_row["decoded_path"]
        if sha256_file(source_path) != source_row["decoded_sha256"]:
            raise PoLUTBaselineError(f"source hash drift: {source_id}")
        source_u8 = _decode_rgb8(source_path)
        source = source_u8.astype(np.float64) / 255.0
        images[(source_id, "source")] = source_u8
        outputs: dict[str, dict[str, Any]] = {}
        output_arrays: dict[str, np.ndarray] = {}
        for asset_row in execution_assets:
            stock_id = asset_row["stock_id"]
            encoded, gamut = apply_polut(source, assets[stock_id], contract)
            if not np.all(np.isfinite(encoded)):
                raise PoLUTBaselineError(f"non-finite output: {source_id}/{stock_id}")
            output_u8 = np.rint(np.clip(encoded, 0.0, 1.0) * 255.0).astype(np.uint8)
            output_path = output_dir / "renders" / stock_id / f"{source_id}.png"
            png_sha = _publish_png(output_path, output_u8)
            boundary, new_boundary = _boundary_fractions(source_u8, output_u8)
            outputs[stock_id] = {
                "relative_path": output_path.relative_to(output_dir).as_posix(),
                "png_sha256": png_sha,
                "pixel_sha256": hashlib.sha256(output_u8.tobytes()).hexdigest(),
                "output_boundary_fraction": boundary,
                "new_output_boundary_fraction": new_boundary,
                "gradient_p999_ratio_vs_source": gradient_p999_ratio(source, output_u8 / 255.0),
                **gamut,
            }
            output_arrays[stock_id] = output_u8
            images[(source_id, stock_id)] = output_u8
        comparisons: dict[str, Any] = {}
        for name, first, second in pair_names:
            delta = _delta_e76(output_arrays[first] / 255.0, output_arrays[second] / 255.0)
            comparisons[name] = {
                "median_delta_e76": float(np.median(delta)),
                "p95_delta_e76": float(np.quantile(delta, 0.95)),
            }
        row_results[source_id] = {
            "source_id": source_id,
            "source_sha256": source_row["decoded_sha256"],
            "width": int(source_u8.shape[1]),
            "height": int(source_u8.shape[0]),
            "outputs": outputs,
            "comparisons": comparisons,
        }
    rows = [row_results[source_id] for source_id in included_ids]
    population: dict[str, float] = {}
    gain_ratios: dict[str, float] = {}
    legacy = contract["comparison"]["legacy_pairwise_population_median_delta_e76"]
    for name, _, _ in pair_names:
        population[name] = float(np.median([row["comparisons"][name]["median_delta_e76"] for row in rows]))
        gain_ratios[name] = population[name] / float(legacy[name])
    all_outputs = [value for row in rows for value in row["outputs"].values()]
    failures: list[str] = []
    gates = contract["comparison"]
    for name, value in population.items():
        if value < float(gates["minimum_each_pairwise_population_median_delta_e76"]):
            failures.append(f"pairwise_separation:{name}")
        if gain_ratios[name] < float(gates["minimum_each_pairwise_gain_ratio_vs_matching_rf3_d0_pair"]):
            failures.append(f"pairwise_gain_ratio:{name}")
    if max(value["new_output_boundary_fraction"] for value in all_outputs) > float(gates["maximum_new_output_boundary_fraction"]):
        failures.append("new_output_boundary")
    if max(value["output_boundary_fraction"] for value in all_outputs) > float(gates["maximum_output_boundary_fraction"]):
        failures.append("output_boundary")
    if max(value["gradient_p999_ratio_vs_source"] for value in all_outputs) > float(gates["maximum_gradient_p999_ratio_vs_source"]):
        failures.append("gradient_tail")
    stock_ids = [row["stock_id"] for row in asset_rows]
    sheet_path = output_dir / "contact_sheet.png"
    sheet_sha = _contact_sheet(rows, images, stock_ids, sheet_path)
    automatic_pass = not failures
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "config_sha256": sha256_file(contract_path),
        "order": order,
        "asset_inventory": asset_inventory,
        "source_count": len(rows),
        "rows": rows,
        "aggregate": {
            "pairwise_population_median_delta_e76": population,
            "pairwise_gain_ratio_vs_rf3_d0": gain_ratios,
            "maximum_new_output_boundary_fraction": max(value["new_output_boundary_fraction"] for value in all_outputs),
            "maximum_output_boundary_fraction": max(value["output_boundary_fraction"] for value in all_outputs),
            "maximum_gradient_p999_ratio_vs_source": max(value["gradient_p999_ratio_vs_source"] for value in all_outputs),
            "maximum_unclipped_out_of_gamut_fraction": max(value["unclipped_out_of_gamut_fraction"] for value in all_outputs),
            "failed_gates": failures,
            "automatic_pass": automatic_pass,
        },
        "contact_sheet": "contact_sheet.png",
        "contact_sheet_sha256": sheet_sha,
        "visual_review_required": bool(automatic_pass and gates["risk_crop_visual_review_required_after_automatic_pass"]),
        "decision": contract["decision_if_pass"] if automatic_pass else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    scientific = dict(report)
    scientific.pop("order")
    scientific.pop("contact_sheet")
    report["scientific_identity"] = _canonical_sha256(scientific)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


__all__ = [
    "PoLUTBaselineError",
    "apply_polut",
    "evaluate",
    "git_blob_sha1",
    "load_contract",
    "sha256_file",
    "trilinear_apply",
]
