"""U1.4C7S source-only Rec.2020-stress selection from untouched GamutMLP rows."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections import defaultdict
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from src.eval.fujifilm_dye_basis_measured_conformance import canonical_json, hash_file
from src.eval.gamutmlp_prophoto_source_audit import (
    _declared_prophoto_to_srgb,
    _member_is_safe,
    _png_header,
    _relative,
    _write_exact,
)
from src.preprocess.color_management import convert_linear_rgb

SCHEMA = "neuro-film.u1-4c7-gamutmlp-rec2020-stress-source-contract.v1"
REPORT_SCHEMA = "neuro-film.u1-4c7-gamutmlp-rec2020-stress-source-report.v1"
EXPERIMENT_ID = "U1.4C7S"
CONTRACT_SHA256 = "31c9ce23c1275a1ca9b180834472670d48aa96e19d053fde9dc911baeb39d019"


class GamutMLPStressSourceError(RuntimeError):
    """Raised when a C7 source identity, selection, or pixel gate drifts."""


def load_contract(path: Path) -> dict[str, Any]:
    if hash_file(path) != CONTRACT_SHA256:
        raise GamutMLPStressSourceError("C7S contract hash drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise GamutMLPStressSourceError("C7S contract structure drift")
    _relative(payload["archive"]["logical_path"])
    _relative(payload["output"]["logical_root"])
    _relative(payload["output"]["reviewed_manifest"])
    _relative(payload["output"]["contact_sheet"])
    for parent in payload["parents"].values():
        _relative(parent["path"])
    selection = payload["selection"]
    if (
        len(selection["camera_models"]) != 8
        or len(set(selection["camera_models"])) != 8
        or int(selection["expected_rows"])
        != len(selection["camera_models"]) * int(selection["rows_per_camera"])
        or selection.get("require_global_unique_source_id") is not True
    ):
        raise GamutMLPStressSourceError("C7S selection contract drift")
    return payload


def _validate_parents(contract: Mapping[str, Any], root: Path) -> set[tuple[str, str]]:
    for parent in contract["parents"].values():
        path = root / _relative(parent["path"])
        if not path.is_file() or hash_file(path) != parent["sha256"]:
            raise GamutMLPStressSourceError("C7S parent identity drift")
    result_parent = contract["parents"]["c6_result"]
    result = json.loads(
        (root / _relative(result_parent["path"])).read_text(encoding="utf-8")
    )
    if result.get("decision") != result_parent["required_decision"]:
        raise GamutMLPStressSourceError("C7S parent decision drift")
    manifest = json.loads(
        (
            root / _relative(contract["parents"]["c6_reviewed_manifest"]["path"])
        ).read_text(encoding="utf-8")
    )
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != 24:
        raise GamutMLPStressSourceError("C7S exclusion manifest drift")
    return {(str(row["camera"]), str(row["source_id"])) for row in rows}


def _read_member(
    document: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    archive: Mapping[str, Any],
) -> tuple[bytes, np.ndarray, np.ndarray, float, float]:
    raw = document.read(info)
    width, height, depth, colour_type = _png_header(raw)
    data = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if (
        width != int(archive["expected_width"])
        or height != int(archive["expected_height"])
        or depth != int(archive["expected_bit_depth"])
        or colour_type != int(archive["expected_color_type"])
        or data is None
        or data.dtype != np.uint16
        or data.shape != (height, width, 3)
    ):
        raise GamutMLPStressSourceError("C7S PNG structure drift")
    rgb = np.ascontiguousarray(data[..., ::-1])
    linear_srgb, encoded_srgb = _declared_prophoto_to_srgb(
        rgb.astype(np.float32) / np.float32(65535.0)
    )
    rec2020 = convert_linear_rgb(
        linear_srgb, source_space="linear_srgb", destination_space="linear_rec2020"
    )
    if not np.isfinite(linear_srgb).all() or not np.isfinite(rec2020).all():
        raise GamutMLPStressSourceError("C7S non-finite declared colour conversion")
    srgb_oog = float(
        np.mean(np.any((linear_srgb < 0.0) | (linear_srgb > 1.0), axis=-1))
    )
    rec2020_oog = float(np.mean(np.any((rec2020 < 0.0) | (rec2020 > 1.0), axis=-1)))
    return raw, rgb, encoded_srgb, srgb_oog, rec2020_oog


def _select(
    document: zipfile.ZipFile,
    infos: list[zipfile.ZipInfo],
    contract: Mapping[str, Any],
    excluded: set[tuple[str, str]],
) -> tuple[list[dict[str, Any]], list[tuple[str, Image.Image]]]:
    selection = contract["selection"]
    eligibility = contract["eligibility"]
    pattern = re.compile(selection["member_regex"])
    cameras = set(selection["camera_models"])
    candidates: dict[str, list[tuple[str, zipfile.ZipInfo, str, str]]] = defaultdict(
        list
    )
    for info in infos:
        match = pattern.fullmatch(PurePosixPath(info.filename).name)
        if (
            match is None
            or match.group(2) not in cameras
            or (match.group(2), match.group(3)) in excluded
        ):
            continue
        candidates[match.group(2)].append(
            (
                hashlib.sha256(info.filename.encode("utf-8")).hexdigest(),
                info,
                match.group(1),
                match.group(3),
            )
        )
    rows: list[dict[str, Any]] = []
    previews: list[tuple[str, Image.Image]] = []
    used_source_ids: set[str] = set()
    pixel_hashes: set[str] = set()
    for camera in sorted(cameras):
        tested = 0
        retained = 0
        for selector, info, style, source_id in sorted(candidates[camera]):
            if source_id in used_source_ids:
                continue
            tested += 1
            raw, rgb, encoded_srgb, srgb_oog, rec2020_oog = _read_member(
                document, info, contract["archive"]
            )
            if srgb_oog < float(
                eligibility[
                    "minimum_declared_prophoto_to_srgb_out_of_gamut_fraction_per_row"
                ]
            ) or rec2020_oog < float(
                eligibility[
                    "minimum_precompression_rec2020_out_of_gamut_fraction_per_row"
                ]
            ):
                continue
            pixel_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
            if pixel_sha in pixel_hashes:
                raise GamutMLPStressSourceError("C7S exact pixel duplicate")
            pixel_hashes.add(pixel_sha)
            used_source_ids.add(source_id)
            relative = Path(camera) / PurePosixPath(info.filename).name
            _write_exact(
                Path(contract["_root"])
                / _relative(contract["output"]["logical_root"])
                / relative,
                raw,
            )
            preview_array = np.asarray(
                np.rint(np.clip(encoded_srgb, 0.0, 1.0) * 255.0), dtype=np.uint8
            )
            preview = Image.fromarray(preview_array, mode="RGB")
            preview.thumbnail((256, 256), Image.Resampling.LANCZOS)
            previews.append((f"{camera} {source_id} {style}", preview.copy()))
            rows.append(
                {
                    "camera": camera,
                    "source_id": source_id,
                    "style": style,
                    "member": info.filename,
                    "path": contract["output"]["logical_root"]
                    + "/"
                    + relative.as_posix(),
                    "selector_sha256": selector,
                    "tested_rank": tested,
                    "member_bytes": info.file_size,
                    "member_sha256": hashlib.sha256(raw).hexdigest(),
                    "pixel_sha256": pixel_sha,
                    "width": int(contract["archive"]["expected_width"]),
                    "height": int(contract["archive"]["expected_height"]),
                    "dtype": "uint16",
                    "declared_prophoto_to_srgb_out_of_gamut_fraction": srgb_oog,
                    "precompression_rec2020_out_of_gamut_fraction": rec2020_oog,
                }
            )
            retained += 1
            if retained == int(selection["rows_per_camera"]):
                break
        if retained != int(selection["rows_per_camera"]):
            raise GamutMLPStressSourceError(f"C7S selection underflow: {camera}")
    if (
        len(rows) != int(selection["expected_rows"])
        or len(used_source_ids) != len(rows)
        or any((row["camera"], row["source_id"]) in excluded for row in rows)
    ):
        raise GamutMLPStressSourceError("C7S selected identity drift")
    return rows, previews


def evaluate(
    contract: Mapping[str, Any], root: Path, output_dir: Path
) -> dict[str, Any]:
    if output_dir.exists():
        raise GamutMLPStressSourceError("C7S output directory must be create-only")
    output_dir.mkdir(parents=True)
    excluded = _validate_parents(contract, root)
    archive = contract["archive"]
    archive_path = root / _relative(archive["logical_path"])
    if (
        not archive_path.is_file()
        or archive_path.stat().st_size != int(archive["bytes"])
        or hash_file(archive_path) != archive["sha256"]
    ):
        raise GamutMLPStressSourceError("C7S archive identity drift")
    mutable = dict(contract)
    mutable["_root"] = str(root)
    with zipfile.ZipFile(archive_path) as document:
        infos = document.infolist()
        files = [info for info in infos if not info.is_dir()]
        if (
            len(infos) != int(archive["expected_entries"])
            or len(files) != int(archive["expected_png_members"])
            or len({info.filename for info in infos}) != len(infos)
            or sum(info.file_size for info in files)
            != int(archive["expected_uncompressed_bytes"])
            or any(info.flag_bits & 1 for info in files)
            or any(
                not _member_is_safe(info.filename, archive["expected_root"])
                for info in infos
            )
        ):
            raise GamutMLPStressSourceError("C7S archive structure drift")
        rows, previews = _select(document, files, mutable, excluded)
    sheet = Image.new("RGB", (1280, 1800), "#202020")
    draw = ImageDraw.Draw(sheet)
    for index, (label, preview) in enumerate(previews):
        x = (index % 4) * 320
        y = (index // 4) * 300
        sheet.paste(preview, (x + (256 - preview.width) // 2, y))
        draw.text((x + 4, y + 260), label, fill="white")
    contact_path = root / _relative(contract["output"]["contact_sheet"])
    buffer = BytesIO()
    sheet.save(buffer, format="PNG", optimize=False)
    _write_exact(contact_path, buffer.getvalue())
    manifest = {
        "schema": "neuro-film.u1-4c7-gamutmlp-rec2020-stress-source-manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "archive_sha256": archive["sha256"],
        "selection_contract_sha256": CONTRACT_SHA256,
        "declared_colour_space": "ProPhoto RGB / ROMM, paper and dataset declaration only",
        "embedded_icc_attested": False,
        "allowed_use": contract["rights"]["allowed_use"],
        "rights_scope": contract["rights"]["scope"],
        "rows": rows,
    }
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "contract_sha256": CONTRACT_SHA256,
        "archive_sha256": archive["sha256"],
        "excluded_c6_rows": len(excluded),
        "selected_manifest": manifest,
        "selected_manifest_canonical_sha256": hashlib.sha256(
            canonical_json(manifest)
        ).hexdigest(),
        "contact_sheet_sha256": hash_file(contact_path),
        "metrics": {
            "selected_rows": len(rows),
            "selected_camera_models": len({row["camera"] for row in rows}),
            "selected_unique_source_ids": len({row["source_id"] for row in rows}),
            "maximum_tested_rank": max(int(row["tested_rank"]) for row in rows),
            "minimum_declared_prophoto_to_srgb_out_of_gamut_fraction": min(
                row["declared_prophoto_to_srgb_out_of_gamut_fraction"] for row in rows
            ),
            "minimum_precompression_rec2020_out_of_gamut_fraction": min(
                row["precompression_rec2020_out_of_gamut_fraction"] for row in rows
            ),
        },
        "automatic_pass": True,
        "visual_review_required": True,
        "algorithm_execution_count": 0,
        "production_default_changed": False,
        "claim_ceiling": contract["claim_ceiling"],
    }
    report["stable_evidence_id"] = hashlib.sha256(canonical_json(report)).hexdigest()
    (output_dir / "report.json").write_bytes(canonical_json(report) + b"\n")
    return report


__all__ = ["GamutMLPStressSourceError", "evaluate", "load_contract"]
