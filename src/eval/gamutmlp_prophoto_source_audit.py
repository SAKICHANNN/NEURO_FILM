"""U1.4C6S byte-bound official GamutMLP NUS-camera source audit."""

from __future__ import annotations

import hashlib
import json
import re
import struct
import zipfile
from collections import Counter, defaultdict
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from scripts.build_fivek_freeze_pack import (
    D50_TO_D65_BRADFORD,
    PROPHOTO_TO_XYZ_D50,
    XYZ_D65_TO_SRGB,
    prophoto_decode,
)
from src.eval.fujifilm_dye_basis_measured_conformance import canonical_json, hash_file

SCHEMA = "neuro-film.u1-4c6-gamutmlp-nus-prophoto-source-contract.v1"
REPORT_SCHEMA = "neuro-film.u1-4c6-gamutmlp-nus-prophoto-source-report.v1"
EXPERIMENT_ID = "U1.4C6S"
CONTRACT_SHA256 = "a3951e74cdf71b35ff4dc12692afeaaed4bb148eae1d66fb33f2a2106a1f6925"


class GamutMLPSourceError(RuntimeError):
    """Raised when the archive, deterministic selection or pixels drift."""


def _relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise GamutMLPSourceError("C6S paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    if hash_file(path) != CONTRACT_SHA256:
        raise GamutMLPSourceError("C6S contract hash drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise GamutMLPSourceError("C6S contract structure drift")
    _relative(payload["archive"]["logical_path"])
    _relative(payload["output"]["logical_root"])
    _relative(payload["output"]["manifest"])
    _relative(payload["output"]["contact_sheet"])
    if (
        len(payload["selection"]["camera_models"]) != 8
        or len(set(payload["selection"]["camera_models"])) != 8
        or int(payload["selection"]["expected_rows"])
        != len(payload["selection"]["camera_models"])
        * int(payload["selection"]["rows_per_camera"])
    ):
        raise GamutMLPSourceError("C6S selection contract drift")
    return payload


def _member_is_safe(name: str, expected_root: str) -> bool:
    path = PurePosixPath(name)
    return bool(
        name
        and "\\" not in name
        and not path.is_absolute()
        and ".." not in path.parts
        and path.parts
        and path.parts[0] == expected_root
    )


def _png_header(raw: bytes) -> tuple[int, int, int, int]:
    if len(raw) < 29 or raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[12:16] != b"IHDR":
        raise GamutMLPSourceError("C6S selected member is not PNG")
    width, height, depth, colour_type, compression, filtering, interlace = (
        struct.unpack(">IIBBBBB", raw[16:29])
    )
    if compression != 0 or filtering != 0 or interlace != 0:
        raise GamutMLPSourceError("C6S selected PNG encoding drift")
    return width, height, depth, colour_type


def _selected_rows(
    infos: list[zipfile.ZipInfo], contract: Mapping[str, Any]
) -> list[dict[str, Any]]:
    selection = contract["selection"]
    pattern = re.compile(selection["member_regex"])
    by_camera: dict[str, list[tuple[str, zipfile.ZipInfo, str, str]]] = defaultdict(list)
    cameras = set(selection["camera_models"])
    for info in infos:
        match = pattern.fullmatch(PurePosixPath(info.filename).name)
        if match is None or match.group(2) not in cameras:
            continue
        selector = hashlib.sha256(info.filename.encode("utf-8")).hexdigest()
        by_camera[match.group(2)].append(
            (selector, info, match.group(1), match.group(3))
        )
    used_source_ids: set[str] = set()
    rows: list[dict[str, Any]] = []
    for camera in sorted(cameras):
        retained = 0
        for selector, info, style, source_id in sorted(by_camera[camera]):
            if source_id in used_source_ids:
                continue
            used_source_ids.add(source_id)
            rows.append(
                {
                    "camera": camera,
                    "source_id": source_id,
                    "style": style,
                    "member": info.filename,
                    "member_bytes": info.file_size,
                    "selector_sha256": selector,
                }
            )
            retained += 1
            if retained == int(selection["rows_per_camera"]):
                break
        if retained != int(selection["rows_per_camera"]):
            raise GamutMLPSourceError(f"C6S camera selection underflow: {camera}")
    if (
        len(rows) != int(selection["expected_rows"])
        or len(used_source_ids) != len(rows)
        or hashlib.sha256(canonical_json(rows).encode("utf-8")).hexdigest()
        != selection["selected_rows_canonical_sha256"]
        or hashlib.sha256(
            json.dumps(
                [row["member"] for row in rows], separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        != selection["selected_members_canonical_sha256"]
        or sum(int(row["member_bytes"]) for row in rows)
        != int(selection["selected_member_bytes"])
    ):
        raise GamutMLPSourceError("C6S selected row identity drift")
    return rows


def _declared_prophoto_to_srgb(encoded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    linear = prophoto_decode(encoded)
    xyz_d50 = linear @ PROPHOTO_TO_XYZ_D50.T
    xyz_d65 = xyz_d50 @ D50_TO_D65_BRADFORD.T
    linear_srgb = xyz_d65 @ XYZ_D65_TO_SRGB.T
    encoded_srgb = np.where(
        linear_srgb <= 0.0031308,
        12.92 * linear_srgb,
        1.055 * np.maximum(linear_srgb, 0.0) ** (1.0 / 2.4) - 0.055,
    )
    return np.asarray(linear_srgb, dtype=np.float32), np.asarray(
        encoded_srgb, dtype=np.float32
    )


def _write_exact(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_bytes() == payload:
            return
        raise GamutMLPSourceError(f"C6S existing output identity drift: {path}")
    temporary = path.with_name(f".{path.name}.part")
    if temporary.exists():
        raise GamutMLPSourceError(f"C6S temporary output already exists: {temporary}")
    temporary.write_bytes(payload)
    temporary.replace(path)


def evaluate(
    contract: Mapping[str, Any], root: Path, output_dir: Path
) -> dict[str, Any]:
    if output_dir.exists():
        raise GamutMLPSourceError("C6S output directory must be create-only")
    output_dir.mkdir(parents=True)
    archive_path = root / _relative(contract["archive"]["logical_path"])
    archive = contract["archive"]
    if (
        not archive_path.is_file()
        or archive_path.stat().st_size != int(archive["bytes"])
        or hash_file(archive_path) != archive["sha256"]
    ):
        raise GamutMLPSourceError("C6S archive identity drift")
    with zipfile.ZipFile(archive_path) as document:
        infos = document.infolist()
        files = [info for info in infos if not info.is_dir()]
        names = [info.filename for info in infos]
        if (
            len(infos) != int(archive["expected_entries"])
            or len(files) != int(archive["expected_png_members"])
            or len(names) != len(set(names))
            or sum(info.file_size for info in files)
            != int(archive["expected_uncompressed_bytes"])
            or any(info.flag_bits & 1 for info in files)
            or any(
                not _member_is_safe(info.filename, archive["expected_root"])
                for info in infos
            )
            or any(PurePosixPath(info.filename).suffix.lower() != ".png" for info in files)
        ):
            raise GamutMLPSourceError("C6S archive structure drift")
        style_counts = Counter(
            PurePosixPath(info.filename).name.split("_", 1)[0] for info in files
        )
        if dict(style_counts) != archive["expected_style_counts"]:
            raise GamutMLPSourceError("C6S archive style inventory drift")
        selected = _selected_rows(files, contract)
        info_by_name = {info.filename: info for info in files}
        extracted_root = root / _relative(contract["output"]["logical_root"])
        facts: list[dict[str, Any]] = []
        previews: list[tuple[str, Image.Image]] = []
        pixel_hashes: set[str] = set()
        minimum_oog = float(
            contract["eligibility"][
                "minimum_declared_prophoto_to_srgb_out_of_gamut_fraction_per_row"
            ]
        )
        for row in selected:
            raw = document.read(info_by_name[row["member"]])
            width, height, depth, colour_type = _png_header(raw)
            data = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if (
                len(raw) != int(row["member_bytes"])
                or width != int(archive["expected_width"])
                or height != int(archive["expected_height"])
                or depth != int(archive["expected_bit_depth"])
                or colour_type != int(archive["expected_color_type"])
                or data is None
                or data.dtype != np.uint16
                or data.shape != (height, width, 3)
            ):
                raise GamutMLPSourceError("C6S selected PNG structure drift")
            rgb = np.ascontiguousarray(data[..., ::-1])
            pixel_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
            if pixel_sha in pixel_hashes:
                raise GamutMLPSourceError("C6S selected exact pixel duplicate")
            pixel_hashes.add(pixel_sha)
            linear_srgb, encoded_srgb = _declared_prophoto_to_srgb(
                rgb.astype(np.float32) / np.float32(65535.0)
            )
            oog_fraction = float(
                np.mean(np.any((linear_srgb < 0.0) | (linear_srgb > 1.0), axis=-1))
            )
            if (
                not np.isfinite(linear_srgb).all()
                or not np.isfinite(encoded_srgb).all()
                or oog_fraction < minimum_oog
            ):
                raise GamutMLPSourceError("C6S declared ProPhoto pixel gate failed")
            relative = Path(row["camera"]) / PurePosixPath(row["member"]).name
            _write_exact(extracted_root / relative, raw)
            preview_array = np.asarray(
                np.rint(np.clip(encoded_srgb, 0.0, 1.0) * 255.0), dtype=np.uint8
            )
            preview = Image.fromarray(preview_array, mode="RGB")
            preview.thumbnail((256, 256), Image.Resampling.LANCZOS)
            previews.append(
                (f"{row['camera']} {row['source_id']} {row['style']}", preview.copy())
            )
            facts.append(
                {
                    **row,
                    "path": (contract["output"]["logical_root"] + "/" + relative.as_posix()),
                    "member_sha256": hashlib.sha256(raw).hexdigest(),
                    "pixel_sha256": pixel_sha,
                    "width": width,
                    "height": height,
                    "dtype": "uint16",
                    "declared_prophoto_to_srgb_out_of_gamut_fraction": oog_fraction,
                }
            )
    sheet = Image.new("RGB", (1280, 1800), "#202020")
    draw = ImageDraw.Draw(sheet)
    for index, (label, preview) in enumerate(previews):
        x = (index % 4) * 320
        y = (index // 4) * 300
        sheet.paste(preview, (x + (256 - preview.width) // 2, y))
        draw.text((x + 4, y + 260), label, fill="white")
    contact_path = root / _relative(contract["output"]["contact_sheet"])
    contact_bytes = BytesIO()
    sheet.save(contact_bytes, format="PNG", optimize=False)
    _write_exact(contact_path, contact_bytes.getvalue())
    manifest = {
        "schema": "neuro-film.u1-4c6-gamutmlp-nus-prophoto-source-manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "archive_sha256": archive["sha256"],
        "selection_contract_sha256": CONTRACT_SHA256,
        "declared_colour_space": "ProPhoto RGB / ROMM, paper and dataset declaration only",
        "embedded_icc_attested": False,
        "allowed_use": contract["rights"]["allowed_use"],
        "rights_scope": contract["rights"]["scope"],
        "rows": facts,
    }
    manifest_sha = hashlib.sha256(canonical_json(manifest).encode("utf-8")).hexdigest()
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "contract_sha256": CONTRACT_SHA256,
        "archive_sha256": archive["sha256"],
        "selected_manifest": manifest,
        "selected_manifest_canonical_sha256": manifest_sha,
        "contact_sheet_sha256": hash_file(contact_path),
        "metrics": {
            "archive_png_members": int(archive["expected_png_members"]),
            "selected_rows": len(facts),
            "selected_camera_models": len({row["camera"] for row in facts}),
            "selected_unique_source_ids": len({row["source_id"] for row in facts}),
            "minimum_declared_prophoto_to_srgb_out_of_gamut_fraction": min(
                row["declared_prophoto_to_srgb_out_of_gamut_fraction"] for row in facts
            ),
            "maximum_declared_prophoto_to_srgb_out_of_gamut_fraction": max(
                row["declared_prophoto_to_srgb_out_of_gamut_fraction"] for row in facts
            ),
        },
        "automatic_pass": True,
        "visual_review_required": bool(
            contract["eligibility"]["visual_review_required_before_algorithm_role"]
        ),
        "algorithm_execution_count": 0,
        "production_default_changed": False,
        "claim_ceiling": contract["claim_ceiling"],
    }
    report["stable_evidence_id"] = hashlib.sha256(
        canonical_json(report).encode("utf-8")
    ).hexdigest()
    report_bytes = (canonical_json(report) + "\n").encode("utf-8")
    (output_dir / "report.json").write_bytes(report_bytes)
    return report


__all__ = ["GamutMLPSourceError", "evaluate", "load_contract"]
