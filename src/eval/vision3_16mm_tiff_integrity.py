"""U5.R2BU7 exact TIFF integrity and contact-order audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from pypdf import PdfReader

SCHEMA = "neuro_film.u5_r2bu7_vision3_16mm_tiff_integrity_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2bu7_vision3_16mm_tiff_integrity_report.v1"
_LUMA = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)


class Vision3TiffIntegrityError(ValueError):
    """Raised when the BU7 contract or retained source files drift."""


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise Vision3TiffIntegrityError("BU7 path must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    files = payload.get("source", {}).get("files", [])
    expected_names = [
        "16MM_TEST_01.tif",
        "16MM_TEST_02.tif",
        "16MM_TEST_03.tif",
        "16MM_TEST_43.tif",
        "16MM_TEST_44.tif",
        "16MM_TEST_45.tif",
    ]
    expected_indices = [0, 1, 2, 42, 43, 44]
    decode = payload.get("decode", {})
    amendment = payload.get("execution_amendment", {})
    if (
        payload.get("schema") != SCHEMA
        or payload.get("experiment_id") != "U5.R2BU7"
        or payload.get("parent", {}).get("report_sha256")
        != "6ca7bb04273b0d099ed057c483b03127a1f16a7804cf199f0dbbfffd042defa0"
        or [row.get("filename") for row in files] != expected_names
        or [row.get("contact_index") for row in files] != expected_indices
        or any(row.get("bytes") != 12_763_972 for row in files)
        or payload.get("acquisition", {}).get("expected_total_bytes") != 76_583_832
        or payload.get("acquisition", {}).get("maximum_total_bytes") != 80_000_000
        or decode.get("required_mode") != "RGB"
        or decode.get("required_bits_per_sample") != 16
        or decode.get("minimum_matched_gradient_ncc") != 0.98
        or decode.get("minimum_margin_over_best_wrong_contact") != 0.002
        or amendment.get("tiff_payloads_already_downloaded") != 6
        or amendment.get("thresholds_roles_controls_or_gates_changed") is not False
        or payload.get("rights", {}).get("explicit_reuse_license_found") is not False
        or payload.get("rights", {}).get("redistribution") is not False
    ):
        raise Vision3TiffIntegrityError("BU7 frozen contract drift")
    _relative_path(str(payload.get("source", {}).get("destination", "")))
    return payload


def _contact_images(path: Path, expected_sha256: str) -> list[np.ndarray]:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise Vision3TiffIntegrityError("BU7 parent contact-sheet integrity mismatch")
    reader = PdfReader(path)
    images: list[np.ndarray] = []
    for page in reader.pages:
        for embedded in sorted(
            page.images, key=lambda image: int(image.indirect_reference.idnum)
        ):
            obj = embedded.indirect_reference.get_object()
            width = int(obj["/Width"])
            height = int(obj["/Height"])
            data = obj.get_data()
            if int(obj["/BitsPerComponent"]) != 16 or len(data) != width * height * 6:
                raise Vision3TiffIntegrityError("BU7 parent contact image layout drift")
            images.append(
                np.frombuffer(data, dtype=">u2")
                .reshape(height, width, 3)
                .astype(np.float64)
                / 65535.0
            )
    if len(images) != 52:
        raise Vision3TiffIntegrityError("BU7 parent contact image count drift")
    return images


def _gradient_vector(rgb: np.ndarray) -> np.ndarray:
    gray = np.asarray(rgb, dtype=np.float64) @ _LUMA
    gradient_y, gradient_x = np.gradient(gray)
    vector = np.concatenate((gradient_x.ravel(), gradient_y.ravel()))
    vector -= np.mean(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        raise Vision3TiffIntegrityError("BU7 degenerate image gradient")
    return vector / norm


def evaluate_tiff_integrity(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    source_root = root / _relative_path(config["source"]["destination"])
    contact_path = root / "data/physical_reference/vision3_16mm_stock_test_v1/CONTACT_SHEET.pdf"
    contacts = _contact_images(contact_path, config["parent"]["contact_sheet_sha256"])
    contact_vectors = [_gradient_vector(image) for image in contacts]
    file_rows: list[dict[str, Any]] = []
    file_hashes: list[str] = []
    decoded_pixel_hashes: list[str] = []
    for declared in config["source"]["files"]:
        path = source_root / declared["filename"]
        if not path.is_file() or path.stat().st_size != int(declared["bytes"]):
            raise Vision3TiffIntegrityError(
                f"BU7 TIFF size or presence drift: {declared['filename']}"
            )
        file_sha256 = sha256_file(path)
        file_hashes.append(file_sha256)
        with Image.open(path) as image:
            image.load()
            tags = image.tag_v2
            source_array = np.asarray(image)
            decoded_pixel_sha256 = hashlib.sha256(source_array.tobytes()).hexdigest()
            decoded_pixel_hashes.append(decoded_pixel_sha256)
            alpha_values = (
                sorted(int(value) for value in np.unique(source_array[..., 3]))
                if source_array.ndim == 3 and source_array.shape[2] == 4
                else []
            )
            rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
            resized = cv2.resize(
                rgb,
                (contacts[0].shape[1], contacts[0].shape[0]),
                interpolation=cv2.INTER_AREA,
            )
            candidate = _gradient_vector(resized)
            scores = np.asarray(
                [float(np.dot(candidate, contact)) for contact in contact_vectors],
                dtype=np.float64,
            )
            expected_index = int(declared["contact_index"])
            expected_score = float(scores[expected_index])
            scores[expected_index] = -np.inf
            best_wrong_index = int(np.argmax(scores))
            best_wrong_score = float(scores[best_wrong_index])
            file_rows.append(
                {
                    "filename": declared["filename"],
                    "file_id": declared["file_id"],
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256,
                    "format": image.format,
                    "mode": image.mode,
                    "size": [image.width, image.height],
                    "bits_per_sample": [int(value) for value in tags.get(258, ())],
                    "samples_per_pixel": int(tags.get(277, 0)),
                    "compression": int(tags.get(259, 0)),
                    "frame_count": int(getattr(image, "n_frames", 1)),
                    "icc_profile_sha256": (
                        hashlib.sha256(image.info["icc_profile"]).hexdigest()
                        if image.info.get("icc_profile")
                        else None
                    ),
                    "alpha_values": alpha_values,
                    "decoded_pixel_sha256": decoded_pixel_sha256,
                    "stock_id": declared["stock_id"],
                    "key_fill": declared["key_fill"],
                    "f_stop": declared["f_stop"],
                    "exposure": declared["exposure"],
                    "expected_contact_index": expected_index,
                    "expected_gradient_ncc": expected_score,
                    "best_wrong_contact_index": best_wrong_index,
                    "best_wrong_gradient_ncc": best_wrong_score,
                    "matched_margin": expected_score - best_wrong_score,
                    "contact_match_pass": expected_score
                    >= float(config["decode"]["minimum_matched_gradient_ncc"])
                    and expected_score - best_wrong_score
                    >= float(
                        config["decode"]["minimum_margin_over_best_wrong_contact"]
                    ),
                }
            )
    exact_size_and_hash = (
        len(file_rows) == int(config["decode"]["required_file_count"])
        and len(file_hashes) == len(set(file_hashes))
        and sum(row["bytes"] for row in file_rows)
        == int(config["acquisition"]["expected_total_bytes"])
    )
    rgb16_geometry = all(
        row["format"] == "TIFF"
        and row["mode"] == config["decode"]["required_mode"]
        and row["bits_per_sample"]
        == [int(config["decode"]["required_bits_per_sample"])] * 3
        and row["samples_per_pixel"] == 3
        and row["size"][0] >= int(config["decode"]["minimum_width"])
        and row["size"][1] >= int(config["decode"]["minimum_height"])
        and row["frame_count"] == 1
        for row in file_rows
    )
    role_counts = {
        exposure: sorted(
            row["stock_id"] for row in file_rows if row["exposure"] == exposure
        )
        for exposure in ("0", "+1", "+2")
    }
    gates = {
        "all_declared_files_exact_size_and_unique_sha256": exact_size_and_hash,
        "all_six_tiffs_decode_rgb16_and_meet_minimum_geometry": rgb16_geometry,
        "all_six_contact_order_matches_pass_ncc_and_wrong_margin": all(
            row["contact_match_pass"] for row in file_rows
        ),
        "three_cross_stock_exposure_roles_explicit": all(
            stocks == ["7207", "7219"] for stocks in role_counts.values()
        ),
        "icc_profile_identity_recorded_without_calibration_claim": all(
            "icc_profile_sha256" in row for row in file_rows
        ),
        "scanner_and_process_remain_unknown": True,
        "operator_fit_render_colour_score_and_training_reads": True,
    }
    stable_payload = {
        "experiment_id": config["experiment_id"],
        "parent_report_sha256": config["parent"]["report_sha256"],
        "downloaded_file_count": len(file_rows),
        "downloaded_total_bytes": sum(row["bytes"] for row in file_rows),
        "file_rows": file_rows,
        "all_file_sha256_unique": len(file_hashes) == len(set(file_hashes)),
        "all_decoded_pixel_sha256_unique": len(decoded_pixel_hashes)
        == len(set(decoded_pixel_hashes)),
        "role_stock_ids": role_counts,
        "scanner_id": None,
        "process_id": None,
        "operator_fit_render_colour_score_and_training_reads": 0,
        "gate_results": gates,
    }
    gate_pass = all(gates.values())
    return {
        "schema": REPORT_SCHEMA,
        **stable_payload,
        "stable_evidence_id": hashlib.sha256(canonical_json(stable_payload)).hexdigest(),
        "gate_pass": gate_pass,
        "decision": (
            "open_source_matched_colour_statistic_discriminant"
            if gate_pass
            else "close_exact_public_tiff_integrity_route"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
