"""U1.4C9 embedded ProPhoto ICC semantic-ingress conformance."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import imagecodecs
import numpy as np
import tifffile
from PIL import ImageCms

from src.eval.fujifilm_dye_basis_measured_conformance import canonical_json, hash_file
from src.preprocess import load_working_image
from src.preprocess.prophoto_icc import prophoto_matrix_shaper_facts

SCHEMA = "neuro-film.u1-4c9-embedded-prophoto-icc-semantic-ingress-contract.v1"
REPORT_SCHEMA = "neuro-film.u1-4c9-embedded-prophoto-icc-semantic-ingress-report.v1"
EXPERIMENT_ID = "U1.4C9"
CONTRACT_SHA256 = "e008e56eefb54ea4c51b50602d80794e603036ce53a8bf76c6c7c772287a6b03"

_D50_TO_D65_BRADFORD = np.asarray(
    [
        [0.9555766, -0.0230393, 0.0631636],
        [-0.0282895, 1.0099416, 0.0210077],
        [0.0122982, -0.0204830, 1.3299098],
    ],
    dtype=np.float64,
)
_XYZ_D65_TO_LINEAR_REC2020 = np.asarray(
    [
        [30757411 / 17917100, -6372589 / 17917100, -4539589 / 17917100],
        [-19765991 / 29648200, 47925759 / 29648200, 467509 / 29648200],
        [792561 / 44930125, -1921689 / 44930125, 42328811 / 44930125],
    ],
    dtype=np.float64,
)
_C4_PROPHOTO_TO_XYZ_D50 = np.asarray(
    [
        [0.7976749, 0.1351917, 0.0313534],
        [0.2880402, 0.7118741, 0.0000857],
        [0.0, 0.0, 0.82521],
    ],
    dtype=np.float32,
).astype(np.float64)


class ProPhotoSemanticIngressError(RuntimeError):
    """Raised when the frozen C9 execution contract drifts."""


def _relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ProPhotoSemanticIngressError("C9 paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    if hash_file(path) != CONTRACT_SHA256:
        raise ProPhotoSemanticIngressError("C9 contract hash drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise ProPhotoSemanticIngressError("C9 contract structure drift")
    for parent in payload["parents"].values():
        _relative(parent["path"])
    if int(payload["reference"]["stripe_rows"]) <= 0:
        raise ProPhotoSemanticIngressError("C9 stripe size is invalid")
    return payload


def _validated_rows(contract: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    for parent in contract["parents"].values():
        path = root / _relative(parent["path"])
        if not path.is_file() or hash_file(path) != parent["sha256"]:
            raise ProPhotoSemanticIngressError("C9 parent identity drift")
    manifest_path = root / _relative(contract["parents"]["c4_source_manifest"]["path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("rows")
    source = contract["source"]
    if (
        manifest.get("schema")
        != "neuro_film.u1-4c4-native-prophoto-source-manifest.v1"
        or not isinstance(rows, list)
        or len(rows) != int(source["expected_rows"])
        or len({row["id"] for row in rows}) != len(rows)
        or manifest.get("embedded_icc_sha256")
        != source["expected_embedded_icc_sha256"]
        or manifest.get("allowed_use") != source["required_allowed_use"]
        or manifest.get("rights_scope") != source["required_rights_scope"]
    ):
        raise ProPhotoSemanticIngressError("C9 source manifest drift")
    return [dict(row) for row in rows]


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _c4_romm_to_rec2020(encoded: np.ndarray) -> np.ndarray:
    linear = np.where(
        encoded < (16.0 / 512.0), encoded / 16.0, np.power(encoded, 1.8)
    )
    xyz_d50 = linear @ _C4_PROPHOTO_TO_XYZ_D50.T
    xyz_d65 = xyz_d50 @ _D50_TO_D65_BRADFORD.T
    return xyz_d65 @ _XYZ_D65_TO_LINEAR_REC2020.T


def evaluate(
    contract: Mapping[str, Any], root: Path, output_dir: Path
) -> dict[str, Any]:
    if imagecodecs.__version__ != contract["reference"]["imagecodecs_version"]:
        raise ProPhotoSemanticIngressError("C9 imagecodecs version drift")
    if ImageCms.core.littlecms_version != contract["reference"]["littlecms_version"]:
        raise ProPhotoSemanticIngressError("C9 LittleCMS version drift")
    rows = _validated_rows(contract, root)
    stripe_rows = int(contract["reference"]["stripe_rows"])
    xyz_profile = imagecodecs.cms_profile("XYZ")
    result_rows: list[dict[str, Any]] = []
    product_reference_maximum = 0.0
    product_reference_sum = 0.0
    c4_product_maximum = 0.0
    c4_product_sum = 0.0
    scalar_count = 0
    for row in rows:
        source_path = root / _relative(row["path"])
        if not source_path.is_file() or hash_file(source_path) != row["sha256"]:
            raise ProPhotoSemanticIngressError(f"C9 source identity drift: {row['id']}")
        working = load_working_image(source_path)
        with tifffile.TiffFile(source_path) as tif:
            page = tif.pages[0]
            encoded = page.asarray()
            profile_tag = page.tags.get(34675)
            profile = bytes(profile_tag.value) if profile_tag is not None else b""
        if (
            encoded.dtype != np.uint16
            or encoded.shape != working.pixels.shape
            or hashlib.sha256(profile).hexdigest()
            != contract["source"]["expected_embedded_icc_sha256"]
            or working.working_space != "linear_rec2020"
            or working.transfer_state != "display_linear"
            or working.pixels.dtype != np.float32
        ):
            raise ProPhotoSemanticIngressError(f"C9 decoded structure drift: {row['id']}")
        profile_facts = prophoto_matrix_shaper_facts(profile)
        row_reference_maximum = 0.0
        row_reference_sum = 0.0
        row_c4_maximum = 0.0
        row_c4_sum = 0.0
        for start in range(0, encoded.shape[0], stripe_rows):
            stop = min(encoded.shape[0], start + stripe_rows)
            stripe = np.ascontiguousarray(encoded[start:stop])
            reference_xyz = imagecodecs.cms_transform(
                stripe,
                profile,
                xyz_profile,
                outdtype=np.float64,
                intent=imagecodecs.CMS.INTENT.RELATIVE_COLORIMETRIC,
            )
            reference = (
                reference_xyz @ _D50_TO_D65_BRADFORD.T
            ) @ _XYZ_D65_TO_LINEAR_REC2020.T
            actual = working.pixels[start:stop].astype(np.float64)
            reference_delta = np.abs(actual - reference)
            encoded_float = stripe.astype(np.float64) / 65535.0
            c4_delta = np.abs(actual - _c4_romm_to_rec2020(encoded_float))
            row_reference_maximum = max(
                row_reference_maximum, float(np.max(reference_delta))
            )
            row_reference_sum += float(np.sum(reference_delta, dtype=np.float64))
            row_c4_maximum = max(row_c4_maximum, float(np.max(c4_delta)))
            row_c4_sum += float(np.sum(c4_delta, dtype=np.float64))
        row_scalars = int(encoded.size)
        scalar_count += row_scalars
        product_reference_maximum = max(
            product_reference_maximum, row_reference_maximum
        )
        product_reference_sum += row_reference_sum
        c4_product_maximum = max(c4_product_maximum, row_c4_maximum)
        c4_product_sum += row_c4_sum
        result_rows.append(
            {
                "id": row["id"],
                "source_sha256": row["sha256"],
                "shape": list(encoded.shape),
                "profile_gamma": profile_facts["gamma"],
                "product_array_sha256": _array_sha256(working.pixels),
                "product_minimum": float(np.min(working.pixels)),
                "product_maximum": float(np.max(working.pixels)),
                "maximum_product_vs_littlecms_absolute_error": row_reference_maximum,
                "mean_product_vs_littlecms_absolute_error": row_reference_sum
                / row_scalars,
                "maximum_product_vs_c4_romm_absolute_error": row_c4_maximum,
                "mean_product_vs_c4_romm_absolute_error": row_c4_sum / row_scalars,
            }
        )
    gates = contract["gates"]
    checks = {
        "recognized_rows": len(result_rows) == int(gates["required_recognized_rows"]),
        "product_vs_littlecms": product_reference_maximum
        <= float(gates["maximum_product_vs_littlecms_absolute_error"]),
        "float32_linear_rec2020": all(
            row["shape"][2] == 3 for row in result_rows
        ),
        "unclipped_execution": any(row["product_minimum"] < 0.0 for row in result_rows),
    }
    automatic_pass = all(checks.values())
    stable = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "contract_sha256": CONTRACT_SHA256,
        "rows": result_rows,
        "aggregate": {
            "source_count": len(result_rows),
            "scalar_count": scalar_count,
            "maximum_product_vs_littlecms_absolute_error": product_reference_maximum,
            "mean_product_vs_littlecms_absolute_error": product_reference_sum
            / scalar_count,
            "maximum_product_vs_c4_romm_absolute_error": c4_product_maximum,
            "mean_product_vs_c4_romm_absolute_error": c4_product_sum / scalar_count,
        },
        "checks": checks,
        "automatic_pass": automatic_pass,
        "decision_without_replay": (
            contract["decisions"]["pass"]
            if automatic_pass
            else contract["decisions"]["fail"]
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    stable["stable_evidence_id"] = hashlib.sha256(canonical_json(stable)).hexdigest()
    report = dict(stable)
    report["output_dir"] = str(output_dir)
    report["report_sha256"] = hashlib.sha256(canonical_json(report)).hexdigest()
    return report


__all__ = [
    "CONTRACT_SHA256",
    "ProPhotoSemanticIngressError",
    "evaluate",
    "load_contract",
]
