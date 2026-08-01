"""Registration feasibility for the exact BP0 B&W composite-derived pairs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from src.eval.flickr_single_author_pair_acquisition import canonical_bytes, sha256_file
from src.eval.flickr_single_author_pair_registration import register_pair


SCHEMA = "neuro-film.u5-r2bp1-flickr-bw-composite-pair-registration.v1"


class FlickrBwCompositeRegistrationError(ValueError):
    """Raised when the BP1 contract or exact derived pixels drift."""


def _validate(root: Path, config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if config.get("schema") != SCHEMA or config.get("status") != "contract_frozen_before_formal_execution":
        raise FlickrBwCompositeRegistrationError("invalid BP1 contract")
    loaded: dict[str, dict[str, Any]] = {}
    for name, parent in config["parents"].items():
        path = root / str(parent["path"])
        if sha256_file(path) != parent["sha256"]:
            raise FlickrBwCompositeRegistrationError(f"parent hash drift: {name}")
        loaded[name] = json.loads(path.read_text(encoding="utf-8"))
    integrity = loaded["integrity_report"]
    if (
        not integrity.get("automatic_pass")
        or integrity.get("stable_evidence_id")
        != config["parents"]["integrity_report"]["required_stable_evidence_id"]
    ):
        raise FlickrBwCompositeRegistrationError("integrity parent did not pass exactly")
    return integrity, loaded["download_manifest"]


def _load(path: Path, expected_sha: str) -> np.ndarray:
    if sha256_file(path) != expected_sha:
        raise FlickrBwCompositeRegistrationError(f"derived pixel hash drift: {path}")
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.uint8)


def evaluate(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    _, manifest = _validate(root, config)
    rows = manifest.get("rows", [])
    if len(rows) != int(config["dataset_gates"]["expected_pairs"]):
        raise FlickrBwCompositeRegistrationError("pair count drift")
    data_root = root / str(config["data_root"])
    results: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda value: int(value["scene_id"])):
        digital = _load(data_root / row["digital_local_path"], row["digital_sha256"])
        film = _load(data_root / row["film_local_path"], row["film_sha256"])
        _, diagnostics = register_pair(digital, film, config["registration"])
        results.append(
            {
                "scene_id": int(row["scene_id"]),
                "photo_id": str(row["photo_id"]),
                "digital_local_path": row["digital_local_path"],
                "digital_sha256": row["digital_sha256"],
                "film_local_path": row["film_local_path"],
                "film_sha256": row["film_sha256"],
                "diagnostics": diagnostics,
            }
        )
    passed = [row for row in results if row["diagnostics"]["registration_gate_passed"]]
    gates = config["dataset_gates"]
    checks = {
        "expected_pairs": len(results) == int(gates["expected_pairs"]),
        "minimum_registered_pairs": len(passed) >= int(gates["minimum_registered_pairs"]),
    }
    stable = {
        "schema": "neuro-film.u5-r2bp1-flickr-bw-composite-pair-registration-report.v1",
        "node": config["node"],
        "parent_report_sha256": config["parents"]["integrity_report"]["sha256"],
        "parent_manifest_sha256": config["parents"]["download_manifest"]["sha256"],
        "metrics": {
            "pairs": len(results),
            "registered_pairs": len(passed),
            "registered_fraction": len(passed) / len(results),
            "failure_reasons": dict(
                sorted(
                    Counter(row["diagnostics"]["failure_reason"] or "passed" for row in results).items()
                )
            ),
        },
        "checks": checks,
        "automatic_pass": all(checks.values()),
        "branch": config["branches"]["pass" if all(checks.values()) else "fail"],
        "pairs": results,
        "training_allowed": False,
        "operator_fitting_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    return {**stable, "stable_evidence_id": hashlib.sha256(canonical_bytes(stable)).hexdigest()}


def render_contact_sheet(root: Path, report: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    if not report.get("automatic_pass"):
        raise FlickrBwCompositeRegistrationError("contact sheet requires automatic pass")
    data_root = root / str(config["data_root"])
    cell_w, cell_h = 300, 230
    canvas = Image.new("RGB", (cell_w * 3, cell_h * len(report["pairs"])), "#181818")
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(report["pairs"]):
        y = index * cell_h
        digital = _load(data_root / row["digital_local_path"], row["digital_sha256"])
        film = _load(data_root / row["film_local_path"], row["film_sha256"])
        homography = row["diagnostics"].get("homography_digital_to_film")
        aligned = (
            np.zeros_like(film)
            if homography is None
            else cv2.warpPerspective(
                digital,
                np.asarray(homography, dtype=np.float64),
                (film.shape[1], film.shape[0]),
                flags=cv2.INTER_LINEAR,
            )
        )
        overlay = np.rint(0.5 * aligned.astype(np.float32) + 0.5 * film.astype(np.float32)).astype(
            np.uint8
        )
        for column, (name, image) in enumerate(
            (("digital", digital), ("film", film), ("overlay", overlay))
        ):
            preview = ImageOps.contain(Image.fromarray(image), (cell_w - 8, cell_h - 30))
            canvas.paste(preview, (column * cell_w + (cell_w - preview.width) // 2, y + 4))
            draw.text((column * cell_w + 6, y + cell_h - 22), f"{row['scene_id']:02d} {name}", fill="white")
        status = "PASS" if row["diagnostics"]["registration_gate_passed"] else "FAIL"
        draw.text((cell_w * 3 - 56, y + cell_h - 22), status, fill="#6cff6c" if status == "PASS" else "#ff6c6c")
    path = root / str(config["visual_review"]["contact_sheet"])
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92, subsampling=0)
    return {"path": path.as_posix(), "sha256": sha256_file(path), "pairs": len(report["pairs"])}


__all__ = [
    "FlickrBwCompositeRegistrationError",
    "SCHEMA",
    "evaluate",
    "render_contact_sheet",
]
