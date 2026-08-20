"""Build the rights-locked SF3.A0K three-stock display stimulus pack."""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.preprocess import srgb_icc_profile

SCHEMA = "neuro-film.sf3-a0k-three-stock-display-stimulus-contract.v1"
REPORT_SCHEMA = "neuro-film.sf3-a0k-three-stock-display-stimulus-report.v1"
MANIFEST_SCHEMA = "neuro-film.sf3-a0k-three-stock-display-stimulus-manifest.v1"


class ThreeStockDisplayStimulusError(RuntimeError):
    """Raised when a frozen SF3.A0K binding or create-only invariant fails."""


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ThreeStockDisplayStimulusError("SF3.A0K paths must be repository-relative")
    return path


def _write_create(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise ThreeStockDisplayStimulusError(f"SF3.A0K refuses to overwrite {path}") from error


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    roles = value.get("roles", {})
    development = roles.get("development", [])
    confirmation = roles.get("confirmation", [])
    ids = [*development, *confirmation]
    diagnostics = value.get("diagnostic_stimuli", {})
    if (
        value.get("schema") != SCHEMA
        or value.get("experiment_id") != "SF3.A0K"
        or len(development) != 8
        or len(confirmation) != 4
        or len(ids) != len(set(ids))
        or diagnostics.get("ids")
        != ["neutral_step_wedge_17", "rgb_primary_secondary_ramps", "slanted_edge_line_pairs"]
        or diagnostics.get("count_toward_scene_minima") is not False
        or value.get("output", {}).get("create_only") is not True
    ):
        raise ThreeStockDisplayStimulusError("SF3.A0K frozen contract drift")
    for parent in value["parents"].values():
        _relative(parent["path"])
    _relative(value["output"]["canonical_pack_root"])
    _relative(value["output"]["formal_run_root"])
    return value


def _png_bytes(array: np.ndarray) -> bytes:
    image = Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, "PNG", compress_level=6, icc_profile=srgb_icc_profile())
    return buffer.getvalue()


def _diagnostics(width: int, height: int) -> dict[str, bytes]:
    wedge = np.empty((height, width, 3), dtype=np.uint8)
    steps = np.linspace(0, 255, 17).round().astype(np.uint8)
    columns = np.minimum(np.arange(width) * 17 // width, 16)
    wedge[:] = steps[columns][None, :, None]

    ramps = np.zeros((height, width, 3), dtype=np.uint8)
    ramp = np.linspace(0, 255, width).round().astype(np.uint8)
    bands = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1]],
        dtype=np.uint8,
    )
    for index, mask in enumerate(bands):
        y0, y1 = index * height // 6, (index + 1) * height // 6
        ramps[y0:y1] = ramp[None, :, None] * mask[None, None, :]

    yy, xx = np.indices((height, width))
    edge = np.zeros((height, width), dtype=np.uint8)
    edge[xx > (width * 0.5 + 0.125 * (yy - height * 0.5))] = 255
    line_pairs = ((xx // np.maximum(1, 2 ** (1 + (yy * 6 // height)))) % 2) * 255
    edge[height // 2 :] = line_pairs[height // 2 :].astype(np.uint8)
    spatial = np.repeat(edge[:, :, None], 3, axis=2)
    return {
        "neutral_step_wedge_17": _png_bytes(wedge),
        "rgb_primary_secondary_ramps": _png_bytes(ramps),
        "slanted_edge_line_pairs": _png_bytes(spatial),
    }


def _bound_parent(root: Path, parent: Mapping[str, Any], name: str) -> Path:
    path = root / _relative(str(parent["path"]))
    if _sha(path) != parent["sha256"]:
        raise ThreeStockDisplayStimulusError(f"SF3.A0K {name} hash drift")
    return path


def build_pack(contract: Mapping[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise ThreeStockDisplayStimulusError("SF3.A0K output directory must not exist")
    acquisition_path = _bound_parent(root, contract["parents"]["controlled_acquisition_contract"], "acquisition")
    source_manifest_path = _bound_parent(root, contract["parents"]["digital_source_manifest"], "source manifest")
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    if acquisition.get("schema") != "neuro-film.sf3-a0-three-stock-controlled-acquisition-contract.v1":
        raise ThreeStockDisplayStimulusError("SF3.A0 parent schema drift")
    source_rows = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    indexed = {row["id"]: row for row in source_rows}
    if len(indexed) != len(source_rows):
        raise ThreeStockDisplayStimulusError("duplicate source IDs")

    policy = contract["required_source_policy"]
    scene_rows: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    for role in ("development", "confirmation"):
        for order, source_id in enumerate(contract["roles"][role]):
            row = indexed.get(source_id)
            if row is None:
                raise ThreeStockDisplayStimulusError(f"missing source {source_id}")
            for key in ("allowed_use", "rights_scope", "decoded_color_state"):
                if row.get(key) != policy[key]:
                    raise ThreeStockDisplayStimulusError(f"{source_id} {key} drift")
            source_path = root / _relative(row["decoded_path"])
            payload = source_path.read_bytes()
            if _sha_bytes(payload) != row["decoded_sha256"]:
                raise ThreeStockDisplayStimulusError(f"{source_id} decoded hash drift")
            with Image.open(io.BytesIO(payload)) as image:
                if image.mode != "RGB" or image.size != (row["width"], row["height"]):
                    raise ThreeStockDisplayStimulusError(f"{source_id} decoded image drift")
            relative = Path("scenes") / role / f"{order + 1:02d}_{source_id}.png"
            _write_create(output_dir / relative, payload)
            scene_rows.append(
                {
                    "role": role,
                    "role_order": order,
                    "scene_id": source_id,
                    "scene_content_group": f"sf3-a0k:{role}:{source_id}",
                    "source_id": row["source_id"],
                    "source_url": row["source_url"],
                    "source_raw_sha256": row["raw_sha256"],
                    "source_decoded_path": row["decoded_path"],
                    "source_decoded_sha256": row["decoded_sha256"],
                    "stimulus_relative_path": relative.as_posix(),
                    "stimulus_sha256": _sha_bytes(payload),
                    "width": row["width"],
                    "height": row["height"],
                    "allowed_use": row["allowed_use"],
                    "rights_scope": row["rights_scope"],
                    "decoded_color_state": row["decoded_color_state"],
                }
            )
            inventory.append({"path": relative.as_posix(), "sha256": _sha_bytes(payload), "bytes": len(payload)})

    diag_payloads = _diagnostics(
        int(contract["diagnostic_stimuli"]["canvas_width"]),
        int(contract["diagnostic_stimuli"]["canvas_height"]),
    )
    diagnostic_rows = []
    for diagnostic_id in contract["diagnostic_stimuli"]["ids"]:
        payload = diag_payloads[diagnostic_id]
        relative = Path("diagnostics") / f"{diagnostic_id}.png"
        _write_create(output_dir / relative, payload)
        diagnostic_rows.append(
            {
                "diagnostic_id": diagnostic_id,
                "counts_toward_scene_minima": False,
                "relative_path": relative.as_posix(),
                "sha256": _sha_bytes(payload),
                "width": contract["diagnostic_stimuli"]["canvas_width"],
                "height": contract["diagnostic_stimuli"]["canvas_height"],
            }
        )
        inventory.append({"path": relative.as_posix(), "sha256": _sha_bytes(payload), "bytes": len(payload)})

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "parent_acquisition_contract_sha256": contract["parents"]["controlled_acquisition_contract"]["sha256"],
        "parent_source_manifest_sha256": contract["parents"]["digital_source_manifest"]["sha256"],
        "required_stocks": list(acquisition["required_stocks"]),
        "scene_rows": scene_rows,
        "diagnostic_rows": diagnostic_rows,
        "scene_counts": {"development": 8, "confirmation": 4},
        "claim_ceiling": contract["claim_ceiling"],
    }
    manifest_payload = _canonical(manifest)
    manifest_relative = Path(contract["output"]["manifest_name"])
    _write_create(output_dir / manifest_relative, manifest_payload)
    inventory.append({"path": manifest_relative.as_posix(), "sha256": _sha_bytes(manifest_payload), "bytes": len(manifest_payload)})
    inventory.sort(key=lambda item: item["path"])
    stable_payload = {
        "contract_sha256": _sha(root / "configs/sf3_a0k_three_stock_display_stimulus_v1.json"),
        "manifest_sha256": _sha_bytes(manifest_payload),
        "inventory": inventory,
        "scene_counts": manifest["scene_counts"],
        "diagnostic_count": len(diagnostic_rows),
        "decision": contract["decision_if_pass"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": REPORT_SCHEMA,
        **stable_payload,
        "stable_evidence_id": _sha_bytes(_canonical(stable_payload)),
        "automatic_pass": True,
    }


def write_report(report: Mapping[str, Any], path: Path) -> str:
    payload = _canonical(report)
    _write_create(path, payload)
    return _sha_bytes(payload)
