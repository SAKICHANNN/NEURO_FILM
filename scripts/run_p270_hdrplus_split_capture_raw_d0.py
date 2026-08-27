from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]


class P270Error(RuntimeError):
    pass


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5_base64(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode("ascii")


@dataclass
class ReadLedger:
    events: list[dict[str, str]] = field(default_factory=list)
    heldout_allowed: bool = False
    freeze_identity: str | None = None
    freeze_event_index: int | None = None

    def record(self, path: Path, role: str) -> None:
        if role == "heldout" and not self.heldout_allowed:
            raise P270Error("heldout frame read before candidate freeze")
        self.events.append({"path": path.name, "role": role})

    def freeze(self, identity: str) -> None:
        if self.heldout_allowed:
            raise P270Error("candidate freeze called twice")
        self.freeze_identity = identity
        self.freeze_event_index = len(self.events)
        self.heldout_allowed = True


def _verify_source_roles(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    parent_config_path = ROOT / "configs/p269_hdrplus_one_burst_acquisition_v1.json"
    parent = json.loads(parent_config_path.read_text(encoding="utf-8"))
    expected = {
        item["path"]: item
        for item in parent["objects"]
        if item["role"] == "burst" and item["path"].endswith(".dng")
    }
    roles = list(config["development_frames"]) + list(config["heldout_frames"])
    if len(roles) != 8 or len(set(roles)) != 8 or set(roles) != set(expected):
        raise P270Error("P270 frame roles differ from exact P269 input DNG inventory")
    source_root = ROOT / config["source_root"]
    for name in roles:
        path = source_root / name
        item = expected[name]
        if (
            path.stat().st_size != item["bytes"]
            or _md5_base64(path) != item["md5_base64"]
        ):
            raise P270Error("P270 source DNG identity differs from P269")
    return expected


def _load_cfa_planes(
    path: Path,
    config: dict[str, Any],
    ledger: ReadLedger,
    role: str,
) -> list[np.ndarray]:
    ledger.record(path, role)
    with tifffile.TiffFile(path) as tif:
        page = max(tif.pages, key=lambda candidate: int(np.prod(candidate.shape)))
        if (
            page.compression.name != "NONE"
            or page.dtype != np.dtype("uint16")
            or page.samplesperpixel != 1
            or len(page.shape) != 2
        ):
            raise P270Error("source DNG is not the frozen uncompressed uint16 mosaic")
    raw = tifffile.memmap(path)
    white = np.float32(config["representation"]["white_level"])
    planes = [
        np.ascontiguousarray(raw[y::2, x::2], dtype=np.float32) / white
        for y, x in config["representation"]["cfa_planes"]
    ]
    del raw
    return planes


def _green_proxy(planes: list[np.ndarray], scale: float) -> np.ndarray:
    green = np.float32(0.5) * (planes[1] + planes[2])
    return cv2.resize(
        green,
        dsize=None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )


def _warp(plane: np.ndarray, shift_xy: tuple[float, float]) -> np.ndarray:
    dx, dy = shift_xy
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(
        plane,
        matrix,
        (plane.shape[1], plane.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float("nan"),
    )


def _central_rmse(reference: np.ndarray, candidate: np.ndarray) -> float:
    margin_y = max(8, reference.shape[0] // 10)
    margin_x = max(8, reference.shape[1] // 10)
    ref = reference[margin_y:-margin_y, margin_x:-margin_x]
    cand = candidate[margin_y:-margin_y, margin_x:-margin_x]
    valid = np.isfinite(cand)
    if int(valid.sum()) < 4096:
        raise P270Error("alignment comparison has insufficient overlap")
    delta = ref[valid].astype(np.float64) - cand[valid].astype(np.float64)
    return float(np.sqrt(np.mean(delta * delta)))


def _align_to_anchor(
    anchor_planes: list[np.ndarray],
    source_planes: list[np.ndarray],
    scale: float,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    anchor_green = _green_proxy(anchor_planes, scale)
    source_green = _green_proxy(source_planes, scale)
    shift_small, response = cv2.phaseCorrelate(anchor_green, source_green)
    proposed = (float(shift_small[0] / scale), float(shift_small[1] / scale))
    positive = proposed
    negative = (-proposed[0], -proposed[1])
    source_full_green = np.float32(0.5) * (source_planes[1] + source_planes[2])
    anchor_full_green = np.float32(0.5) * (anchor_planes[1] + anchor_planes[2])
    positive_rmse = _central_rmse(anchor_full_green, _warp(source_full_green, positive))
    negative_rmse = _central_rmse(anchor_full_green, _warp(source_full_green, negative))
    chosen = positive if positive_rmse <= negative_rmse else negative
    aligned = [_warp(plane, chosen) for plane in source_planes]
    unaligned_rmse = _central_rmse(anchor_full_green, source_full_green)
    aligned_rmse = _central_rmse(
        anchor_full_green, np.float32(0.5) * (aligned[1] + aligned[2])
    )
    return aligned, {
        "phase_response": float(response),
        "proposed_shift_xy": list(proposed),
        "positive_rmse": positive_rmse,
        "negative_rmse": negative_rmse,
        "chosen_shift_xy": list(chosen),
        "unaligned_rmse": unaligned_rmse,
        "aligned_rmse": aligned_rmse,
    }


def _hash_planes(planes: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for plane in planes:
        digest.update(np.ascontiguousarray(plane, dtype="<f4").tobytes())
    return digest.hexdigest()


def _mean_and_std(
    stacks: list[list[np.ndarray]],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []
    for plane_index in range(4):
        stack = np.stack([frame[plane_index] for frame in stacks])
        valid = np.all(np.isfinite(stack), axis=0)
        mean = np.mean(stack, axis=0, dtype=np.float32)
        std = np.std(stack, axis=0, dtype=np.float32)
        mean[~valid] = np.nan
        std[~valid] = np.nan
        means.append(mean)
        stds.append(std)
        valid_masks.append(valid)
    return means, stds, valid_masks


def _select_blocks(
    candidate: list[np.ndarray],
    temporal_std: list[np.ndarray],
    valid_masks: list[np.ndarray],
    block_side: int,
    count: int,
    gradient_weight: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for plane_index, plane in enumerate(candidate):
        gx = cv2.Sobel(plane, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(plane, cv2.CV_32F, 0, 1, ksize=3)
        gradient = cv2.magnitude(gx, gy)
        height, width = plane.shape
        for y in range(0, height - block_side + 1, block_side):
            for x in range(0, width - block_side + 1, block_side):
                area = np.s_[y : y + block_side, x : x + block_side]
                if not bool(np.all(valid_masks[plane_index][area])):
                    continue
                risk = float(np.mean(temporal_std[plane_index][area], dtype=np.float64))
                risk += gradient_weight * float(
                    np.mean(gradient[area], dtype=np.float64)
                )
                rows.append({"plane": plane_index, "y": y, "x": x, "risk": risk})
    rows.sort(key=lambda row: (row["risk"], row["plane"], row["y"], row["x"]))
    if len(rows) < count:
        raise P270Error("insufficient source-only reliable blocks")
    return rows[:count]


def _controls(
    anchor: list[np.ndarray], sigmas: list[float]
) -> dict[str, list[np.ndarray]]:
    outputs: dict[str, list[np.ndarray]] = {
        "identity": [plane.copy() for plane in anchor]
    }
    for sigma in sigmas:
        outputs[f"gaussian_sigma_{sigma}"] = [
            cv2.GaussianBlur(
                plane,
                (0, 0),
                sigmaX=float(sigma),
                sigmaY=float(sigma),
                borderType=cv2.BORDER_REFLECT_101,
            )
            for plane in anchor
        ]
    return outputs


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    delta = left.astype(np.float64) - right.astype(np.float64)
    return float(np.sqrt(np.mean(delta * delta)))


def _reduction(candidate: float, baseline: float) -> float:
    if baseline <= 1e-12:
        return 0.0 if candidate <= 1e-12 else -1e9
    return (baseline - candidate) / baseline


def _score_blocks(
    blocks: list[dict[str, Any]],
    candidate: list[np.ndarray],
    target: list[np.ndarray],
    target_valid: list[np.ndarray],
    unaligned: list[np.ndarray],
    controls: dict[str, list[np.ndarray]],
    block_side: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block in blocks:
        plane_index, y, x = block["plane"], block["y"], block["x"]
        area = np.s_[y : y + block_side, x : x + block_side]
        if not bool(np.all(target_valid[plane_index][area])):
            continue
        truth = target[plane_index][area]
        candidate_rmse = _rmse(candidate[plane_index][area], truth)
        unaligned_rmse = _rmse(unaligned[plane_index][area], truth)
        control_errors = {
            name: _rmse(planes[plane_index][area], truth)
            for name, planes in controls.items()
        }
        strongest_name = min(
            control_errors, key=lambda name: (control_errors[name], name)
        )
        strongest_rmse = control_errors[strongest_name]
        rows.append(
            {
                "plane": plane_index,
                "y": y,
                "x": x,
                "candidate_rmse": candidate_rmse,
                "unaligned_rmse": unaligned_rmse,
                "strongest_single_name": strongest_name,
                "strongest_single_rmse": strongest_rmse,
                "vs_unaligned_reduction": _reduction(candidate_rmse, unaligned_rmse),
                "vs_strongest_single_reduction": _reduction(
                    candidate_rmse, strongest_rmse
                ),
            }
        )
    rows.sort(key=lambda row: (row["plane"], row["y"], row["x"]))
    return rows


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "win_rate": float(np.mean(array > 0.0)),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "worst": float(np.min(array)),
    }


def build_report(config_path: Path, *, reverse: bool) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _verify_source_roles(config)
    source_root = ROOT / config["source_root"]
    ledger = ReadLedger()
    development_names = list(config["development_frames"])
    heldout_names = list(config["heldout_frames"])
    load_development = (
        list(reversed(development_names)) if reverse else development_names
    )
    loaded_development = {
        name: _load_cfa_planes(source_root / name, config, ledger, "development")
        for name in load_development
    }
    anchor = loaded_development[development_names[0]]
    scale = float(config["representation"]["phase_scale"])
    aligned_frames: list[list[np.ndarray]] = []
    unaligned_frames: list[list[np.ndarray]] = []
    alignment: list[dict[str, Any]] = []
    for name in development_names:
        planes = loaded_development[name]
        unaligned_frames.append(planes)
        if name == development_names[0]:
            aligned = [plane.copy() for plane in planes]
            diagnostic = {
                "phase_response": 1.0,
                "proposed_shift_xy": [0.0, 0.0],
                "positive_rmse": 0.0,
                "negative_rmse": 0.0,
                "chosen_shift_xy": [0.0, 0.0],
                "unaligned_rmse": 0.0,
                "aligned_rmse": 0.0,
            }
        else:
            aligned, diagnostic = _align_to_anchor(anchor, planes, scale)
        aligned_frames.append(aligned)
        alignment.append({"path": name, **diagnostic})
    candidate, temporal_std, valid_masks = _mean_and_std(aligned_frames)
    unaligned, _, _ = _mean_and_std(unaligned_frames)
    control_outputs = _controls(anchor, config["controls"]["gaussian_sigmas"])
    block_config = config["block_selection"]
    blocks = _select_blocks(
        candidate,
        temporal_std,
        valid_masks,
        int(block_config["block_side"]),
        int(block_config["required_blocks"]),
        float(block_config["gradient_weight"]),
    )
    freeze = {
        "candidate_sha256": _hash_planes(candidate),
        "unaligned_control_sha256": _hash_planes(unaligned),
        "single_frame_control_sha256": {
            name: _hash_planes(planes)
            for name, planes in sorted(control_outputs.items())
        },
        "blocks_sha256": _sha256_bytes(_canonical_bytes(blocks)),
        "block_count": len(blocks),
        "development_alignment": alignment,
    }
    freeze_identity = _sha256_bytes(_canonical_bytes(freeze))
    ledger.freeze(freeze_identity)
    load_heldout = list(reversed(heldout_names)) if reverse else heldout_names
    loaded_heldout = {
        name: _load_cfa_planes(source_root / name, config, ledger, "heldout")
        for name in load_heldout
    }
    aligned_heldout: list[list[np.ndarray]] = []
    heldout_alignment: list[dict[str, Any]] = []
    for name in heldout_names:
        aligned, diagnostic = _align_to_anchor(anchor, loaded_heldout[name], scale)
        aligned_heldout.append(aligned)
        heldout_alignment.append({"path": name, **diagnostic})
    target, _, target_valid = _mean_and_std(aligned_heldout)
    block_scores = _score_blocks(
        blocks,
        candidate,
        target,
        target_valid,
        unaligned,
        control_outputs,
        int(block_config["block_side"]),
    )
    vs_single = _summary([row["vs_strongest_single_reduction"] for row in block_scores])
    vs_unaligned = _summary([row["vs_unaligned_reduction"] for row in block_scores])
    shifts = [
        abs(component)
        for row in alignment + heldout_alignment
        for component in row["chosen_shift_xy"]
    ]
    overlap = min(float(np.mean(mask)) for mask in valid_masks)
    gates_config = config["gates"]
    gates = {
        "candidate_frozen_before_heldout": ledger.freeze_event_index == 4
        and all(event["role"] == "development" for event in ledger.events[:4])
        and all(event["role"] == "heldout" for event in ledger.events[4:]),
        "shift": max(shifts)
        <= config["representation"]["maximum_abs_halfres_shift_pixels"],
        "overlap": overlap >= gates_config["minimum_aligned_even_overlap"],
        "block_support": len(block_scores)
        >= gates_config["minimum_valid_heldout_blocks"],
        "vs_strongest_single_rate": vs_single["win_rate"]
        >= gates_config["minimum_vs_strongest_single_win_rate"],
        "vs_strongest_single_median": vs_single["median"]
        >= gates_config["minimum_vs_strongest_single_median_reduction"],
        "vs_strongest_single_worst": vs_single["worst"]
        >= gates_config["minimum_vs_strongest_single_worst_reduction"],
        "vs_unaligned_rate": vs_unaligned["win_rate"]
        >= gates_config["minimum_vs_unaligned_win_rate"],
        "vs_unaligned_median": vs_unaligned["median"]
        >= gates_config["minimum_vs_unaligned_median_reduction"],
        "zero_result_and_model_reads": True,
    }
    scientific = {
        "freeze": freeze,
        "freeze_identity": freeze_identity,
        "candidate_freeze_event_index": ledger.freeze_event_index,
        "read_roles": [event["role"] for event in ledger.events],
        "heldout_alignment": heldout_alignment,
        "aligned_even_overlap": overlap,
        "valid_heldout_blocks": len(block_scores),
        "vs_strongest_single": vs_single,
        "vs_unaligned": vs_unaligned,
        "block_scores": block_scores,
        "result_pixel_reads": 0,
        "fit_train_inference_reads": 0,
        "gates": gates,
    }
    status = (
        "PASS_PRIVATE_HDRPLUS_SPLIT_CAPTURE_RAW_MECHANISM_D0"
        if all(gates.values())
        else "FAIL_CLOSED_HDRPLUS_SPLIT_CAPTURE_RAW_MECHANISM_D0"
    )
    return {
        "schema": "neuro-film.p270-hdrplus-split-capture-raw-d0-result.v1",
        "experiment_id": "P270",
        "status": status,
        "scientific_identity": _sha256_bytes(_canonical_bytes(scientific)),
        "scientific": scientific,
        "execution": {"reverse_enumeration": reverse},
        "identities": {
            "config_sha256": _sha256_file(config_path),
            "contract_sha256": _sha256_file(
                ROOT / "docs/planning/P270_HDRPLUS_SPLIT_CAPTURE_RAW_D0_CONTRACT.md"
            ),
            "runner_sha256": _sha256_file(Path(__file__).resolve()),
        },
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    report = build_report(args.config, reverse=args.reverse)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_bytes(report))
    return 0 if report["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    sys.exit(main())
