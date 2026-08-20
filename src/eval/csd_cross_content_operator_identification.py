"""Cross-content explicit-operator identification with the official CSD model.

The blind ``score`` phase reads only P401's public manifest and anonymous PNG
assets.  The ``aggregate`` phase may read the frozen private mapping only after
two score locks exist.  This separation is part of the scientific contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

FEATURE_KEYS = ("csd_style", "csd_content", "oklab_moments", "rgb_histogram")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        "utf-8",
    )


def _l2(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("descriptor has invalid norm")
    return vector / norm


def _srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        np.power((rgb + 0.055) / 1.055, 2.4),
    )
    lms = linear @ np.asarray(
        [
            [0.4122214708, 0.2119034982, 0.0883024619],
            [0.5363325363, 0.6806995451, 0.2817188376],
            [0.0514459929, 0.1073969566, 0.6299787005],
        ],
        dtype=np.float64,
    )
    lms = np.cbrt(lms)
    return lms @ np.asarray(
        [
            [0.2104542553, 1.9779984951, 0.0259040371],
            [0.7936177850, -2.4285922050, 0.7827717662],
            [-0.0040720468, 0.4505937099, -0.8086757660],
        ],
        dtype=np.float64,
    )


def fixed_descriptors(image: Image.Image) -> dict[str, list[float]]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    flat_rgb = rgb.reshape(-1, 3)
    lab = _srgb_to_oklab(flat_rgb)
    moments: list[float] = []
    for channel in range(3):
        values = lab[:, channel]
        moments.extend(
            [
                float(np.mean(values)),
                float(np.std(values)),
                *[float(item) for item in np.quantile(values, [0.1, 0.5, 0.9])],
            ]
        )
    histograms: list[float] = []
    for channel in range(3):
        hist, _ = np.histogram(flat_rgb[:, channel], bins=32, range=(0.0, 1.0))
        histograms.extend((hist.astype(np.float64) / flat_rgb.shape[0]).tolist())
    return {
        "oklab_moments": _l2(np.asarray(moments)).tolist(),
        "rgb_histogram": _l2(np.asarray(histograms)).tolist(),
    }


@dataclass(frozen=True)
class AnonymousAsset:
    presentation_id: str
    role: str
    label: str
    relative_path: str
    path: Path

    @property
    def row_id(self) -> str:
        return f"{self.presentation_id}:{self.label}"


def anonymous_assets(contract: dict[str, Any], project_root: Path) -> list[AnonymousAsset]:
    inputs = contract["consumed_inputs"]
    public_manifest_path = project_root / inputs["public_manifest"]
    if sha256_file(public_manifest_path) != inputs["public_manifest_sha256"]:
        raise ValueError("public manifest hash mismatch")
    manifest = _read_json(public_manifest_path)
    public_root = project_root / inputs["public_root"]
    assets: list[AnonymousAsset] = []
    for row in manifest["rows"]:
        if int(row["round"]) != int(inputs["round"]):
            continue
        for label in row["labels"]:
            relative = row["assets"][label]
            path = public_root / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            assets.append(
                AnonymousAsset(
                    presentation_id=row["presentation_id"],
                    role=row["role"],
                    label=label,
                    relative_path=relative,
                    path=path,
                )
            )
    if len(assets) != int(inputs["anonymous_asset_count"]):
        raise ValueError("anonymous asset count mismatch")
    if len({asset.row_id for asset in assets}) != len(assets):
        raise ValueError("duplicate anonymous asset identity")
    return sorted(assets, key=lambda item: item.relative_path)


def _verify_official_source(contract: dict[str, Any], source_root: Path) -> None:
    import subprocess

    head = subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={source_root.resolve().as_posix()}",
            "-C",
            str(source_root),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()
    if head != contract["external_asset"]["repository_commit"]:
        raise ValueError("official source commit mismatch")
    for relative, expected in contract["official_source_sha256"].items():
        if sha256_file(source_root / relative) != expected:
            raise ValueError(f"official source hash mismatch: {relative}")


class CSDRuntime:
    def __init__(
        self,
        contract: dict[str, Any],
        source_root: Path,
        checkpoint_path: Path,
        clip_cache: Path,
        device: str,
    ) -> None:
        import torch

        _verify_official_source(contract, source_root)
        if sha256_file(checkpoint_path) != contract["external_asset"]["model_sha256"]:
            raise ValueError("CSD checkpoint hash mismatch")
        clip_path = clip_cache / "ViT-L-14.pt"
        if sha256_file(clip_path) != contract["external_asset"]["openai_clip_sha256"]:
            raise ValueError("OpenAI CLIP hash mismatch")

        sys.path.insert(0, str(source_root / "models"))
        sys.path.insert(0, str(source_root))
        import clip  # type: ignore[import-not-found]

        original_load = clip.load

        def isolated_load(name: str, *args: Any, **kwargs: Any) -> Any:
            kwargs["download_root"] = str(clip_cache)
            return original_load(name, *args, **kwargs)

        clip.load = isolated_load
        try:
            from CSD.loss_utils import transforms_branch0
            from CSD.model import CSD_CLIP
            from CSD.utils import convert_state_dict

            model = CSD_CLIP("vit_large", "default")
        finally:
            clip.load = original_load

        safe_numpy_globals = [
            (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
            np.dtype,
            type(np.dtype(np.float64)),
        ]
        with torch.serialization.safe_globals(safe_numpy_globals):
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError("unexpected CSD checkpoint structure")
        state = convert_state_dict(checkpoint["model_state_dict"])
        result = model.load_state_dict(state, strict=False)
        missing = list(result.missing_keys)
        unexpected = list(result.unexpected_keys)
        if missing or unexpected:
            raise ValueError(
                f"CSD state mismatch: missing={missing}, unexpected={unexpected}"
            )
        self.model = model.eval().to(device)
        self.preprocess = transforms_branch0
        self.device = device
        self.load_receipt = {
            "weights_only": True,
            "safe_globals": [
                "numpy.core.multiarray.scalar",
                "numpy.dtype",
                "numpy.dtypes.Float64DType",
            ],
            "state_tensor_count": len(state),
            "state_parameter_count": int(sum(value.numel() for value in state.values())),
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "device": device,
        }

    def embed(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        import torch

        tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            _, content, style = self.model(tensor)
        content_np = content[0].detach().cpu().float().numpy().astype(np.float64)
        style_np = style[0].detach().cpu().float().numpy().astype(np.float64)
        return _l2(content_np), _l2(style_np)


def create_score_lock(
    contract_path: Path,
    project_root: Path,
    source_root: Path,
    checkpoint_path: Path,
    clip_cache: Path,
    output_path: Path,
    order: str,
    device: str,
) -> dict[str, Any]:
    import torch

    contract = _read_json(contract_path)
    if order not in {"canonical", "reverse"}:
        raise ValueError("order must be canonical or reverse")
    torch.manual_seed(int(contract["resource_execution"]["random_seed"]))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    runtime = CSDRuntime(contract, source_root, checkpoint_path, clip_cache, device)
    assets = anonymous_assets(contract, project_root)
    if order == "reverse":
        assets = list(reversed(assets))
    rows: list[dict[str, Any]] = []
    for asset in assets:
        with Image.open(asset.path) as opened:
            image = opened.convert("RGB")
            geometry = [image.width, image.height]
            fixed = fixed_descriptors(image)
            content, style = runtime.embed(image)
        rows.append(
            {
                "row_id": asset.row_id,
                "presentation_id": asset.presentation_id,
                "role": asset.role,
                "label": asset.label,
                "relative_path": asset.relative_path,
                "asset_sha256": sha256_file(asset.path),
                "geometry": geometry,
                "features": {
                    "csd_style": style.tolist(),
                    "csd_content": content.tolist(),
                    **fixed,
                },
            }
        )
    lock = {
        "schema": "neuro-film.u5-r2csd0-blind-score-lock.v1",
        "status": "BLIND_SCORE_LOCK_COMPLETE",
        "experiment_id": contract["experiment_id"],
        "order": order,
        "contract_sha256": sha256_file(contract_path),
        "model_sha256": sha256_file(checkpoint_path),
        "clip_sha256": sha256_file(clip_cache / "ViT-L-14.pt"),
        "inventory": {
            "anonymous_asset_count": len(rows),
            "private_mapping_reads": 0,
            "direct_result_reads": 0,
        },
        "load": runtime.load_receipt,
        "rows": rows,
    }
    _write_json(output_path, lock)
    return lock


def create_smoke_receipt(
    contract_path: Path,
    project_root: Path,
    source_root: Path,
    checkpoint_path: Path,
    clip_cache: Path,
    output_path: Path,
    device: str,
) -> dict[str, Any]:
    import torch

    contract = _read_json(contract_path)
    torch.manual_seed(int(contract["resource_execution"]["random_seed"]))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    runtime = CSDRuntime(contract, source_root, checkpoint_path, clip_cache, device)
    asset = anonymous_assets(contract, project_root)[0]
    with Image.open(asset.path) as opened:
        image = opened.convert("RGB")
        content, style = runtime.embed(image)
    receipt = {
        "schema": "neuro-film.u5-r2csd0-mechanics-smoke.v1",
        "status": "PASS_SINGLE_ASSET_MECHANICS_SMOKE",
        "experiment_id": contract["experiment_id"],
        "contract_sha256": sha256_file(contract_path),
        "model_sha256": sha256_file(checkpoint_path),
        "clip_sha256": sha256_file(clip_cache / "ViT-L-14.pt"),
        "row_id": asset.row_id,
        "asset_sha256": sha256_file(asset.path),
        "content_norm": float(np.linalg.norm(content)),
        "style_norm": float(np.linalg.norm(style)),
        "all_finite": bool(np.isfinite(content).all() and np.isfinite(style).all()),
        "private_mapping_reads": 0,
        "direct_result_reads": 0,
        "load": runtime.load_receipt,
    }
    _write_json(output_path, receipt)
    return receipt


def audit_score_locks(
    contract_path: Path, lock_paths: Iterable[Path]
) -> dict[str, Any]:
    contract = _read_json(contract_path)
    locks = [_read_json(path) for path in lock_paths]
    if len(locks) != int(contract["resource_execution"]["fresh_process_count"]):
        raise ValueError("fresh process count mismatch")
    expected_orders = {"canonical", "reverse"}
    if {lock["order"] for lock in locks} != expected_orders:
        raise ValueError("canonical/reverse order mismatch")
    contract_sha = sha256_file(contract_path)
    for lock in locks:
        if lock["status"] != "BLIND_SCORE_LOCK_COMPLETE":
            raise ValueError("incomplete score lock")
        if lock["contract_sha256"] != contract_sha:
            raise ValueError("score lock contract mismatch")
        if lock["model_sha256"] != contract["external_asset"]["model_sha256"]:
            raise ValueError("score lock model mismatch")
        if lock["clip_sha256"] != contract["external_asset"]["openai_clip_sha256"]:
            raise ValueError("score lock CLIP mismatch")
        if lock["inventory"]["private_mapping_reads"] != 0:
            raise ValueError("private mapping read before score lock")
        if lock["inventory"]["direct_result_reads"] != 0:
            raise ValueError("direct result read before score lock")
        if lock["load"]["missing_keys"] or lock["load"]["unexpected_keys"]:
            raise ValueError("non-strict state receipt")
    maps = [{row["row_id"]: row for row in lock["rows"]} for lock in locks]
    if set(maps[0]) != set(maps[1]):
        raise ValueError("score lock row-set mismatch")
    maximum_error = 0.0
    maximum_norm_error = 0.0
    for row_id in sorted(maps[0]):
        first, second = maps[0][row_id], maps[1][row_id]
        for key in ("asset_sha256", "geometry", "relative_path", "role", "label"):
            if first[key] != second[key]:
                raise ValueError(f"row fact mismatch: {row_id}:{key}")
        for feature_key in FEATURE_KEYS:
            left = np.asarray(first["features"][feature_key], dtype=np.float64)
            right = np.asarray(second["features"][feature_key], dtype=np.float64)
            maximum_error = max(maximum_error, float(np.max(np.abs(left - right))))
            maximum_norm_error = max(
                maximum_norm_error,
                abs(float(np.linalg.norm(left)) - 1.0),
                abs(float(np.linalg.norm(right)) - 1.0),
            )
            if not np.isfinite(left).all() or not np.isfinite(right).all():
                raise ValueError("non-finite embedding")
    gates = contract["mechanics_gates"]
    passed = (
        maximum_error <= gates["fresh_process_embedding_max_abs_error_max"]
        and maximum_norm_error <= gates["all_embedding_norm_abs_error_max"]
    )
    return {
        "status": (
            "PASS_MECHANICS_READY_FOR_PRIVATE_AGGREGATION"
            if passed
            else "INVALID_MECHANICS"
        ),
        "maximum_embedding_error": maximum_error,
        "maximum_norm_error": maximum_norm_error,
        "row_count": len(maps[0]),
    }


def _variant_rows(
    contract: dict[str, Any], lock: dict[str, Any], mapping_path: Path
) -> dict[str, dict[str, dict[str, Any]]]:
    expected = contract["consumed_inputs"]["private_mapping_sha256"]
    if sha256_file(mapping_path) != expected:
        raise ValueError("private mapping hash mismatch")
    mapping = _read_json(mapping_path)
    score_rows = {row["row_id"]: row for row in lock["rows"]}
    variants: dict[str, dict[str, dict[str, Any]]] = {}
    for row in mapping["rows"]:
        if int(row["round"]) != int(contract["consumed_inputs"]["round"]):
            continue
        presentation = row["presentation_id"]
        source_id = row["source_id"]
        variants[source_id] = {}
        for label, variant in row["label_to_variant"].items():
            variants[source_id][variant] = score_rows[f"{presentation}:{label}"]
    if len(variants) != int(contract["consumed_inputs"]["source_count"]):
        raise ValueError("mapped source count mismatch")
    if any(set(value) != {"candidate", "identity"} for value in variants.values()):
        raise ValueError("candidate/identity mapping incomplete")
    return variants


def _metrics_for_feature(
    variants: dict[str, dict[str, dict[str, Any]]], feature_key: str, epsilon: float
) -> dict[str, Any]:
    sources = sorted(variants)
    direction_rows: dict[str, list[dict[str, Any]]] = {"candidate": [], "identity": []}
    for direction, collected_rows in direction_rows.items():
        opposite = "identity" if direction == "candidate" else "candidate"
        for reference_source in sources:
            reference_row = variants[reference_source][direction]
            reference = np.asarray(reference_row["features"][feature_key], dtype=np.float64)
            for target_source in sources:
                if target_source == reference_source:
                    continue
                correct_row = variants[target_source][direction]
                wrong_row = variants[target_source][opposite]
                correct = np.asarray(correct_row["features"][feature_key], dtype=np.float64)
                wrong = np.asarray(wrong_row["features"][feature_key], dtype=np.float64)
                margin = float(reference @ correct - reference @ wrong)
                collected_rows.append(
                    {
                        "reference_source": reference_source,
                        "target_source": target_source,
                        "reference_role": reference_row["role"],
                        "target_role": correct_row["role"],
                        "margin": margin,
                        "correct": margin > epsilon,
                        "tie": abs(margin) <= epsilon,
                    }
                )
    result: dict[str, Any] = {"directions": {}}
    combined = []
    for direction, rows in direction_rows.items():
        combined.extend(rows)
        by_reference: dict[str, float] = {}
        for source in sources:
            selected = [row for row in rows if row["reference_source"] == source]
            by_reference[source] = sum(row["correct"] for row in selected) / len(selected)
        role_target = {
            role: sum(row["correct"] for row in rows if row["target_role"] == role)
            for role in sorted({row["target_role"] for row in rows})
        }
        role_reference = {
            role: sum(row["correct"] for row in rows if row["reference_role"] == role)
            for role in sorted({row["reference_role"] for row in rows})
        }
        result["directions"][direction] = {
            "correct": int(sum(row["correct"] for row in rows)),
            "ties": int(sum(row["tie"] for row in rows)),
            "total": len(rows),
            "median_margin": float(np.median([row["margin"] for row in rows])),
            "per_reference_accuracy": by_reference,
            "target_role_correct": role_target,
            "reference_role_correct": role_reference,
        }
    result["combined_correct"] = int(sum(row["correct"] for row in combined))
    result["combined_total"] = len(combined)
    result["combined_median_margin"] = float(
        np.median([row["margin"] for row in combined])
    )
    result["signs"] = [
        {
            "direction": direction,
            "reference_source": row["reference_source"],
            "target_source": row["target_source"],
            "sign": 1 if row["correct"] else (0 if row["tie"] else -1),
        }
        for direction in ("candidate", "identity")
        for row in direction_rows[direction]
    ]
    return result


def aggregate_score_locks(
    contract_path: Path,
    lock_paths: Iterable[Path],
    mapping_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    lock_paths = list(lock_paths)
    mechanics = audit_score_locks(contract_path, lock_paths)
    if mechanics["status"] != "PASS_MECHANICS_READY_FOR_PRIVATE_AGGREGATION":
        raise ValueError("score locks did not pass mechanics")
    contract = _read_json(contract_path)
    locks = [_read_json(path) for path in lock_paths]
    all_metrics: list[dict[str, Any]] = []
    all_signs: list[dict[str, list[dict[str, Any]]]] = []
    for lock in locks:
        variants = _variant_rows(contract, lock, mapping_path)
        metrics = {
            key: _metrics_for_feature(
                variants, key, float(contract["aggregation"]["tie_epsilon"])
            )
            for key in FEATURE_KEYS
        }
        all_metrics.append(metrics)
        all_signs.append({key: metrics[key]["signs"] for key in FEATURE_KEYS})
    signs_exact = all_signs[0] == all_signs[1]
    metrics = all_metrics[0]
    primary = metrics["csd_style"]
    controls = {key: metrics[key] for key in FEATURE_KEYS if key != "csd_style"}
    strongest_control_correct = max(value["combined_correct"] for value in controls.values())
    gates = contract["scientific_gates"]
    failures: list[str] = []
    for direction in ("candidate", "identity"):
        direction_metrics = primary["directions"][direction]
        if direction_metrics["correct"] < gates[f"{direction}_reference_correct_min"]:
            failures.append(f"{direction} reference absolute correct")
        passing_references = sum(
            accuracy >= gates["per_reference_accuracy_min"]
            for accuracy in direction_metrics["per_reference_accuracy"].values()
        )
        if passing_references < gates[f"{direction}_references_passing_min"]:
            failures.append(f"{direction} reference consistency")
        if min(direction_metrics["target_role_correct"].values()) < gates["each_target_role_correct_min"]:
            failures.append(f"{direction} target-role floor")
        if min(direction_metrics["reference_role_correct"].values()) < gates["each_reference_role_correct_min"]:
            failures.append(f"{direction} reference-role floor")
    if primary["combined_correct"] < gates["combined_correct_min"]:
        failures.append("combined absolute correct")
    if primary["combined_correct"] - strongest_control_correct < gates["csd_style_minus_strongest_control_correct_min"]:
        failures.append("strongest-control material gain")
    if primary["combined_median_margin"] <= 0.0:
        failures.append("median signed margin")
    if not signs_exact:
        failures.append("fresh-process pairwise signs")
    status = "PASS_PRIVATE_CROSS_CONTENT_OPERATOR_IDENTIFICATION" if not failures else "FAIL_CLOSED_EXACT_CSD_OPERATOR_OBSERVATION"
    report = {
        "schema": "neuro-film.u5-r2csd0-cross-content-result.v1",
        "experiment_id": contract["experiment_id"],
        "status": status,
        "failed_gates": failures,
        "mechanics": mechanics,
        "pairwise_signs_exact": signs_exact,
        "private_mapping_sha256": sha256_file(mapping_path),
        "score_lock_sha256": [sha256_file(path) for path in lock_paths],
        "metrics": {key: {k: v for k, v in value.items() if k != "signs"} for key, value in metrics.items()},
        "strongest_control_combined_correct": strongest_control_correct,
        "csd_style_minus_strongest_control_correct": primary["combined_correct"] - strongest_control_correct,
        "claim_ceiling": contract["claim_ceiling"],
    }
    _write_json(output_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score")
    score.add_argument("--contract", type=Path, required=True)
    score.add_argument("--project-root", type=Path, required=True)
    score.add_argument("--source-root", type=Path, required=True)
    score.add_argument("--checkpoint", type=Path, required=True)
    score.add_argument("--clip-cache", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--order", choices=("canonical", "reverse"), required=True)
    score.add_argument("--device", default="cuda")
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--contract", type=Path, required=True)
    smoke.add_argument("--project-root", type=Path, required=True)
    smoke.add_argument("--source-root", type=Path, required=True)
    smoke.add_argument("--checkpoint", type=Path, required=True)
    smoke.add_argument("--clip-cache", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--device", default="cuda")
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--contract", type=Path, required=True)
    aggregate.add_argument("--score-lock", type=Path, action="append", required=True)
    aggregate.add_argument("--mapping", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "score":
        result = create_score_lock(
            args.contract,
            args.project_root,
            args.source_root,
            args.checkpoint,
            args.clip_cache,
            args.output,
            args.order,
            args.device,
        )
    elif args.command == "smoke":
        result = create_smoke_receipt(
            args.contract,
            args.project_root,
            args.source_root,
            args.checkpoint,
            args.clip_cache,
            args.output,
            args.device,
        )
    else:
        result = aggregate_score_locks(
            args.contract, args.score_lock, args.mapping, args.output
        )
    print(json.dumps({"status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
