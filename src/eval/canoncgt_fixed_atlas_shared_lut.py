"""Reference-only fixed-atlas CanonCGT shared-LUT D0."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from skimage.color import rgb2lab

from src.eval.canoncgt_reference_condition import (
    CanonCGTReferenceError,
    _load_gold_samples,
    _resolve,
    _sha256_array,
    out_of_range_fraction,
)
from src.eval.global_frontier import new_hard_clipping_fraction, sha256_file
from src.real_film.gold_matrix_transplant import (
    sample_rgb_image,
    style_and_basic_residual,
)


def load_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    parent_spec = config["parent_contract"]
    parent_path = _resolve(root, str(parent_spec["path"]))
    if sha256_file(parent_path) != str(parent_spec["sha256"]):
        raise CanonCGTReferenceError("parent contract hash mismatch")
    return json.loads(parent_path.read_text(encoding="utf-8"))


def make_lattice_atlas(
    *,
    atlas_id: str,
    size: int,
    bins: int,
    multiplier: int,
) -> np.ndarray:
    if sorted(atlas_id) != ["b", "g", "r"] or len(atlas_id) != 3:
        raise CanonCGTReferenceError("invalid atlas channel permutation")
    if size < bins or bins < 2:
        raise CanonCGTReferenceError("invalid atlas geometry")
    count = bins**3
    indices = (np.arange(size * size, dtype=np.int64) * multiplier) % count
    rgb = np.stack(
        [
            indices // (bins * bins),
            (indices // bins) % bins,
            indices % bins,
        ],
        axis=1,
    ).astype(np.float32) / np.float32(bins - 1)
    lookup = {"r": 0, "g": 1, "b": 2}
    atlas = rgb[:, [lookup[channel] for channel in atlas_id]].reshape(
        size, size, 3
    )
    return np.ascontiguousarray(atlas, dtype=np.float32)


def aggregate_atlas_luts(
    luts: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, float]]:
    if len(luts) < 2:
        raise CanonCGTReferenceError("at least two atlas LUTs are required")
    ordered = [(key, np.asarray(luts[key], dtype=np.float32)) for key in sorted(luts)]
    shape = ordered[0][1].shape
    if any(value.shape != shape for _, value in ordered):
        raise CanonCGTReferenceError("atlas LUT shape mismatch")
    if any(not np.all(np.isfinite(value)) for _, value in ordered):
        raise CanonCGTReferenceError("non-finite atlas LUT")
    pairwise: list[float] = []
    for index, (_, left) in enumerate(ordered):
        for _, right in ordered[index + 1 :]:
            pairwise.append(
                float(np.sqrt(np.mean((left.astype(np.float64) - right) ** 2)))
            )
    mean = np.mean(
        np.stack([value.astype(np.float64) for _, value in ordered]),
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)
    return np.ascontiguousarray(mean), {
        "pairwise_count": len(pairwise),
        "pairwise_rmse_median": float(np.median(pairwise)),
        "pairwise_rmse_p95": float(np.percentile(pairwise, 95)),
        "pairwise_rmse_maximum": float(np.max(pairwise)),
    }


def _reference_only_validation(
    root: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], Path]:
    parent = load_contract(root, config)
    external = parent["external_model"]
    repo = _resolve(root, str(external["local_repository"]))
    required = (
        (repo / "LICENSE", external["license_sha256"]),
        (_resolve(root, str(external["config_path"])), external["config_sha256"]),
        (
            _resolve(root, str(external["checkpoint_path"])),
            external["checkpoint_sha256"],
        ),
    )
    for path, expected in required:
        if sha256_file(path) != str(expected):
            raise CanonCGTReferenceError("external asset hash mismatch")
    for row in parent["references"]:
        path = _resolve(root, str(row["local_path"]))
        if sha256_file(path) != str(row["sha256"]):
            raise CanonCGTReferenceError("reference hash mismatch")
    return parent, repo


def _load_model_reference_only(
    root: Path, config: Mapping[str, Any], device: str
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    import torch
    import yaml

    parent, repo = _reference_only_validation(root, config)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from models.networks.end_to_end_finetuning import CanonCGT_E2E

    external = parent["external_model"]
    payload = yaml.safe_load(
        _resolve(root, str(external["config_path"])).read_text(encoding="utf-8")
    )
    model = CanonCGT_E2E(SimpleNamespace(**payload))
    checkpoint = torch.load(
        _resolve(root, str(external["checkpoint_path"])),
        map_location="cpu",
        weights_only=True,
    )
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if incompatible.missing_keys or any(
        not key.startswith("style_centroids.")
        for key in incompatible.unexpected_keys
    ):
        raise CanonCGTReferenceError("checkpoint compatibility failure")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise CanonCGTReferenceError("CUDA requested but unavailable")
    seed = 20260820
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    model.to(device).eval()
    return torch, model, parent, {
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "torch_version": torch.__version__,
        "device": device,
    }


def _tensor_from_array(values: np.ndarray, torch: Any, device: str) -> Any:
    return torch.from_numpy(values.transpose(2, 0, 1)).unsqueeze(0).to(device)


def _tensor_from_image(path: Path, torch: Any, device: str) -> Any:
    with Image.open(path) as image:
        values = np.asarray(
            ImageOps.exif_transpose(image).convert("RGB"), dtype=np.float32
        ) / np.float32(255.0)
    return _tensor_from_array(np.ascontiguousarray(values), torch, device)


def build_lut_bank(
    *, root: Path, config: Mapping[str, Any], output_dir: Path, device: str,
    reverse: bool = False,
) -> dict[str, Any]:
    torch, model, parent, model_audit = _load_model_reference_only(
        root, config, device
    )
    spec = config["build_contract"]
    atlas_ids = list(spec["atlas_ids"])
    reference_rows = list(parent["references"])
    if reverse:
        atlas_ids.reverse()
        reference_rows.reverse()
    atlases = {
        atlas_id: make_lattice_atlas(
            atlas_id=atlas_id,
            size=int(spec["atlas_size"]),
            bins=int(spec["atlas_lattice_bins"]),
            multiplier=int(spec["atlas_index_multiplier"]),
        )
        for atlas_id in atlas_ids
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for row in reference_rows:
            reference_id = str(row["reference_id"])
            reference = _tensor_from_image(
                _resolve(root, str(row["local_path"])), torch, device
            )
            condition = model.Embedding_Net(reference)
            luts: dict[str, np.ndarray] = {}
            for atlas_id in atlas_ids:
                atlas = _tensor_from_array(atlases[atlas_id], torch, device)
                result = model.Restyler(atlas, condition=condition)
                luts[atlas_id] = np.ascontiguousarray(
                    result["LUT"][0].detach().cpu().numpy(), dtype=np.float32
                )
            shared, drift = aggregate_atlas_luts(luts)
            lut_path = output_dir / f"{reference_id}.npy"
            np.save(lut_path, np.asarray(shared, dtype="<f4"), allow_pickle=False)
            records.append(
                {
                    "reference_id": reference_id,
                    "reference_sha256": row["sha256"],
                    "provenance_bucket": row["provenance_bucket"],
                    "lut": lut_path.name,
                    "lut_file_sha256": sha256_file(lut_path),
                    "lut_array_sha256": _sha256_array(shared),
                    "lut_minimum": float(np.min(shared)),
                    "lut_maximum": float(np.max(shared)),
                    "lut_out_of_range_fraction": out_of_range_fraction(shared),
                    "atlas_lut_drift": drift,
                }
            )
    records.sort(key=lambda row: str(row["reference_id"]))
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "phase": "build",
        "application_source_file_reads": 0,
        "application_source_pixel_decodes": 0,
        "canonicalizer_executions": 0,
        "reference_count": len(records),
        "atlas_ids_aggregated_in_order": sorted(atlas_ids),
        "model": model_audit,
        "records": records,
        "claim_ceiling": config["claim_ceiling"],
    }
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    path = output_dir / "build_manifest.json"
    path.write_bytes(encoded)
    return {"path": path, "sha256": hashlib.sha256(encoded).hexdigest(), "manifest": manifest}


def apply_lut_bank(
    *, root: Path, config: Mapping[str, Any], build_manifest_path: Path,
    output_dir: Path, device: str, reverse: bool = False,
) -> dict[str, Any]:
    torch, model, parent, model_audit = _load_model_reference_only(
        root, config, device
    )
    build = json.loads(build_manifest_path.read_text(encoding="utf-8"))
    expected_references = {
        str(row["reference_id"]): str(row["sha256"])
        for row in parent["references"]
    }
    build_references = {
        str(row["reference_id"]): str(row["reference_sha256"])
        for row in build.get("records", [])
    }
    if (
        build.get("experiment_id") != config["experiment_id"]
        or build.get("phase") != "build"
        or build.get("application_source_file_reads") != 0
        or build.get("application_source_pixel_decodes") != 0
        or build.get("canonicalizer_executions") != 0
        or build_references != expected_references
    ):
        raise CanonCGTReferenceError("invalid build manifest")
    samples = _load_gold_samples(root, parent)
    build_rows = list(build["records"])
    sample_rows = list(samples.items())
    if reverse:
        build_rows.reverse()
        sample_rows.reverse()
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        frozen_luts: list[tuple[Mapping[str, Any], Any]] = []
        for lut_row in build_rows:
            lut_path = build_manifest_path.parent / str(lut_row["lut"])
            if sha256_file(lut_path) != str(lut_row["lut_file_sha256"]):
                raise CanonCGTReferenceError("frozen LUT hash mismatch")
            lut_np = np.load(lut_path, allow_pickle=False).astype(np.float32)
            if _sha256_array(lut_np) != str(lut_row["lut_array_sha256"]):
                raise CanonCGTReferenceError("frozen LUT array mismatch")
            frozen_luts.append(
                (lut_row, torch.from_numpy(lut_np).unsqueeze(0).to(device))
            )
        for sample_id, sample in sample_rows:
            source_path = _resolve(root, str(sample["source_path"]))
            source = _tensor_from_image(source_path, torch, device)
            for lut_row, lut in frozen_luts:
                reference_id = str(lut_row["reference_id"])
                candidate_dir = output_dir / reference_id
                candidate_dir.mkdir(parents=True, exist_ok=True)
                raw = model.TrilinearInterpolation(source, lut)
                raw_np = raw[0].permute(1, 2, 0).detach().cpu().numpy()
                if not np.all(np.isfinite(raw_np)):
                    raise CanonCGTReferenceError("non-finite shared-LUT output")
                output_path = candidate_dir / f"{sample_id}.png"
                pixels = np.rint(np.clip(raw_np, 0.0, 1.0) * 255.0).astype(np.uint8)
                Image.fromarray(pixels, mode="RGB").save(
                    output_path, format="PNG", compress_level=6
                )
                records.append(
                    {
                        "reference_id": reference_id,
                        "sample_id": sample_id,
                        "source_sha256": sample["source_sha256"],
                        "lut_array_sha256": lut_row["lut_array_sha256"],
                        "output": f"{reference_id}/{sample_id}.png",
                        "output_sha256": sha256_file(output_path),
                        "raw_out_of_range_fraction": out_of_range_fraction(raw_np),
                    }
                )
    records.sort(key=lambda row: (str(row["reference_id"]), str(row["sample_id"])))
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "phase": "apply",
        "build_manifest_sha256": sha256_file(build_manifest_path),
        "application_source_file_reads_after_build_freeze": len(samples),
        "application_source_pixel_decodes_after_build_freeze": len(samples),
        "per_source_parameter_changes": 0,
        "model": model_audit,
        "records": records,
        "claim_ceiling": config["claim_ceiling"],
    }
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    path = output_dir / "apply_manifest.json"
    path.write_bytes(encoded)
    return {"path": path, "sha256": hashlib.sha256(encoded).hexdigest(), "manifest": manifest}


def _pairwise_reference_sensitivity(
    apply_path: Path, records: Mapping[tuple[str, str], Mapping[str, Any]],
    reference_ids: Sequence[str], sample_ids: Sequence[str], budget: int,
) -> dict[str, float]:
    values: list[float] = []
    for sample_id in sample_ids:
        labs = []
        for reference_id in reference_ids:
            row = records[(reference_id, sample_id)]
            rgb = sample_rgb_image(apply_path.parent / str(row["output"]), budget)
            labs.append(rgb2lab(rgb.reshape(-1, 1, 3)).reshape(-1, 3))
        for index, left in enumerate(labs):
            for right in labs[index + 1 :]:
                values.append(float(np.median(np.linalg.norm(left - right, axis=1))))
    return {
        "pair_count": len(values),
        "median_pairwise_output_delta_e76": float(np.median(values)),
        "minimum_pairwise_output_delta_e76": float(np.min(values)),
        "maximum_pairwise_output_delta_e76": float(np.max(values)),
    }


def evaluate(
    *, root: Path, config: Mapping[str, Any], build_manifest_path: Path,
    apply_manifest_path: Path,
) -> dict[str, Any]:
    parent = load_contract(root, config)
    samples = _load_gold_samples(root, parent)
    references = {str(row["reference_id"]): row for row in parent["references"]}
    build = json.loads(build_manifest_path.read_text(encoding="utf-8"))
    apply = json.loads(apply_manifest_path.read_text(encoding="utf-8"))
    if (
        build.get("experiment_id") != config["experiment_id"]
        or build.get("phase") != "build"
        or apply.get("experiment_id") != config["experiment_id"]
        or apply.get("phase") != "apply"
        or apply.get("build_manifest_sha256") != sha256_file(build_manifest_path)
        or apply.get("application_source_file_reads_after_build_freeze")
        != len(samples)
        or apply.get("application_source_pixel_decodes_after_build_freeze")
        != len(samples)
        or apply.get("per_source_parameter_changes") != 0
    ):
        raise CanonCGTReferenceError("invalid build/apply manifest chain")
    expected = {(rid, sid) for rid in references for sid in samples}
    records = {
        (str(row["reference_id"]), str(row["sample_id"])): row
        for row in apply["records"]
    }
    if set(records) != expected:
        raise CanonCGTReferenceError("incomplete apply manifest")
    build_rows = {str(row["reference_id"]): row for row in build["records"]}
    metrics = config["metrics"]
    budget = int(metrics["maximum_pixels_per_image"])
    epsilon = float(metrics["new_hard_clipping_epsilon"])
    summaries: dict[str, Any] = {}
    for reference_id in references:
        per_image = []
        expected_lut_sha = str(build_rows[reference_id]["lut_array_sha256"])
        for sample_id, sample in samples.items():
            row = records[(reference_id, sample_id)]
            if row["lut_array_sha256"] != expected_lut_sha:
                raise CanonCGTReferenceError("per-source LUT drift")
            output_path = apply_manifest_path.parent / str(row["output"])
            if sha256_file(output_path) != str(row["output_sha256"]):
                raise CanonCGTReferenceError("output hash mismatch")
            source_path = _resolve(root, str(sample["source_path"]))
            source_rgb = sample_rgb_image(source_path, budget)
            output_rgb = sample_rgb_image(output_path, budget)
            style, residual = style_and_basic_residual(source_rgb, output_rgb)
            per_image.append(
                {
                    "sample_id": sample_id,
                    "style": style,
                    "non_basic": residual,
                    "new_clipping": new_hard_clipping_fraction(source_rgb, output_rgb, epsilon),
                    "raw_out_of_range": float(row["raw_out_of_range_fraction"]),
                }
            )
        summary = {
            "median_style": float(np.median([row["style"] for row in per_image])),
            "median_non_basic": float(np.median([row["non_basic"] for row in per_image])),
            "worst_new_clipping": float(max(row["new_clipping"] for row in per_image)),
            "worst_raw_out_of_range": float(max(row["raw_out_of_range"] for row in per_image)),
            "atlas_lut_drift": build_rows[reference_id]["atlas_lut_drift"],
            "per_image": per_image,
        }
        gates = {
            "atlas_stability": summary["atlas_lut_drift"]["pairwise_rmse_p95"] <= float(metrics["maximum_atlas_lut_pairwise_rmse_p95"]),
            "style": summary["median_style"] >= float(metrics["minimum_gold_median_style_delta_e76"]),
            "non_basic": summary["median_non_basic"] >= float(metrics["minimum_gold_median_non_basic_residual_delta_e76"]),
            "clipping": summary["worst_new_clipping"] <= float(metrics["maximum_worst_gold_new_hard_clipping_fraction"]),
            "raw_range": summary["worst_raw_out_of_range"] <= float(metrics["maximum_worst_gold_raw_final_out_of_range_fraction"]),
        }
        summary["gates"] = gates
        summary["survivor"] = all(gates.values())
        summaries[reference_id] = summary
    sensitivity = _pairwise_reference_sensitivity(
        apply_manifest_path, records, list(references), list(samples), budget
    )
    sensitivity_pass = sensitivity["median_pairwise_output_delta_e76"] >= float(metrics["minimum_reference_bank_median_pairwise_output_delta_e76"])
    survivors = sorted(key for key, value in summaries.items() if value["survivor"])
    pass_all = sensitivity_pass and len(survivors) >= int(metrics["minimum_automatic_survivors"])
    return {
        "decision": "PASS_AUTOMATIC_OPEN_BLIND_REVIEW" if pass_all else "FAIL_CLOSED_FIXED_ATLAS_SHARED_LUT",
        "source_independent_shared_lut": True,
        "build_application_source_reads": build["application_source_file_reads"],
        "per_source_parameter_changes": apply["per_source_parameter_changes"],
        "reference_sensitivity": sensitivity,
        "reference_sensitivity_gate": sensitivity_pass,
        "automatic_survivors": survivors,
        "candidates": summaries,
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "aggregate_atlas_luts",
    "apply_lut_bank",
    "build_lut_bank",
    "evaluate",
    "load_contract",
    "make_lattice_atlas",
]
