"""Isolated explicit-LUT evaluation for the U5.R2F1 CanonCGT control."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from skimage.color import rgb2lab

from src.eval.global_frontier import new_hard_clipping_fraction, sha256_file
from src.real_film.gold_matrix_transplant import (
    sample_rgb_image,
    style_and_basic_residual,
)


class CanonCGTReferenceError(ValueError):
    """Raised when the frozen F1 contract or evidence is invalid."""


def out_of_range_fraction(values: np.ndarray) -> float:
    array = np.asarray(values)
    if not np.all(np.isfinite(array)):
        raise CanonCGTReferenceError("non-finite explicit-LUT result")
    return float(np.mean((array < 0.0) | (array > 1.0)))


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype="<f4")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_gold_samples(
    root: Path,
    config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    spec = config["input_set"]
    path = _resolve(root, str(spec["manifest"]))
    if sha256_file(path) != str(spec["manifest_sha256"]):
        raise CanonCGTReferenceError("input-set manifest hash mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    frozen = payload.get("frozen_set", payload)
    rows = frozen.get("samples")
    if not isinstance(rows, list):
        raise CanonCGTReferenceError("input set has no sample list")
    wanted = [str(value) for value in spec["sample_ids"]]
    by_id = {str(row["id"]): dict(row) for row in rows}
    if set(by_id).issuperset(wanted) is False:
        raise CanonCGTReferenceError("frozen gold sample is missing")
    samples = {sample_id: by_id[sample_id] for sample_id in wanted}
    if len(samples) != int(spec["expected_samples"]):
        raise CanonCGTReferenceError("gold sample count mismatch")
    for sample_id, row in samples.items():
        if row.get("split") != "gold" or row.get("availability") != "available":
            raise CanonCGTReferenceError(f"non-gold input selected: {sample_id}")
        source = _resolve(root, str(row["source_path"]))
        if sha256_file(source) != str(row["source_sha256"]):
            raise CanonCGTReferenceError(f"source hash mismatch: {sample_id}")
    return samples


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    external = config["external_model"]
    repo = _resolve(root, str(external["local_repository"]))
    if not repo.is_dir():
        raise CanonCGTReferenceError("external repository is absent")
    required = (
        ("license_sha256", repo / "LICENSE"),
        ("config_sha256", _resolve(root, str(external["config_path"]))),
        ("checkpoint_sha256", _resolve(root, str(external["checkpoint_path"]))),
    )
    for hash_key, path in required:
        if sha256_file(path) != str(external[hash_key]):
            raise CanonCGTReferenceError(f"external {hash_key} mismatch")
    references: dict[str, dict[str, Any]] = {}
    for row in config["references"]:
        reference_id = str(row["reference_id"])
        if reference_id in references:
            raise CanonCGTReferenceError("duplicate reference ID")
        path = _resolve(root, str(row["local_path"]))
        if sha256_file(path) != str(row["sha256"]):
            raise CanonCGTReferenceError(f"reference hash mismatch: {reference_id}")
        if row.get("allowed_use") != "internal_A0_runtime_reference_condition_only":
            raise CanonCGTReferenceError(f"reference use drift: {reference_id}")
        references[reference_id] = dict(row)
    if len(references) != int(config["candidate_count"]):
        raise CanonCGTReferenceError("reference count mismatch")
    for spec in config["comparator_manifests"]:
        path = _resolve(root, str(spec["path"]))
        if sha256_file(path) != str(spec["sha256"]):
            raise CanonCGTReferenceError("comparator manifest hash mismatch")
    return {
        "samples": _load_gold_samples(root, config),
        "references": references,
        "repository": repo,
    }


def _image_tensor(path: Path, torch: Any, device: str) -> Any:
    with Image.open(path) as image:
        pixels = (
            np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.float32)
            / np.float32(255.0)
        )
    return (
        torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)))
        .unsqueeze(0)
        .to(device)
    )


def _load_external_model(
    *,
    root: Path,
    config: Mapping[str, Any],
    device: str,
) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    import yaml

    validated = validate_contract(root, config)
    repo = validated["repository"]
    repo_text = str(repo)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    from models.networks.end_to_end_finetuning import CanonCGT_E2E

    external = config["external_model"]
    yaml_path = _resolve(root, str(external["config_path"]))
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    model_config = SimpleNamespace(**payload)
    model = CanonCGT_E2E(model_config)
    checkpoint = torch.load(
        _resolve(root, str(external["checkpoint_path"])),
        map_location="cpu",
        weights_only=True,
    )
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if incompatible.missing_keys:
        raise CanonCGTReferenceError(
            f"missing checkpoint parameters: {incompatible.missing_keys[:3]}"
        )
    if any(not key.startswith("style_centroids.") for key in incompatible.unexpected_keys):
        raise CanonCGTReferenceError(
            f"unexpected forward parameters: {incompatible.unexpected_keys[:3]}"
        )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != int(external["expected_loaded_forward_parameter_count"]):
        raise CanonCGTReferenceError("forward parameter count mismatch")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise CanonCGTReferenceError("CUDA was requested but is unavailable")
    torch.manual_seed(int(config["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config["seed"]))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    model.to(device).eval()
    audit = {
        "parameter_count": parameter_count,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_nonforward_keys": list(incompatible.unexpected_keys),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "torch_version": torch.__version__,
        "device": str(device),
    }
    return torch, model, audit


def render_bank(
    *,
    root: Path,
    config: Mapping[str, Any],
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    validated = validate_contract(root, config)
    torch, model, model_audit = _load_external_model(
        root=root,
        config=config,
        device=device,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    references = {
        reference_id: _image_tensor(
            _resolve(root, str(row["local_path"])),
            torch,
            device,
        )
        for reference_id, row in validated["references"].items()
    }
    records: list[dict[str, Any]] = []
    tolerance = float(
        config["rendering"][
            "require_low_resolution_official_vs_explicit_lut_replay_max_abs_error"
        ]
    )
    with torch.inference_mode():
        for reference_id, reference in references.items():
            reference_row = validated["references"][reference_id]
            candidate_dir = output_dir / reference_id
            lut_dir = candidate_dir / "luts"
            lut_dir.mkdir(parents=True, exist_ok=True)
            for sample_id, sample in validated["samples"].items():
                source_path = _resolve(root, str(sample["source_path"]))
                source = _image_tensor(source_path, torch, device)
                outputs = model(source, reference)
                canonical_lut = outputs["canonicalize_LUT"]
                restyle_lut = outputs["restylize_LUT"]
                canonical_replay = model.TrilinearInterpolation(
                    source,
                    canonical_lut,
                )
                final_replay = model.TrilinearInterpolation(
                    canonical_replay,
                    restyle_lut,
                )
                canonical_error = float(
                    torch.max(
                        torch.abs(canonical_replay - outputs["canonicalized"])
                    ).item()
                )
                final_error = float(
                    torch.max(torch.abs(final_replay - outputs["restyled"])).item()
                )
                if max(canonical_error, final_error) > tolerance:
                    raise CanonCGTReferenceError(
                        f"explicit LUT replay mismatch: {reference_id}/{sample_id}"
                    )
                canonical_np = (
                    canonical_replay[0].permute(1, 2, 0).detach().cpu().numpy()
                )
                final_np = final_replay[0].permute(1, 2, 0).detach().cpu().numpy()
                canonical_lut_np = canonical_lut[0].detach().cpu().numpy()
                restyle_lut_np = restyle_lut[0].detach().cpu().numpy()
                if not all(
                    np.all(np.isfinite(value))
                    for value in (
                        canonical_np,
                        final_np,
                        canonical_lut_np,
                        restyle_lut_np,
                    )
                ):
                    raise CanonCGTReferenceError("non-finite model output")
                canonical_lut_path = lut_dir / f"{sample_id}_canonical.npy"
                restyle_lut_path = lut_dir / f"{sample_id}_restyle.npy"
                np.save(
                    canonical_lut_path,
                    np.asarray(canonical_lut_np, dtype="<f4"),
                    allow_pickle=False,
                )
                np.save(
                    restyle_lut_path,
                    np.asarray(restyle_lut_np, dtype="<f4"),
                    allow_pickle=False,
                )
                output_path = candidate_dir / f"{sample_id}.png"
                pixels = np.rint(np.clip(final_np, 0.0, 1.0) * 255.0).astype(
                    np.uint8
                )
                Image.fromarray(pixels, mode="RGB").save(
                    output_path,
                    format="PNG",
                    compress_level=6,
                )
                records.append(
                    {
                        "reference_id": reference_id,
                        "provenance_bucket": reference_row["provenance_bucket"],
                        "reference_sha256": reference_row["sha256"],
                        "sample_id": sample_id,
                        "source_sha256": sample["source_sha256"],
                        "output": f"{reference_id}/{sample_id}.png",
                        "output_sha256": sha256_file(output_path),
                        "canonical_lut": (
                            f"{reference_id}/luts/{sample_id}_canonical.npy"
                        ),
                        "canonical_lut_file_sha256": sha256_file(
                            canonical_lut_path
                        ),
                        "canonical_lut_array_sha256": _sha256_array(
                            canonical_lut_np
                        ),
                        "canonical_lut_minimum": float(
                            np.min(canonical_lut_np)
                        ),
                        "canonical_lut_maximum": float(
                            np.max(canonical_lut_np)
                        ),
                        "canonical_lut_out_of_range_fraction": (
                            out_of_range_fraction(canonical_lut_np)
                        ),
                        "restyle_lut": (
                            f"{reference_id}/luts/{sample_id}_restyle.npy"
                        ),
                        "restyle_lut_file_sha256": sha256_file(restyle_lut_path),
                        "restyle_lut_array_sha256": _sha256_array(
                            restyle_lut_np
                        ),
                        "restyle_lut_minimum": float(np.min(restyle_lut_np)),
                        "restyle_lut_maximum": float(np.max(restyle_lut_np)),
                        "restyle_lut_out_of_range_fraction": (
                            out_of_range_fraction(restyle_lut_np)
                        ),
                        "canonical_raw_minimum": float(np.min(canonical_np)),
                        "canonical_raw_maximum": float(np.max(canonical_np)),
                        "canonical_raw_out_of_range_fraction": (
                            out_of_range_fraction(canonical_np)
                        ),
                        "final_raw_minimum": float(np.min(final_np)),
                        "final_raw_maximum": float(np.max(final_np)),
                        "final_raw_out_of_range_fraction": (
                            out_of_range_fraction(final_np)
                        ),
                        "canonical_explicit_replay_max_abs_error": (
                            canonical_error
                        ),
                        "final_explicit_replay_max_abs_error": final_error,
                    }
                )
                del source, outputs, canonical_replay, final_replay
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256_file(
            root / "configs" / "u5_r2f1_canoncgt_reference_condition_v1.json"
        ),
        "model": {
            "repository_commit": config["external_model"]["repository_commit"],
            "checkpoint_sha256": config["external_model"]["checkpoint_sha256"],
            **model_audit,
        },
        "candidate_count": len(validated["references"]),
        "sample_count": len(validated["samples"]),
        "records": records,
        "claim_ceiling": config["claim_ceiling"],
    }
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    path = output_dir / "manifest.json"
    path.write_bytes(encoded)
    return {
        "manifest": manifest,
        "manifest_path": path,
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def shortlist_candidates(
    summaries: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    bank_sensitive: bool,
) -> list[str]:
    if not bank_sensitive:
        return []
    by_bucket: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    for reference_id, summary in summaries.items():
        if summary["automatic_survivor"]:
            by_bucket.setdefault(
                str(summary["provenance_bucket"]),
                [],
            ).append((reference_id, summary))
    representatives = []
    for rows in by_bucket.values():
        rows.sort(
            key=lambda item: (
                -float(item[1]["gold_median_non_basic_residual_delta_e76"]),
                -float(item[1]["gold_median_style_delta_e76"]),
                item[0],
            )
        )
        representatives.append(rows[0])
    representatives.sort(
        key=lambda item: (
            -float(item[1]["gold_median_non_basic_residual_delta_e76"]),
            -float(item[1]["gold_median_style_delta_e76"]),
            item[0],
        )
    )
    return [
        reference_id
        for reference_id, _ in representatives[
            : int(config["shortlist"]["maximum_candidates"])
        ]
    ]


def _reference_sensitivity(
    *,
    manifest_path: Path,
    records: Mapping[tuple[str, str], Mapping[str, Any]],
    reference_ids: Sequence[str],
    sample_ids: Sequence[str],
    budget: int,
) -> dict[str, Any]:
    per_sample = []
    all_pairwise = []
    for sample_id in sample_ids:
        pixels = {
            reference_id: sample_rgb_image(
                manifest_path.parent
                / str(records[(reference_id, sample_id)]["output"]),
                budget,
            )
            for reference_id in reference_ids
        }
        pairwise = []
        for index, first in enumerate(reference_ids):
            first_lab = rgb2lab(
                np.clip(pixels[first], 0.0, 1.0).reshape(-1, 1, 3)
            ).reshape(-1, 3)
            for second in reference_ids[index + 1 :]:
                second_lab = rgb2lab(
                    np.clip(pixels[second], 0.0, 1.0).reshape(-1, 1, 3)
                ).reshape(-1, 3)
                value = float(
                    np.median(np.linalg.norm(first_lab - second_lab, axis=1))
                )
                pairwise.append(value)
                all_pairwise.append(value)
        per_sample.append(
            {
                "sample_id": sample_id,
                "median_pairwise_output_delta_e76": float(
                    np.median(pairwise)
                ),
                "minimum_pairwise_output_delta_e76": float(np.min(pairwise)),
                "maximum_pairwise_output_delta_e76": float(np.max(pairwise)),
            }
        )
    return {
        "pair_count": len(all_pairwise),
        "median_pairwise_output_delta_e76": float(np.median(all_pairwise)),
        "minimum_pairwise_output_delta_e76": float(np.min(all_pairwise)),
        "maximum_pairwise_output_delta_e76": float(np.max(all_pairwise)),
        "per_sample": per_sample,
    }


def evaluate_bank(
    *,
    root: Path,
    config: Mapping[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    validated = validate_contract(root, config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference_ids = list(validated["references"])
    sample_ids = list(validated["samples"])
    expected = {
        (reference_id, sample_id)
        for reference_id in reference_ids
        for sample_id in sample_ids
    }
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for row in manifest.get("records", []):
        key = (str(row["reference_id"]), str(row["sample_id"]))
        if key not in expected or key in records:
            raise CanonCGTReferenceError(f"invalid manifest record: {key}")
        records[key] = dict(row)
    if records.keys() != expected:
        raise CanonCGTReferenceError("incomplete render manifest")
    budget = int(config["metrics"]["maximum_pixels_per_image"])
    epsilon = float(config["metrics"]["new_hard_clipping_epsilon"])
    summaries: dict[str, dict[str, Any]] = {}
    for reference_id in reference_ids:
        per_image = []
        for sample_id in sample_ids:
            row = records[(reference_id, sample_id)]
            output_path = manifest_path.parent / str(row["output"])
            if sha256_file(output_path) != str(row["output_sha256"]):
                raise CanonCGTReferenceError("output hash mismatch")
            for key in ("canonical_lut", "restyle_lut"):
                path = manifest_path.parent / str(row[key])
                if sha256_file(path) != str(row[f"{key}_file_sha256"]):
                    raise CanonCGTReferenceError("LUT file hash mismatch")
                array = np.load(path, allow_pickle=False)
                if _sha256_array(array) != str(row[f"{key}_array_sha256"]):
                    raise CanonCGTReferenceError("LUT array hash mismatch")
            source_path = _resolve(
                root,
                str(validated["samples"][sample_id]["source_path"]),
            )
            with Image.open(source_path) as source_image, Image.open(
                output_path
            ) as output_image:
                source_size = ImageOps.exif_transpose(source_image).size
                output = ImageOps.exif_transpose(output_image)
                if output.mode != "RGB" or output.size != source_size:
                    raise CanonCGTReferenceError("output decode/dimension mismatch")
            source_pixels = sample_rgb_image(source_path, budget)
            output_pixels = sample_rgb_image(output_path, budget)
            style, residual = style_and_basic_residual(
                source_pixels,
                output_pixels,
            )
            per_image.append(
                {
                    "sample_id": sample_id,
                    "median_style_delta_e76": style,
                    "median_non_basic_residual_delta_e76": residual,
                    "new_hard_clipping_fraction": new_hard_clipping_fraction(
                        source_pixels,
                        output_pixels,
                        epsilon,
                    ),
                    "final_raw_out_of_range_fraction": float(
                        row["final_raw_out_of_range_fraction"]
                    ),
                    "output_sha256": row["output_sha256"],
                }
            )
        summary = {
            "provenance_bucket": validated["references"][reference_id][
                "provenance_bucket"
            ],
            "gold_median_style_delta_e76": float(
                np.median([row["median_style_delta_e76"] for row in per_image])
            ),
            "gold_median_non_basic_residual_delta_e76": float(
                np.median(
                    [
                        row["median_non_basic_residual_delta_e76"]
                        for row in per_image
                    ]
                )
            ),
            "worst_gold_new_hard_clipping_fraction": float(
                max(row["new_hard_clipping_fraction"] for row in per_image)
            ),
            "worst_gold_raw_final_out_of_range_fraction": float(
                max(
                    row["final_raw_out_of_range_fraction"]
                    for row in per_image
                )
            ),
            "per_image": per_image,
        }
        gates = {
            "style_floor": summary["gold_median_style_delta_e76"]
            >= float(config["metrics"]["minimum_gold_median_style_delta_e76"]),
            "non_basic_floor": summary[
                "gold_median_non_basic_residual_delta_e76"
            ]
            >= float(
                config["metrics"][
                    "minimum_gold_median_non_basic_residual_delta_e76"
                ]
            ),
            "gold_clipping": summary[
                "worst_gold_new_hard_clipping_fraction"
            ]
            <= float(
                config["metrics"][
                    "maximum_worst_gold_new_hard_clipping_fraction"
                ]
            ),
            "raw_final_range": summary[
                "worst_gold_raw_final_out_of_range_fraction"
            ]
            <= float(
                config["metrics"][
                    "maximum_worst_gold_raw_final_out_of_range_fraction"
                ]
            ),
        }
        summary["automatic_gates"] = gates
        summary["automatic_survivor"] = all(gates.values())
        summaries[reference_id] = summary
    sensitivity = _reference_sensitivity(
        manifest_path=manifest_path,
        records=records,
        reference_ids=reference_ids,
        sample_ids=sample_ids,
        budget=budget,
    )
    bank_sensitive = sensitivity["median_pairwise_output_delta_e76"] >= float(
        config["metrics"][
            "minimum_reference_bank_median_pairwise_output_delta_e76"
        ]
    )
    shortlist = shortlist_candidates(
        summaries,
        config,
        bank_sensitive=bank_sensitive,
    )
    return {
        "candidate_count": len(reference_ids),
        "sample_count": len(sample_ids),
        "all_source_reference_output_and_lut_hashes_verified": True,
        "all_outputs_rgb_and_dimension_matched": True,
        "reference_sensitivity": sensitivity,
        "reference_sensitivity_gate_passed": bank_sensitive,
        "candidates": summaries,
        "automatic_survivors": sorted(
            key for key, value in summaries.items() if value["automatic_survivor"]
        ),
        "shortlist": shortlist,
        "automatic_decision": (
            "visual_gate_required"
            if shortlist
            else (
                "reference_conditioning_too_weak"
                if not bank_sensitive
                else "no_automatic_survivor"
            )
        ),
    }


def build_blind_sheets(
    *,
    root: Path,
    config: Mapping[str, Any],
    manifest_path: Path,
    shortlist: Sequence[str],
    output_dir: Path,
) -> dict[str, Any]:
    if not shortlist:
        return {"rounds": [], "mapping": {}, "reason": "empty shortlist"}
    validated = validate_contract(root, config)
    own_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = {
        (str(row["reference_id"]), str(row["sample_id"])): (
            manifest_path.parent / str(row["output"])
        )
        for row in own_manifest["records"]
    }
    for spec in config["comparator_manifests"]:
        comparator_path = _resolve(root, str(spec["path"]))
        comparator = json.loads(comparator_path.read_text(encoding="utf-8"))
        allowed = set(str(value) for value in spec["candidate_ids"])
        for row in comparator["records"]:
            candidate = str(row.get("candidate_id"))
            if candidate not in allowed:
                continue
            relative = Path(str(row["output"]))
            output_path = (
                _resolve(root, str(relative))
                if relative.parts and relative.parts[0] == "outputs"
                else comparator_path.parent / relative
            )
            paths[(candidate, str(row["sample_id"]))] = output_path
    columns = [*config["comparators"], *shortlist]
    gold_ids = list(validated["samples"])
    output_dir.mkdir(parents=True, exist_ok=True)
    private_mapping: dict[str, dict[str, str]] = {}
    round_paths = []
    for round_index in range(1, int(config["visual"]["blind_rounds"]) + 1):
        shuffled = list(columns)
        random.Random(int(config["seed"]) + round_index).shuffle(shuffled)
        labels = {
            candidate: chr(ord("A") + index)
            for index, candidate in enumerate(shuffled)
        }
        private_mapping[f"round_{round_index}"] = {
            label: candidate for candidate, label in labels.items()
        }
        sheet = _contact_sheet(
            root=root,
            samples=validated["samples"],
            sample_ids=gold_ids,
            columns=shuffled,
            labels=labels,
            paths=paths,
        )
        path = output_dir / f"blind_round_{round_index}.png"
        sheet.save(path, format="PNG", compress_level=6)
        round_paths.append(str(path.relative_to(root)).replace("\\", "/"))
    mapping_path = output_dir / "private_mapping.json"
    mapping_path.write_text(
        json.dumps(private_mapping, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "rounds": round_paths,
        "mapping": str(mapping_path.relative_to(root)).replace("\\", "/"),
    }


def _contact_sheet(
    *,
    root: Path,
    samples: Mapping[str, Mapping[str, Any]],
    sample_ids: Sequence[str],
    columns: Sequence[str],
    labels: Mapping[str, str],
    paths: Mapping[tuple[str, str], Path],
) -> Image.Image:
    tile_width, tile_height, header = 300, 210, 26
    canvas = Image.new(
        "RGB",
        ((len(columns) + 1) * tile_width, len(sample_ids) * (tile_height + header)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, sample_id in enumerate(sample_ids):
        y = row_index * (tile_height + header)
        row_paths = [
            _resolve(root, str(samples[sample_id]["source_path"])),
            *(paths[(candidate, sample_id)] for candidate in columns),
        ]
        names = [f"INPUT {sample_id}", *(labels[value] for value in columns)]
        for column_index, (path, name) in enumerate(zip(row_paths, names)):
            with Image.open(path) as image:
                tile = ImageOps.fit(
                    ImageOps.exif_transpose(image).convert("RGB"),
                    (tile_width, tile_height),
                    method=Image.Resampling.LANCZOS,
                )
            x = column_index * tile_width
            canvas.paste(tile, (x, y + header))
            draw.text((x + 5, y + 5), name, fill="black")
    return canvas


__all__ = [
    "CanonCGTReferenceError",
    "build_blind_sheets",
    "evaluate_bank",
    "out_of_range_fraction",
    "render_bank",
    "shortlist_candidates",
    "validate_contract",
]
