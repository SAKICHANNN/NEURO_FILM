#!/usr/bin/env python3
"""Audit official CanonCGT weights on the frozen reference-match matrix."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.color_match import (  # noqa: E402
    aggregate_known_operator_samples,
    canonical_sha256,
    evaluate_known_operator_batch,
)
from src.inference import atomic_write_json  # noqa: E402
from src.preprocess import load_working_image  # noqa: E402
from src.preprocess.raster_decode import (  # noqa: E402
    working_image_to_srgb_float,
)


REPORT_SCHEMA_ID = "neuro-film.reference-match-canoncgt-report.v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _linearize_srgb(encoded: np.ndarray) -> np.ndarray:
    value = np.asarray(encoded, dtype=np.float32)
    return np.where(
        value <= 0.04045,
        value / 12.92,
        ((value + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)


def _paths(payload: dict[str, Any], key: str) -> dict[str, Path]:
    values = payload.get(key)
    if not isinstance(values, dict):
        raise ValueError(f"baseline report {key} is invalid")
    return {
        str(sample_id): Path(str(path))
        for sample_id, path in values.items()
    }


def _verify_paths(
    paths: dict[str, Path],
    hashes: dict[str, str],
    label: str,
) -> None:
    if set(paths) != set(hashes):
        raise ValueError(f"{label} path/hash IDs differ")
    for sample_id, path in paths.items():
        if not path.is_file() or sha256_file(path) != hashes[sample_id]:
            raise ValueError(
                f"{label} file does not match frozen hash: {sample_id}"
            )


def _validate_external(
    config: dict[str, Any],
    external_root: Path,
    weights: Path,
) -> dict[str, str]:
    expected = config["external"]
    if (
        subprocess.run(
            ["git", "-C", str(external_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        != expected["repository_commit"]
    ):
        raise ValueError("CanonCGT repository commit mismatch")
    license_path = external_root / "LICENSE"
    if sha256_file(license_path) != expected["license_sha256"]:
        raise ValueError("CanonCGT license hash mismatch")
    if sha256_file(weights) != expected["weight_sha256"]:
        raise ValueError("CanonCGT weight hash mismatch")
    return {
        "repository_commit": expected["repository_commit"],
        "license_sha256": sha256_file(license_path),
        "weight_sha256": sha256_file(weights),
    }


def _load_model(
    config: dict[str, Any],
    external_root: Path,
    weights: Path,
    device: str,
) -> tuple[Any, Any]:
    import torch
    import yaml

    sys.path.insert(0, str(external_root))
    try:
        from models.networks.SSL_training import CanonCGT_SSL
    finally:
        sys.path.pop(0)

    external_config = yaml.safe_load(
        (
            external_root / config["external"]["configuration"]
        ).read_text(encoding="utf-8")
    )
    namespace = SimpleNamespace(
        **external_config,
        yaml="Stage3_SSL_training_Flickr2K_PPR10K_LSDIR",
    )
    model = CanonCGT_SSL(namespace)
    checkpoint = torch.load(
        weights,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    if parameter_count != int(config["external"]["parameter_count"]):
        raise ValueError("CanonCGT parameter count mismatch")
    model.to(torch.device(device)).eval()
    return model, torch


def _tensor(torch: Any, working: Any, device: str) -> Any:
    encoded = working_image_to_srgb_float(working)
    return (
        torch.from_numpy(encoded)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(torch.device(device))
    )


def _rendered_working(
    source: Any,
    encoded: np.ndarray,
) -> tuple[Any, float]:
    preclip = float(
        np.mean(
            np.any((encoded < 0.0) | (encoded > 1.0), axis=-1),
            dtype=np.float64,
        )
    )
    pixels = _linearize_srgb(np.clip(encoded, 0.0, 1.0))
    return replace(source, pixels=pixels), preclip


def _mode_summary(
    rows: list[Any],
    preclip: list[float],
    gates: dict[str, Any],
) -> dict[str, Any]:
    aggregate = aggregate_known_operator_samples(rows)
    improvements = np.asarray(
        [row.median_improvement_fraction for row in rows],
        dtype=np.float64,
    )
    summary = {
        "known_operator": asdict(aggregate),
        "improved_fraction": float(np.mean(improvements > 0.0)),
        "median_improvement_fraction": float(np.median(improvements)),
        "worst_improvement_fraction": float(np.min(improvements)),
        "maximum_new_boundary_fraction": float(
            max(row.candidate_new_boundary_fraction for row in rows)
        ),
        "maximum_preclip_out_of_gamut_fraction": float(max(preclip)),
    }
    summary["gates"] = {
        "improved_fraction": summary["improved_fraction"]
        >= float(gates["minimum_improved_cross_content_fraction"]),
        "median_improvement": summary["median_improvement_fraction"]
        >= float(gates["minimum_median_improvement_fraction"]),
        "worst_improvement": summary["worst_improvement_fraction"]
        >= float(gates["minimum_worst_improvement_fraction"]),
        "new_boundary": summary["maximum_new_boundary_fraction"]
        <= float(gates["maximum_new_boundary_fraction"]),
        "preclip_out_of_gamut": (
            summary["maximum_preclip_out_of_gamut_fraction"]
            <= float(gates["maximum_preclip_out_of_gamut_fraction"])
        ),
    }
    summary["passes"] = all(summary["gates"].values())
    return summary


def evaluate(
    config: dict[str, Any],
    baseline: dict[str, Any],
    model: Any,
    torch: Any,
    device: str,
) -> dict[str, Any]:
    sample_ids = tuple(str(value) for value in baseline["sample_ids"])
    reference_paths = _paths(baseline, "reference_paths")
    source_paths = _paths(baseline, "source_paths")
    target_paths = _paths(baseline, "target_paths")
    _verify_paths(
        reference_paths,
        dict(baseline["reference_file_sha256"]),
        "reference",
    )
    _verify_paths(
        source_paths,
        dict(baseline["source_file_sha256"]),
        "source",
    )
    _verify_paths(
        target_paths,
        dict(baseline["target_file_sha256"]),
        "target",
    )
    sources = {
        sample_id: load_working_image(source_paths[sample_id])
        for sample_id in sample_ids
    }
    references = {
        sample_id: load_working_image(reference_paths[sample_id])
        for sample_id in sample_ids
    }
    targets = {
        sample_id: load_working_image(target_paths[sample_id])
        for sample_id in sample_ids
    }
    reference_tensors = {
        sample_id: _tensor(torch, references[sample_id], device)
        for sample_id in sample_ids
    }
    modes: dict[str, dict[str, Any]] = {}
    published_rows: list[Any] = []
    published_preclip: list[float] = []
    restyler_luts: dict[str, list[np.ndarray]] = {
        sample_id: [] for sample_id in sample_ids
    }
    with torch.inference_mode():
        for reference_id in sample_ids:
            for source_id in sample_ids:
                if source_id == reference_id:
                    continue
                output = model(
                    _tensor(torch, sources[source_id], device),
                    reference_tensors[reference_id],
                )
                encoded = (
                    output["restyled"][0]
                    .permute(1, 2, 0)
                    .detach()
                    .cpu()
                    .numpy()
                )
                candidate, preclip = _rendered_working(
                    sources[source_id],
                    encoded,
                )
                row = evaluate_known_operator_batch(
                    [sources[source_id]],
                    [targets[source_id]],
                    [candidate],
                    sample_ids=[f"{reference_id}->{source_id}"],
                ).samples[0]
                published_rows.append(row)
                published_preclip.append(preclip)
                restyler_luts[reference_id].append(
                    output["restylize_LUT"][0].detach().cpu().numpy()
                )
                del output
                gc.collect()
    variation: dict[str, dict[str, float]] = {}
    for reference_id, luts in restyler_luts.items():
        distances = [
            float(np.sqrt(np.mean((luts[first] - luts[second]) ** 2)))
            for first in range(len(luts))
            for second in range(first)
        ]
        variation[reference_id] = {
            "pairwise_rmse_median": float(np.median(distances)),
            "pairwise_rmse_maximum": float(np.max(distances)),
        }
    maximum_lut_variation = max(
        row["pairwise_rmse_maximum"] for row in variation.values()
    )
    published = _mode_summary(
        published_rows,
        published_preclip,
        config["gates"],
    )
    published["restyler_lut_source_variation"] = variation
    published["maximum_restyler_lut_source_rmse"] = maximum_lut_variation
    published["gates"]["restyler_lut_source_invariance"] = (
        maximum_lut_variation
        <= float(
            config["gates"][
                "maximum_source_conditioned_restyler_lut_rmse"
            ]
        )
    )
    published["passes"] = all(published["gates"].values())
    modes["published-source-conditioned"] = published

    fixed_rows: list[Any] = []
    fixed_preclip: list[float] = []
    reference_reconstruction: dict[str, dict[str, float]] = {}
    with torch.inference_mode():
        for reference_id in sample_ids:
            reference = reference_tensors[reference_id]
            target_token = model.Embedding_Net(reference)
            canonical_reference = model.Canonicalizer(
                reference,
                condition=target_token,
            )["result"]
            fixed_lut = model.Restyler(
                canonical_reference,
                condition=target_token,
            )["LUT"]
            reconstructed = model.TrilinearInterpolation(
                canonical_reference,
                fixed_lut,
            )
            difference = torch.abs(reconstructed - reference)
            reference_reconstruction[reference_id] = {
                "mean_absolute_error": float(difference.mean()),
                "maximum_absolute_error": float(difference.max()),
            }
            for source_id in sample_ids:
                if source_id == reference_id:
                    continue
                source = _tensor(torch, sources[source_id], device)
                source_token = model.Embedding_Net(source)
                canonical_source = model.Canonicalizer(
                    source,
                    condition=source_token,
                )["result"]
                encoded = (
                    model.TrilinearInterpolation(canonical_source, fixed_lut)[
                        0
                    ]
                    .permute(1, 2, 0)
                    .detach()
                    .cpu()
                    .numpy()
                )
                candidate, preclip = _rendered_working(
                    sources[source_id],
                    encoded,
                )
                row = evaluate_known_operator_batch(
                    [sources[source_id]],
                    [targets[source_id]],
                    [candidate],
                    sample_ids=[f"{reference_id}->{source_id}"],
                ).samples[0]
                fixed_rows.append(row)
                fixed_preclip.append(preclip)
                gc.collect()
    fixed = _mode_summary(fixed_rows, fixed_preclip, config["gates"])
    fixed["reference_reconstruction"] = reference_reconstruction
    fixed["target_lut_fixed_across_sources"] = True
    modes["fixed-reference-pivot-lut"] = fixed
    return modes


def main() -> int:
    args = _parser().parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["execution"] != {
        "official_code_unmodified": True,
        "official_weights_unmodified": True,
        "evaluation_only": True,
        "targets_used_only_after_render": True,
        "product_integration_allowed": False,
        "output_rgb_is_not_retained": True,
    }:
        raise ValueError("CanonCGT execution boundary mismatch")
    external = _validate_external(
        config,
        args.external_root.resolve(),
        args.weights.resolve(),
    )
    baseline_path = ROOT / str(config["baseline_report"])
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    model, torch = _load_model(
        config,
        args.external_root.resolve(),
        args.weights.resolve(),
        args.device,
    )
    modes = evaluate(config, baseline, model, torch, args.device)
    payload = {
        "schema_id": REPORT_SCHEMA_ID,
        "report_id": "",
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256_file(args.config),
        "baseline_report_sha256": sha256_file(baseline_path),
        "external": external,
        "device": args.device,
        "modes": modes,
        "promotion_open": any(mode["passes"] for mode in modes.values()),
        "claim_ceiling": config["claim_ceiling"],
    }
    payload["report_id"] = canonical_sha256(
        {
            key: value
            for key, value in payload.items()
            if key != "report_id"
        }
    )
    report_sha256 = atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "report_id": payload["report_id"],
                "report_sha256": report_sha256,
                "promotion_open": payload["promotion_open"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
