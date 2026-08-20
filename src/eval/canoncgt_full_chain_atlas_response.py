"""Reference-only CanonCGT full-chain fixed-atlas response audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.canoncgt_fixed_atlas_shared_lut import (
    _load_model_reference_only,
    _tensor_from_array,
    _tensor_from_image,
    make_lattice_atlas,
)
from src.eval.canoncgt_reference_condition import CanonCGTReferenceError, _resolve
from src.eval.global_frontier import sha256_file


def _verify_binding(root: Path, binding: Mapping[str, Any]) -> None:
    path = _resolve(root, str(binding["path"]))
    if sha256_file(path) != str(binding["sha256"]):
        raise CanonCGTReferenceError(f"binding mismatch: {binding['path']}")


def _stats(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float32)
    finite = bool(np.all(np.isfinite(array)))
    digest = hashlib.sha256(array.tobytes(order="C")).hexdigest()
    if not finite:
        return {
            "finite": False,
            "minimum": None,
            "maximum": None,
            "out_of_range_fraction": None,
            "sha256": digest,
        }
    outside = (array < 0.0) | (array > 1.0)
    return {
        "finite": True,
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "out_of_range_fraction": float(np.mean(outside)),
        "sha256": digest,
    }


def summarize_rows(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    spec = config["prescore_gates"]
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (str(row["reference_id"]), str(row["atlas_id"])),
    )
    reference_ids = sorted({str(row["reference_id"]) for row in ordered})
    counts = {
        reference_id: sum(str(row["reference_id"]) == reference_id for row in ordered)
        for reference_id in reference_ids
    }
    finite = all(
        bool(row["canonical"]["finite"]) and bool(row["response"]["finite"])
        for row in ordered
    )
    response_bounds = finite and all(
        float(row["response"]["minimum"]) >= float(spec["minimum_response_value"])
        and float(row["response"]["maximum"]) <= float(spec["maximum_response_value"])
        and float(row["response"]["out_of_range_fraction"])
        <= float(spec["maximum_response_out_of_range_fraction"])
        for row in ordered
    )
    canonical_support = finite and all(
        float(row["canonical"]["out_of_range_fraction"])
        <= float(spec["maximum_canonical_out_of_range_fraction"])
        for row in ordered
    )
    sensitivity = [
        float(row["reference_pairwise_response_rmse_median"]) for row in ordered
    ]
    sensitivity_median = float(np.median(sensitivity)) if sensitivity else 0.0
    gates = {
        "complete_inventory": len(reference_ids)
        == int(spec["required_reference_count"])
        and all(
            count == int(spec["required_atlas_rows_per_reference"])
            for count in counts.values()
        ),
        "all_values_finite": finite,
        "response_intrinsically_cube_bounded": response_bounds,
        "canonical_support": canonical_support,
        "reference_response_sensitivity": sensitivity_median
        >= float(spec["minimum_reference_response_pairwise_rmse_median"]),
    }
    response_outside = [
        float(row["response"]["out_of_range_fraction"])
        for row in ordered
        if row["response"]["out_of_range_fraction"] is not None
    ]
    canonical_outside = [
        float(row["canonical"]["out_of_range_fraction"])
        for row in ordered
        if row["canonical"]["out_of_range_fraction"] is not None
    ]
    return {
        "row_count": len(ordered),
        "reference_count": len(reference_ids),
        "atlas_rows_per_reference": counts,
        "reference_pairwise_response_rmse_median": sensitivity_median,
        "maximum_response_out_of_range_fraction": (
            max(response_outside) if response_outside else None
        ),
        "maximum_canonical_out_of_range_fraction": (
            max(canonical_outside) if canonical_outside else None
        ),
        "gates": gates,
        "decision": (
            "PASS_FULL_CHAIN_ATLAS_OBSERVATION_PRESCORE"
            if all(gates.values())
            else "FAIL_CLOSED_FULL_CHAIN_ATLAS_OBSERVATION_PRESCORE"
        ),
    }


def run_audit(
    *, root: Path, config: Mapping[str, Any], device: str, reverse: bool
) -> dict[str, Any]:
    _verify_binding(root, config["parent_contract"])
    _verify_binding(root, config["fixed_atlas_control"])
    _verify_binding(
        root,
        {
            "path": config["fixed_atlas_control"]["core_path"],
            "sha256": config["fixed_atlas_control"]["core_sha256"],
        },
    )
    torch, model, parent, model_audit = _load_model_reference_only(root, config, device)
    build = config["build"]
    atlas_ids = list(build["atlas_ids"])
    references = list(parent["references"])
    if reverse:
        atlas_ids.reverse()
        references.reverse()
    atlases = {
        atlas_id: make_lattice_atlas(
            atlas_id=atlas_id,
            size=int(build["atlas_size"]),
            bins=int(build["atlas_lattice_bins"]),
            multiplier=int(build["atlas_index_multiplier"]),
        )
        for atlas_id in atlas_ids
    }
    response_arrays: dict[tuple[str, str], np.ndarray] = {}
    preliminary: dict[tuple[str, str], dict[str, Any]] = {}
    with torch.inference_mode():
        for reference_row in references:
            reference_id = str(reference_row["reference_id"])
            reference = _tensor_from_image(
                _resolve(root, str(reference_row["local_path"])), torch, device
            )
            reference_condition = model.Embedding_Net(reference)
            for atlas_id in atlas_ids:
                atlas = _tensor_from_array(atlases[atlas_id], torch, device)
                atlas_condition = model.Embedding_Net(atlas)
                canonical_tensor = model.Canonicalizer(
                    atlas, condition=atlas_condition
                )["result"]
                response_tensor = model.Restyler(
                    canonical_tensor, condition=reference_condition
                )["result"]
                canonical = np.ascontiguousarray(
                    canonical_tensor[0].permute(1, 2, 0).cpu().numpy(),
                    dtype=np.float32,
                )
                response = np.ascontiguousarray(
                    response_tensor[0].permute(1, 2, 0).cpu().numpy(),
                    dtype=np.float32,
                )
                key = (reference_id, atlas_id)
                response_arrays[key] = response
                preliminary[key] = {
                    "reference_id": reference_id,
                    "reference_sha256": str(reference_row["sha256"]),
                    "atlas_id": atlas_id,
                    "atlas_sha256": hashlib.sha256(
                        atlases[atlas_id].tobytes(order="C")
                    ).hexdigest(),
                    "canonical": _stats(canonical),
                    "response": _stats(response),
                }
    all_reference_ids = sorted({key[0] for key in response_arrays})
    rows: list[dict[str, Any]] = []
    for key, row in preliminary.items():
        reference_id, atlas_id = key
        current = response_arrays[key].astype(np.float64)
        distances = sorted(
            float(
                np.sqrt(
                    np.mean(
                        (
                            current
                            - response_arrays[(other_reference, atlas_id)].astype(
                                np.float64
                            )
                        )
                        ** 2
                    )
                )
            )
            for other_reference in all_reference_ids
            if other_reference != reference_id
        )
        row["reference_pairwise_response_rmse"] = distances
        row["reference_pairwise_response_rmse_median"] = float(np.median(distances))
        rows.append(row)
    rows.sort(key=lambda row: (str(row["reference_id"]), str(row["atlas_id"])))
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "application_source_file_reads": 0,
        "application_source_pixel_decodes": 0,
        "operator_fit_executions": 0,
        "model": model_audit,
        "rows": rows,
        "summary": summarize_rows(rows, config),
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/u5_r2repid12_canoncgt_full_chain_atlas_response_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    config = json.loads(_resolve(root, str(args.config)).read_text(encoding="utf-8"))
    report = run_audit(
        root=root, config=config, device=str(args.device), reverse=bool(args.reverse)
    )
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    output = _resolve(root, str(args.output))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(encoded)
    print(hashlib.sha256(encoded).hexdigest())
    print(report["summary"]["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
