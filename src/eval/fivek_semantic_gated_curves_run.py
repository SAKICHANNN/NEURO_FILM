"""Hash-bound runner for U5.R2BV0 semantic-gated explicit curves."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel

from src.eval.fivek_bilateral_gain_capacity_run import _selected_rows
from src.eval.fivek_semantic_gated_curves import evaluate_semantic_gated_curves


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    if (
        config.get("status") != "contract_frozen_implementation_ready"
        or config.get("confirmation_pixels_allowed")
        or config.get("production_integration_allowed")
        or config.get("film_or_stock_claim_allowed")
    ):
        raise ValueError("BV0 boundary drift")
    paper = config["research_basis"]
    paper_path = root / paper["paper_pdf_path"]
    if (
        not paper_path.is_file()
        or paper_path.stat().st_size != int(paper["paper_pdf_bytes"])
        or _sha256(paper_path) != paper["paper_pdf_sha256"]
    ):
        raise ValueError("BV0 paper identity drift")
    population = config["population"]
    manifest_path = root / population["manifest"]
    if _sha256(manifest_path) != population["manifest_sha256"]:
        raise ValueError("BV0 population drift")
    model = config["semantic_gate"]
    model_root = root / model["root"]
    for name, key in (
        ("model.safetensors", "model_sha256"),
        ("config.json", "config_sha256"),
        ("preprocessor_config.json", "preprocessor_sha256"),
    ):
        if _sha256(model_root / name) != model[key]:
            raise ValueError(f"BV0 model identity drift: {name}")
    operator = config["explicit_operator"]
    if (
        operator["family"] != "semantic_gated_nine_piecewise_linear_residual_curves"
        or int(operator["curves"]) != 9
        or int(operator["parameter_count"]) != 63
        or operator.get("hard_output_clipping_allowed")
        or operator.get("learned_final_rgb_allowed")
        or model.get("training_or_finetuning_allowed")
    ):
        raise ValueError("BV0 representation drift")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _extract_semantic_gates(
    rows: list[dict[str, Any]], root: Path, spec: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    device = str(spec["device"])
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("BV0 frozen CUDA device unavailable")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    model = (
        AutoModel.from_pretrained(
            root / spec["root"],
            local_files_only=True,
            attn_implementation=str(spec["attention_implementation"]),
        )
        .eval()
        .to(device)
    )
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    gates: dict[str, np.ndarray] = {}
    batch_size = int(spec["batch_size"])
    patch_rows, patch_columns = map(int, spec["patch_grid"])
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        tensors = [
            torch.from_numpy(np.asarray(row["source"], dtype=np.float32)).permute(
                2, 0, 1
            )
            for row in batch
        ]
        resized = torch.stack(
            [
                F.interpolate(
                    item[None],
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )[0]
                for item in tensors
            ]
        ).to(device)
        with torch.inference_mode():
            result = model(
                pixel_values=(resized - mean) / std,
                output_attentions=True,
            )
        attention = result.attentions[-1].mean(dim=1)[:, 0, 1:]
        if attention.shape[1] != patch_rows * patch_columns:
            raise ValueError("BV0 attention grid drift")
        patch = attention.reshape(-1, 1, patch_rows, patch_columns)
        patch_mean = patch.mean(dim=(2, 3), keepdim=True)
        patch_std = patch.std(dim=(2, 3), correction=0, keepdim=True)
        if torch.any(patch_std <= 1e-12):
            raise ValueError("BV0 attention is spatially constant")
        patch = torch.sigmoid((patch - patch_mean) / patch_std)
        for index, row in enumerate(batch):
            shape = np.asarray(row["source"]).shape[:2]
            gate = F.interpolate(
                patch[index : index + 1],
                size=shape,
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            gates[str(row["pair_id"])] = gate.cpu().numpy().astype(np.float64)
    del model
    torch.cuda.empty_cache()
    return gates


def run_semantic_gated_curves(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    output_path: Path,
    software_commit: str,
) -> dict[str, Any]:
    manifest = validate_contract(root, config)
    first_variant = str(config["population"]["target_variants"][0])
    base_rows = _selected_rows(root, manifest, config, first_variant)
    gates = _extract_semantic_gates(base_rows, root, config["semantic_gate"])
    variants = {}
    all_rows = []
    all_gate_hashes: dict[str, str] = {}
    for variant in config["population"]["target_variants"]:
        rows = _selected_rows(root, manifest, config, str(variant))
        for row in rows:
            row["semantic_gate"] = gates[str(row["pair_id"])]
        result = evaluate_semantic_gated_curves(rows, config)
        variants[str(variant)] = {
            "automatic_pass": result["automatic_pass"],
            "metrics": result["metrics"],
            "gates": result["gates"],
        }
        all_rows.extend(result["rows"])
        all_gate_hashes.update(result["gate_sha256"])
    gates_spec = config["evaluation"]["automatic_gates"]
    population_gates = {
        "target_variant_count": len(variants)
        == int(gates_spec["target_variant_count_exact"]),
        "target_variant_identity": sorted(variants)
        == sorted(map(str, config["population"]["target_variants"])),
    }
    body = {
        "schema": "neuro_film.u5_r2bv0_fivek_semantic_gated_curves_formal_report.v1",
        "experiment_id": config["experiment_id"],
        "config_sha256": _sha256(config_path),
        "software_commit": software_commit,
        "paper_sha256": config["research_basis"]["paper_pdf_sha256"],
        "model_sha256": config["semantic_gate"]["model_sha256"],
        "manifest_sha256": config["population"]["manifest_sha256"],
        "confirmation_rows_loaded": 0,
        "semantic_gate_sha256": dict(sorted(all_gate_hashes.items())),
        "variants": variants,
        "population_gates": population_gates,
        "automatic_pass": all(population_gates.values())
        and all(row["automatic_pass"] for row in variants.values()),
        "rows": sorted(
            all_rows,
            key=lambda row: (row["target_variant"], row["pair_id"], row["fold"]),
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    stable_body = {
        key: value for key, value in body.items() if key != "software_commit"
    }
    body["stable_evidence_id"] = hashlib.sha256(_canonical(stable_body)).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical(body))
    return body


__all__ = ["run_semantic_gated_curves", "validate_contract"]
