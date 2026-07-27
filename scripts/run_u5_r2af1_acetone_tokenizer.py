#!/usr/bin/env python
"""Run the exact external AceTone tokenizer for U5.R2AF1."""

from __future__ import annotations

import argparse
import importlib
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from src.eval.acetone_lut_tokenizer import (
    MANIFEST_SCHEMA,
    array_sha256,
    build_population,
    canonical_sha256,
    sha256_file,
)


def _relative_to_root(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("output artifact must remain inside project root") from exc


def _artifact_record(path: Path, value: np.ndarray, root: Path) -> dict[str, object]:
    return {
        "path": _relative_to_root(path, root),
        "file_sha256": sha256_file(path),
        "array_sha256": array_sha256(value),
    }


def _verify_external(config: dict, external_root: Path) -> None:
    source = config["external_source"]
    revision = subprocess.run(
        ["git", "-C", str(external_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != source["revision"]:
        raise RuntimeError("external repository revision mismatch")
    model_source = external_root / source["model_source_path"]
    checkpoint = external_root / source["checkpoint_path"]
    if sha256_file(model_source) != source["model_source_sha256"]:
        raise RuntimeError("external model-source hash mismatch")
    if sha256_file(checkpoint) != source["checkpoint_sha256"]:
        raise RuntimeError("external checkpoint hash mismatch")
    if checkpoint.stat().st_size != source["checkpoint_bytes"]:
        raise RuntimeError("external checkpoint size mismatch")


def _load_model(config: dict, external_root: Path) -> torch.nn.Module:
    checkpoint_path = external_root / config["external_source"]["checkpoint_path"]
    sys.path.insert(0, str(external_root.resolve()))
    try:
        module = importlib.import_module("model.vq")
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        args = checkpoint["args"]
        model = module.VQVAE3DLUT(
            codebook_size=args["codebook_size"],
            embedding_dim=args["embedding_dim"],
            beta=args["beta"],
            hidden=args["hidden"],
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        return model
    finally:
        sys.path.pop(0)


def run(
    config_path: Path,
    external_root: Path,
    output_dir: Path,
    run_id: str,
) -> dict:
    project_root = Path.cwd().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runtime = config["runtime"]
    if ".".join(map(str, sys.version_info[:3])) != runtime["python"]:
        raise RuntimeError("Python runtime mismatch")
    if torch.__version__ != runtime["torch"]:
        raise RuntimeError("PyTorch runtime mismatch")
    if runtime["device"] != "cpu":
        raise RuntimeError("AF1 contract requires CPU execution")
    _verify_external(config, external_root)

    seed = int(runtime["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(runtime["torch_threads"]))
    torch.use_deterministic_algorithms(bool(runtime["deterministic_algorithms"]))
    model = _load_model(config, external_root)

    population = build_population(config)
    names = list(population)
    duplicate_id = config["population"]["duplicate_control_id"]
    batch = np.stack([population[name] for name in names] + [population[duplicate_id]])
    tensor = torch.from_numpy(batch).permute(0, 4, 1, 2, 3).contiguous()
    with torch.inference_mode():
        indices = model.encode_indices(tensor)
        reconstructed = model.decode_indices(indices)
    codes = indices.cpu().numpy()
    outputs = reconstructed.permute(0, 2, 3, 4, 1).cpu().numpy()

    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, name in enumerate(names):
        source_path = output_dir / f"{name}_source.npy"
        reconstruction_path = output_dir / f"{name}_reconstruction.npy"
        codes_path = output_dir / f"{name}_codes.npy"
        np.save(source_path, population[name], allow_pickle=False)
        np.save(reconstruction_path, outputs[index], allow_pickle=False)
        np.save(codes_path, codes[index], allow_pickle=False)
        source_record = _artifact_record(source_path, population[name], project_root)
        reconstruction_record = _artifact_record(
            reconstruction_path, outputs[index], project_root
        )
        codes_record = _artifact_record(codes_path, codes[index], project_root)
        records.append(
            {
                "id": name,
                **{f"source_{key}": value for key, value in source_record.items()},
                **{
                    f"reconstruction_{key}": value
                    for key, value in reconstruction_record.items()
                },
                **{f"codes_{key}": value for key, value in codes_record.items()},
            }
        )

    duplicate_index = len(names)
    duplicate_reconstruction_path = output_dir / "duplicate_reconstruction.npy"
    duplicate_codes_path = output_dir / "duplicate_codes.npy"
    np.save(
        duplicate_reconstruction_path,
        outputs[duplicate_index],
        allow_pickle=False,
    )
    np.save(duplicate_codes_path, codes[duplicate_index], allow_pickle=False)
    duplicate_reconstruction = _artifact_record(
        duplicate_reconstruction_path, outputs[duplicate_index], project_root
    )
    duplicate_codes = _artifact_record(
        duplicate_codes_path, codes[duplicate_index], project_root
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "experiment_id": config["experiment_id"],
        "run_id": run_id,
        "config_sha256": canonical_sha256(config),
        "external_revision": config["external_source"]["revision"],
        "model_source_sha256": config["external_source"]["model_source_sha256"],
        "checkpoint_sha256": config["external_source"]["checkpoint_sha256"],
        "checkpoint_bytes": config["external_source"]["checkpoint_bytes"],
        "runtime": runtime,
        "records": records,
        "duplicate_control": {
            "id": duplicate_id,
            **{
                f"reconstruction_{key}": value
                for key, value in duplicate_reconstruction.items()
            },
            **{f"codes_{key}": value for key, value in duplicate_codes.items()},
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run(args.config, args.external_root, args.output_dir, args.run_id)


if __name__ == "__main__":
    main()
