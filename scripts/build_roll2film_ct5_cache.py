"""Build hash-verified equal-image CT5 training and internal-dev pixel caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.roll2film.ct5_data import (  # noqa: E402
    CT5DataContract,
    sample_ct5_internal_dev,
    sample_ct5_training,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "roll2film_ct5_baselines.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "roll2film" / "ct5_v1" / "data_cache",
    )
    parser.add_argument(
        "--include-internal-dev",
        action="store_true",
        help="Evaluator-only: decode the frozen 465-identity internal dev lockbox.",
    )
    return parser.parse_args()


def _commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_npy(path: Path, values: np.ndarray) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    contract = CT5DataContract.from_config(args.config, ROOT)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    training = sample_ct5_training(contract, progress=lambda message: print(message, flush=True))
    hashes = {
        "training_source.npy": _write_npy(output / "training_source.npy", training.source_pixels)
    }
    for domain, values in training.target_pixels.items():
        name = f"training_target_{domain}.npy"
        hashes[name] = _write_npy(output / name, values)
    membership: dict[str, object] = {
        "training_source_ids": training.source_ids,
        "training_target_ids": training.target_ids,
    }
    dev_summary = None
    if args.include_internal_dev:
        dev = sample_ct5_internal_dev(contract, progress=lambda message: print(message, flush=True))
        for fold in ("pilot", "confirmatory"):
            mask = np.asarray([value == fold for value in dev.folds])
            input_name = f"dev_{fold}_input.npy"
            hashes[input_name] = _write_npy(output / input_name, dev.input_pixels[mask])
            for domain, values in dev.target_pixels.items():
                name = f"dev_{fold}_target_{domain}.npy"
                hashes[name] = _write_npy(output / name, values[mask])
        membership["internal_dev"] = [
            {"content_id": content, "cluster_id": cluster, "fold": fold}
            for content, cluster, fold in zip(dev.content_ids, dev.cluster_ids, dev.folds)
        ]
        dev_summary = {
            "identities": len(dev.content_ids),
            "pilot": dev.folds.count("pilot"),
            "confirmatory": dev.folds.count("confirmatory"),
        }
    membership_path = output / "membership.json"
    membership_path.write_bytes((json.dumps(membership, indent=2, sort_keys=True) + "\n").encode())
    hashes["membership.json"] = hashlib.sha256(membership_path.read_bytes()).hexdigest()
    report = {
        "schema_version": 1,
        "experiment_id": contract.experiment_id,
        "config_sha256": contract.config_sha256,
        "software_commit": _commit(),
        "manifest_sha256": contract.manifest_hashes,
        "training": {
            "source_identities": len(training.source_ids),
            "target_identities": {key: len(value) for key, value in training.target_ids.items()},
            "pixels_per_image": contract.pixels_per_training_image,
        },
        "internal_dev": dev_summary,
        "arrays_sha256": hashes,
        "final_628_parsed_or_decoded": False,
        "claim_boundary": "FilmSet recipe research only; generated caches are internal and non-redistributable.",
    }
    report_path = output / "report.json"
    report_path.write_bytes((json.dumps(report, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps({"report": str(report_path), "internal_dev": dev_summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
