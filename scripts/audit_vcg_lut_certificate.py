from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.trilinear_lut_certificate import numerical_cell_bounds

CONFIG = "configs/ai_vcg_lut_certificate_v1.json"
OWNED = [
    CONFIG,
    "src/eval/trilinear_lut_certificate.py",
    "scripts/audit_vcg_lut_certificate.py",
    "tests/test_trilinear_lut_certificate.py",
]


def read_bound(path: Path, expected: str) -> bytes:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"Source hash drift: {path}")
    return payload


def load_sources(root: Path, config: dict) -> list[tuple[dict, np.ndarray]]:
    sources = config["sources"]
    if len(sources) != 6 or len({row["id"] for row in sources}) != 6:
        raise ValueError("Expected the six distinct frozen LUT identities")
    reports = {}
    loaded = []
    for row in sources:
        report_key = (row["report"], row["report_sha256"])
        if report_key not in reports:
            reports[report_key] = json.loads(
                read_bound(root / row["report"], row["report_sha256"])
            )
        report = reports[report_key]
        if len(report["rows"]) != 3 or row["row"] not in range(3):
            raise ValueError("Unexpected parent report row contract")
        original = report["rows"][row["row"]]
        if "lut_sha256" in original and original["lut_sha256"] != row["lut_sha256"]:
            raise ValueError("Parent LUT hash mismatch")
        payload = read_bound(root / row["lut"], row["lut_sha256"])
        array = np.load(io.BytesIO(payload), allow_pickle=False)
        if (
            list(array.shape) != config["stored_shape"]
            or str(array.dtype) != config["stored_dtype"]
            or not np.isfinite(array).all()
        ):
            raise ValueError("Unexpected saved VCG LUT shape, dtype or values")
        cube = array.astype(np.float64).reshape((config["lut_size"],) * 3 + (3,))
        loaded.append((row, cube))
    return loaded


def main():
    config_bytes = (ROOT / CONFIG).read_bytes()
    cfg = json.loads(config_bytes)
    if cfg["lut_size"] != 16 or cfg["variants"] != ["raw_nodes", "unit_clipped_nodes"]:
        raise ValueError("Unsupported frozen analysis contract")
    out = ROOT / cfg["destination"]
    output_root = (ROOT / "outputs").resolve()
    if (
        out.exists()
        or out.resolve().parent != output_root
        or out.resolve().drive.upper() != "P:"
    ):
        raise ValueError("Fresh direct P-backed outputs child required")
    loaded = load_sources(ROOT, cfg)
    report = {
        "schema": cfg["schema"],
        "config": cfg,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "owned_worktree_status": subprocess.check_output(
            ["git", "status", "--short", "--", *OWNED], cwd=ROOT, text=True
        ).splitlines(),
        "code_identity": "Base commit plus exact implementation hashes; uncommitted files are disclosed",
        "implementation_sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in OWNED
        },
        "numpy_version": np.__version__,
        "python_version": sys.version,
        "float_precision": "float64 analysis of exact saved float32 node values",
        "inputs_verified_before_output_creation": True,
        "photo_reads": 0,
        "model_forwards": 0,
        "training_or_projection": False,
        "product_promotion": False,
        "rounding_error_certified": False,
        "rows": [],
    }
    out.mkdir()
    for source, raw in loaded:
        for variant in cfg["variants"]:
            cube = raw if variant == "raw_nodes" else np.clip(raw, 0.0, 1.0)
            bounds, summary = numerical_cell_bounds(
                cube,
                absolute_margin=cfg["numerical_margin"]["absolute"],
                relative_margin=cfg["numerical_margin"]["relative"],
            )
            target = out / f"{source['id']}_{variant}_cell_bounds.npy"
            with target.open("xb") as handle:
                np.save(handle, bounds, allow_pickle=False)
            report["rows"].append(
                {
                    "source": source,
                    "variant": variant,
                    "node_minimum": float(cube.min()),
                    "node_maximum": float(cube.max()),
                    "raw_out_of_unit_range_component_count": int(
                        ((raw < 0) | (raw > 1)).sum()
                    ),
                    "node_component_count": int(raw.size),
                    "continuous_range_in_unit_cube_by_node_convexity": bool(
                        cube.min() >= 0 and cube.max() <= 1
                    ),
                    "cell_bounds": target.name,
                    "cell_bounds_sha256": hashlib.sha256(
                        target.read_bytes()
                    ).hexdigest(),
                    "summary": summary,
                }
            )
            print(
                source["id"],
                variant,
                "gain",
                summary["global_gain_upper_numerical_estimate"],
                "residual",
                summary["global_residual_upper_numerical_estimate"],
                summary["global_residual_condition"],
                flush=True,
            )
    report["stage"] = "COMPLETE_NUMERICAL_ANALYSIS_NO_PHOTOGRAPHIC_DECISION"
    with (out / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        "report_sha256", hashlib.sha256((out / "report.json").read_bytes()).hexdigest()
    )


if __name__ == "__main__":
    main()
