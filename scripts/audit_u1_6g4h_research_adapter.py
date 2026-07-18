"""Run the frozen U1.6G4H research-adapter parity and isolation audit."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.filmfx import composite_layers, execute_staged_density_halation_default  # noqa: E402
import src.filmfx.staged_density_adapter as adapter_module  # noqa: E402
from src.filmfx.staged_density_adapter import (  # noqa: E402
    STAGED_DENSITY_ADAPTER_VERSION,
    render_staged_density_halation_research,
)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_path(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _sha_array(value: np.ndarray) -> str:
    contiguous = value if value.flags.c_contiguous else np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _metadata_payload(value) -> dict:
    return dataclasses.asdict(value)


def _metadata_sha(value) -> str:
    return _sha_bytes(
        json.dumps(
            _metadata_payload(value), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )


def _contains_forbidden_metadata_value(value: object) -> bool:
    if isinstance(value, (np.ndarray, list, dict, set)):
        return True
    if dataclasses.is_dataclass(value):
        return any(
            _contains_forbidden_metadata_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        )
    if isinstance(value, tuple):
        return any(_contains_forbidden_metadata_value(item) for item in value)
    return False


def synthetic_case(case: dict) -> np.ndarray:
    shape = tuple(int(item) for item in case["shape"])
    rng = np.random.default_rng(int(case["seed"]))
    y, x = np.mgrid[: shape[0], : shape[1]]
    base = rng.uniform(0.01, 0.30, size=shape).astype(np.float32)
    hot_a = np.exp(
        -((y - shape[0] * 0.38) ** 2 + (x - shape[1] * 0.31) ** 2) / 180.0
    ).astype(np.float32)
    hot_b = np.exp(
        -((y - shape[0] * 0.69) ** 2 + (x - shape[1] * 0.74) ** 2) / 240.0
    ).astype(np.float32)
    base += hot_a[..., None] * np.asarray([0.96, 0.74, 0.40], np.float32)
    base += hot_b[..., None] * np.asarray([0.56, 0.80, 1.00], np.float32)
    return np.clip(base, 0.0, 1.0).astype(np.float32)


def _real_case(case: dict) -> np.ndarray:
    path = ROOT / case["path"]
    if _sha_path(path) != case["sha256"]:
        raise ValueError(f"source hash mismatch: {case['case_id']}")
    with Image.open(path) as source:
        base = np.asarray(source.convert("RGB"), np.float32) / np.float32(255.0)
    if list(base.shape) != case["shape"]:
        raise ValueError(f"source shape mismatch: {case['case_id']}")
    return base


def _policy(
    base: np.ndarray,
    *,
    tile_size: int,
    row_chunk: int,
    output_margin: int,
    config: dict,
) -> dict:
    input_sha = _sha_array(base)
    input_writeable = bool(base.flags.writeable)
    reference_layer, _ = execute_staged_density_halation_default(
        base,
        tile_size=tile_size,
        source_row_chunk=int(config["source_row_chunk"]),
        coarse_row_chunk=int(config["coarse_row_chunk"]),
        name="staged_density_halation_research",
    )
    reference = composite_layers(base, [reference_layer], output_margin=output_margin)
    first, first_metadata = render_staged_density_halation_research(
        base,
        tile_size=tile_size,
        source_row_chunk=int(config["source_row_chunk"]),
        coarse_row_chunk=int(config["coarse_row_chunk"]),
        composite_row_chunk=row_chunk,
        output_margin=output_margin,
    )
    second, second_metadata = render_staged_density_halation_research(
        base,
        tile_size=tile_size,
        source_row_chunk=int(config["source_row_chunk"]),
        coarse_row_chunk=int(config["coarse_row_chunk"]),
        composite_row_chunk=row_chunk,
        output_margin=output_margin,
    )
    float_identical = first.tobytes() == reference.tobytes()
    srgb8_identical = np.rint(first * 255).astype(np.uint8).tobytes() == np.rint(
        reference * 255
    ).astype(np.uint8).tobytes()
    repeated = first.tobytes() == second.tobytes() and first_metadata == second_metadata
    input_unchanged = _sha_array(base) == input_sha and base.flags.writeable == input_writeable
    metadata_forbidden = _contains_forbidden_metadata_value(first_metadata)
    returned_array_count = sum(
        isinstance(item, np.ndarray) for item in (first, first_metadata)
    )
    row = {
        "tile_size": int(tile_size),
        "composite_row_chunk": int(row_chunk),
        "output_margin": int(output_margin),
        "float_byte_identical": float_identical,
        "srgb8_byte_identical": srgb8_identical,
        "input_byte_and_writeability_unchanged": input_unchanged,
        "repeat_output_and_metadata_identical": repeated,
        "metadata_contains_forbidden_mutable_or_ndarray": metadata_forbidden,
        "returned_array_count": returned_array_count,
        "scratch_bytes": first_metadata.scratch_bytes,
        "output_sha256": _sha_array(first),
        "metadata_sha256": _metadata_sha(first_metadata),
        "adapter_version": first_metadata.version,
        "executor_version": first_metadata.executor_version,
    }
    gates = config["gates"]
    row["passed"] = bool(
        row["float_byte_identical"] == gates["compositor_float_byte_identical"]
        and row["srgb8_byte_identical"] == gates["compositor_srgb8_byte_identical"]
        and row["input_byte_and_writeability_unchanged"] == gates["input_byte_unchanged"]
        and row["repeat_output_and_metadata_identical"]
        == gates["repeat_output_and_metadata_identical"]
        and row["metadata_contains_forbidden_mutable_or_ndarray"]
        == gates["metadata_contains_ndarray"]
        and row["returned_array_count"] == gates["returned_array_count"]
        and row["scratch_bytes"] == gates["scratch_bytes"]
        and row["adapter_version"] == config["adapter_version"]
        and row["executor_version"] == config["executor_version"]
    )
    return row


def _run_case(case: dict, base: np.ndarray, config: dict) -> dict:
    policies = []
    for tile_size in case["tile_sizes"]:
        for row_chunk in config["composite_row_chunks"]:
            for output_margin in config["output_margins"]:
                policies.append(
                    _policy(
                        base,
                        tile_size=int(tile_size),
                        row_chunk=int(row_chunk),
                        output_margin=int(output_margin),
                        config=config,
                    )
                )
    return {
        "case_id": case["case_id"],
        "shape": list(base.shape),
        "input_sha256": _sha_array(base),
        "policies": policies,
        "passed": all(policy["passed"] for policy in policies),
    }


def failure_probes() -> dict:
    case = {"shape": [131, 197, 3], "seed": 459}
    base = synthetic_case(case)
    parameters = dict(
        tile_size=37,
        source_row_chunk=31,
        coarse_row_chunk=3,
        composite_row_chunk=17,
        output_margin=0,
    )
    original_executor = adapter_module.execute_staged_density_halation_default
    original_compositor = adapter_module.composite_layers
    results = {}
    try:
        def fail_executor(*args, **kwargs):
            raise RuntimeError("injected executor failure")

        adapter_module.execute_staged_density_halation_default = fail_executor
        published = None
        try:
            published = render_staged_density_halation_research(base, **parameters)
        except RuntimeError as error:
            results["executor_failure"] = {
                "raised": "injected executor" in str(error),
                "partial_result_count": int(published is not None),
            }
        finally:
            adapter_module.execute_staged_density_halation_default = original_executor

        calls = 0

        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected second-row failure")
            return original_compositor(*args, **kwargs)

        adapter_module.composite_layers = fail_second
        published = None
        try:
            published = render_staged_density_halation_research(base, **parameters)
        except RuntimeError as error:
            results["second_row_failure"] = {
                "raised": "injected second-row" in str(error),
                "compositor_calls": calls,
                "partial_result_count": int(published is not None),
            }
        finally:
            adapter_module.composite_layers = original_compositor
    finally:
        adapter_module.execute_staged_density_halation_default = original_executor
        adapter_module.composite_layers = original_compositor
    results["passed"] = bool(
        set(results) >= {"executor_failure", "second_row_failure"}
        and results["executor_failure"]["raised"]
        and results["executor_failure"]["partial_result_count"] == 0
        and results["second_row_failure"]["raised"]
        and results["second_row_failure"]["compositor_calls"] == 2
        and results["second_row_failure"]["partial_result_count"] == 0
    )
    return results


def static_isolation(pre_contract_head: str) -> dict:
    production_files = [
        ROOT / "scripts/render_film.py",
        ROOT / "src/filmfx/__init__.py",
        *sorted((ROOT / "src/inference").rglob("*.py")),
    ]
    needles = (
        "staged_density_adapter",
        "render_staged_density_halation_research",
        "StagedDensityAdapterMetadata",
    )
    references = []
    for path in production_files:
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            if needle in text:
                references.append({"path": path.relative_to(ROOT).as_posix(), "needle": needle})
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", f"{pre_contract_head}..HEAD"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    sensitive_prefixes = (
        "scripts/render_film.py",
        "src/inference/",
        "src/filmfx/__init__.py",
        "configs/schemas/",
        "configs/render_profiles/",
    )
    sensitive_changes = [
        path for path in changed if any(path == prefix or path.startswith(prefix) for prefix in sensitive_prefixes)
    ]
    return {
        "production_references": references,
        "production_import_count": len(references),
        "sensitive_changes_since_pre_contract": sensitive_changes,
        "renderer_cli_schema_change_count": len(sensitive_changes),
        "passed": not references and not sensitive_changes,
    }


def run(config: dict, *, config_sha256: str) -> dict:
    cases = []
    for case in config["synthetic_cases"]:
        cases.append(_run_case(case, synthetic_case(case), config))
    real_case = config["real_case"]
    cases.append(_run_case(real_case, _real_case(real_case), config))
    failures = failure_probes()
    isolation = static_isolation(config["pre_contract_head"])
    automatic = bool(
        all(case["passed"] for case in cases)
        and failures["passed"]
        and isolation["passed"]
        and isolation["production_import_count"]
        == config["gates"]["production_import_count"]
        and isolation["renderer_cli_schema_change_count"]
        == config["gates"]["renderer_cli_schema_change_count"]
    )
    return {
        "schema_version": 1,
        "node": config["node"],
        "config_sha256": config_sha256,
        "adapter_version": STAGED_DENSITY_ADAPTER_VERSION,
        "cases": cases,
        "failure_probes": failures,
        "static_isolation": isolation,
        "gate_result": {
            "automatic_pass": automatic,
            "renderer_integration_allowed": False,
        },
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = run(config, config_sha256=_sha_path(args.config))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["gate_result"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
