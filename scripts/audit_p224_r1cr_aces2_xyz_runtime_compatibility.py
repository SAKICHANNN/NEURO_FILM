"""Audit exact R1CR XYZ output-transform compatibility across OCIO runtimes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from src.preprocess.ocio_aces2_output import OCIO_VERSION, build_aces2_numeric_fixture

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p224_r1cr_aces2_xyz_runtime_compatibility_v1.json"
SCHEMA = "neuro-film.p224-r1cr-aces2-xyz-runtime-compatibility-contract.v1"


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def run_command(
    command: list[str], *, cwd: Path | None = None, check: bool = True
) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(command, cwd=cwd, capture_output=True, check=False)
    if check and completed.returncode != 0:
        detail = (completed.stdout + completed.stderr).decode(errors="replace")
        raise RuntimeError(f"command failed ({completed.returncode}): {detail}")
    return completed


def git_bytes(repository: Path, *arguments: str) -> bytes:
    return run_command(["git", *arguments], cwd=repository).stdout


def git_text(repository: Path, *arguments: str) -> str:
    return git_bytes(repository, *arguments).decode("utf-8").strip()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != SCHEMA:
        raise RuntimeError("P224 contract schema differs")
    if config.get("status") != "FROZEN_AFTER_R1CS_CLOSURE_BEFORE_CROSS_RUNTIME_SCORE":
        raise RuntimeError("P224 contract is not frozen after R1CS closure")
    if "analytic" not in config["excluded_closed_route"]["exclusion"]:
        raise RuntimeError("P224 does not explicitly exclude the closed R1CS route")
    if config["execution"]["media_reads"] != 0:
        raise RuntimeError("P224 forbids media reads")


def _runtime_version(python: Path) -> str:
    completed = run_command(
        [python.as_posix(), "-c", "import PyOpenColorIO as o; print(o.__version__)"]
    )
    return completed.stdout.decode("utf-8").strip()


def inspect_identities(config: dict[str, Any], producer_python: Path) -> dict[str, Any]:
    producer = config["producer"]
    repository = Path(producer["repository"])
    if not repository.is_dir():
        raise RuntimeError("P224 producer repository is unavailable")
    return {
        "producer_evidence_commit_type": git_text(
            repository, "cat-file", "-t", producer["evidence_commit"]
        ),
        "producer_implementation_commit_type": git_text(
            repository, "cat-file", "-t", producer["implementation_commit"]
        ),
        "producer_source_blob": git_text(
            repository,
            "rev-parse",
            f'{producer["evidence_commit"]}:{producer["source_path"]}',
        ),
        "producer_evidence_blob": git_text(
            repository,
            "rev-parse",
            f'{producer["evidence_commit"]}:{producer["evidence_path"]}',
        ),
        "producer_evidence_sha256": sha256_bytes(
            git_bytes(
                repository,
                "show",
                f'{producer["evidence_commit"]}:{producer["evidence_path"]}',
            )
        ),
        "consumer_source_blob": git_text(
            ROOT, "hash-object", config["consumer"]["source_path"]
        ),
        "producer_opencolorio_version": _runtime_version(producer_python),
        "consumer_opencolorio_version": OCIO_VERSION,
    }


def identities_exact(config: dict[str, Any], observed: dict[str, Any]) -> bool:
    producer = config["producer"]
    consumer = config["consumer"]
    return (
        observed["producer_evidence_commit_type"] == "commit"
        and observed["producer_implementation_commit_type"] == "commit"
        and observed["producer_source_blob"] == producer["source_blob"]
        and observed["producer_evidence_blob"] == producer["evidence_blob"]
        and observed["producer_evidence_sha256"] == producer["evidence_sha256"]
        and observed["consumer_source_blob"] == consumer["source_blob"]
    )


def _run_runtime(
    *,
    python: Path,
    source: Path,
    fixture: np.ndarray,
    order: str,
    producer_source: bool,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    indices = np.arange(fixture.shape[0])
    execution_indices = indices if order == "forward" else indices[::-1]
    ordered = np.ascontiguousarray(fixture[execution_indices])
    with tempfile.TemporaryDirectory(prefix="neuro-film-p224-") as directory:
        root = Path(directory)
        input_path = root / "input.npy"
        output_path = root / "output.npy"
        facts_path = root / "facts.json"
        invocation = root / "invoke.py"
        np.save(input_path, ordered, allow_pickle=False)
        invocation.write_text(
            """from __future__ import annotations
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import PyOpenColorIO as ocio

source, input_path, output_path, facts_path, mode, scene_builtin, output_builtin, scale = sys.argv[1:]
values = np.load(input_path, allow_pickle=False)
before = hashlib.sha256(values.tobytes(order="C")).hexdigest()
if mode == "producer":
    spec = importlib.util.spec_from_file_location("frozen_r1cr_bridge", Path(source))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    output = module.render_acescg_to_xyz_d65_nits(values)
else:
    group = ocio.GroupTransform()
    group.appendTransform(ocio.BuiltinTransform(scene_builtin))
    group.appendTransform(ocio.BuiltinTransform(output_builtin))
    output = np.array(values, dtype=np.float32, order="C", copy=True)
    ocio.Config.CreateRaw().getProcessor(group).getDefaultCPUProcessor().applyRGB(output)
    output *= np.float32(float(scale))
after = hashlib.sha256(values.tobytes(order="C")).hexdigest()
np.save(output_path, output, allow_pickle=False)
Path(facts_path).write_text(json.dumps({
    "input_unchanged": before == after,
    "output_dtype": str(output.dtype),
    "output_c_contiguous": bool(output.flags.c_contiguous),
    "opencolorio_version": ocio.__version__,
}), encoding="utf-8")
""",
            encoding="utf-8",
        )
        run_command(
            [
                python.as_posix(),
                invocation.as_posix(),
                source.as_posix(),
                input_path.as_posix(),
                output_path.as_posix(),
                facts_path.as_posix(),
                "producer" if producer_source else "consumer",
                config["producer"]["scene_to_reference_builtin"],
                config["producer"]["output_builtin"],
                str(config["producer"]["display_reference_scale_nits"]),
            ]
        )
        output = np.load(output_path, allow_pickle=False)
        facts = json.loads(facts_path.read_text(encoding="utf-8"))
    canonical = np.empty_like(output)
    canonical[execution_indices] = output
    return canonical, facts


def execute(order: str, producer_python: Path, config_path: Path = CONFIG) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    observed = inspect_identities(config, producer_python)
    exact_identities = identities_exact(config, observed)
    if not exact_identities:
        raise RuntimeError("P224 frozen producer or consumer identity differs")

    fixture = build_aces2_numeric_fixture()
    fixture_before = fixture.copy()
    with tempfile.TemporaryDirectory(prefix="neuro-film-p224-source-") as directory:
        producer_source = Path(directory) / "aces2_render_bridge.py"
        producer_source.write_bytes(
            git_bytes(
                Path(config["producer"]["repository"]),
                "show",
                f'{config["producer"]["evidence_commit"]}:{config["producer"]["source_path"]}',
            )
        )
        producer_output, producer_facts = _run_runtime(
            python=producer_python,
            source=producer_source,
            fixture=fixture,
            order=order,
            producer_source=True,
            config=config,
        )
    consumer_output, consumer_facts = _run_runtime(
        python=Path(os.sys.executable),
        source=ROOT / config["consumer"]["source_path"],
        fixture=fixture,
        order=order,
        producer_source=False,
        config=config,
    )
    error = np.abs(
        producer_output.astype(np.float64) - consumer_output.astype(np.float64)
    )
    metrics = {
        "fixture_rows": int(fixture.shape[0]),
        "maximum_absolute_difference_nits": float(np.max(error)),
        "rmse_difference_nits": float(np.sqrt(np.mean(np.square(error)))),
        "producer_output_sha256": sha256_bytes(
            np.ascontiguousarray(producer_output).tobytes()
        ),
        "consumer_output_sha256": sha256_bytes(
            np.ascontiguousarray(consumer_output).tobytes()
        ),
    }
    thresholds = config["gates"]
    gates = {
        "producer_and_consumer_identities_exact": exact_identities,
        "producer_and_consumer_runtime_versions_exact": (
            observed["producer_opencolorio_version"]
            == config["producer"]["required_opencolorio_version"]
            and observed["consumer_opencolorio_version"]
            == config["consumer"]["required_opencolorio_version"]
        ),
        "fixture_rows_exact": metrics["fixture_rows"]
        == config["consumer"]["fixture_rows"],
        "inputs_unchanged": bool(
            producer_facts["input_unchanged"]
            and consumer_facts["input_unchanged"]
            and np.array_equal(fixture, fixture_before)
        ),
        "outputs_finite": bool(
            np.isfinite(producer_output).all() and np.isfinite(consumer_output).all()
        ),
        "maximum_absolute_difference_nits": metrics[
            "maximum_absolute_difference_nits"
        ]
        <= thresholds["maximum_absolute_difference_nits"],
        "rmse_difference_nits": metrics["rmse_difference_nits"]
        <= thresholds["rmse_difference_nits"],
        "canonical_order_reconstruction_exact": True,
    }
    decision = (
        "PASS_PRIVATE_R1CR_XYZ_CROSS_RUNTIME_COMPATIBILITY"
        if all(gates.values())
        else "FAIL_CLOSED_R1CR_XYZ_CROSS_RUNTIME_COMPATIBILITY"
    )
    scientific = {
        "protocol": config["schema"],
        "producer": {
            "evidence_commit": config["producer"]["evidence_commit"],
            "implementation_commit": config["producer"]["implementation_commit"],
            "opencolorio_version": observed["producer_opencolorio_version"],
        },
        "consumer": {
            "source_blob": observed["consumer_source_blob"],
            "opencolorio_version": observed["consumer_opencolorio_version"],
        },
        "excluded_closed_route": config["excluded_closed_route"],
        "observed_identities": observed,
        "runtime_facts": {
            "producer": producer_facts,
            "consumer": consumer_facts,
        },
        "metrics": metrics,
        "gates": gates,
        "decision": decision,
        "claim_ceiling": config["claim_ceiling"],
    }
    scientific["stable_identity"] = "sha256:" + sha256_bytes(
        canonical_bytes(scientific)
    )
    return {
        "schema": "neuro_film.p224_r1cr_aces2_xyz_runtime_compatibility_result.v1",
        "experiment_id": "P224",
        "scientific": scientific,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--producer-python", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = execute(arguments.order, arguments.producer_python, arguments.config)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_bytes(canonical_bytes(report))
    os.replace(temporary, arguments.output)
    print(json.dumps(report["scientific"]["gates"], sort_keys=True))
    return 0 if all(report["scientific"]["gates"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
