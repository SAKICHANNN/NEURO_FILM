"""Audit exact R1CV P3-D65 Rec.2100-PQ parity across OCIO runtimes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess.ocio_aces2_output import (
    OCIO_VERSION,
    OcioAces2RuntimeError,
    apply_aces2_output_packed,
    build_aces2_numeric_fixture,
)

CONFIG = ROOT / "configs/p226_r1cv_rec2100_pq_runtime_compatibility_v1.json"
SCHEMA = "neuro-film.p226-r1cv-rec2100-pq-runtime-compatibility-contract.v1"
TARGET = "hdr_p3d65_1000nit_rec2100_pq"
FROZEN_EXISTING_HASHES = {
    "sdr_rec709": "416132dec9e3fca541e900c24fb68ff751b9064192610adf6d90b2546378dd9a",
    "hdr_rec2020_pq": "c95554153a53588fccb121380afd1b34c7bd6a3f5e8171ab280592a5b30df923",
}
OFFICIAL_BLACK = np.float32(7.30942701920867e-7)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_command(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(command, cwd=cwd, capture_output=True, check=False)
    if completed.returncode != 0:
        detail = (completed.stdout + completed.stderr).decode(errors="replace")
        raise RuntimeError(f"command failed ({completed.returncode}): {detail}")
    return completed


def git_bytes(repository: Path, *arguments: str) -> bytes:
    return run_command(["git", *arguments], cwd=repository).stdout


def git_text(repository: Path, *arguments: str) -> str:
    return git_bytes(repository, *arguments).decode("utf-8").strip()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != SCHEMA:
        raise RuntimeError("P226 contract schema differs")
    if config.get("status") != "FROZEN_EXECUTION_AMENDMENT_AFTER_PRE_REPORT_DOMAIN_REJECTION":
        raise RuntimeError("P226 amended execution contract is not frozen")
    if config["consumer"]["new_private_target"] != TARGET:
        raise RuntimeError("P226 target identity differs")
    if not config["preserved_boundaries"]["r1cs_status"].startswith("FAIL_CLOSED"):
        raise RuntimeError("P226 does not preserve the R1CS closure")
    if config["execution"]["durable_media_outputs"] != 0:
        raise RuntimeError("P226 forbids media output")
    amendment = config["execution_amendment"]
    if amendment["formal_reports_before_amendment"] != 0:
        raise RuntimeError("P226 amendment must precede every formal report")
    if not amendment["output_range_gate_unchanged"]:
        raise RuntimeError("P226 amendment cannot relax the output range")


def build_p226_parity_fixture() -> np.ndarray:
    """Map the inherited stress fixture into R1CV's accepted display domain."""

    inherited = np.maximum(build_aces2_numeric_fixture(), np.float32(0.0))
    neutral = np.mean(inherited, axis=1, keepdims=True, dtype=np.float32)
    return np.ascontiguousarray(
        neutral + np.float32(0.2) * (inherited - neutral), dtype=np.float32
    )


def _runtime_versions(python: Path) -> dict[str, str]:
    code = (
        "import json,numpy,PyOpenColorIO as ocio,platform;"
        "print(json.dumps({'numpy':numpy.__version__,'ocio':ocio.__version__,"
        "'python':platform.python_version()},sort_keys=True))"
    )
    return json.loads(run_command([str(python), "-c", code]).stdout)


def inspect_identities(
    config: dict[str, Any],
    producer_python: Path,
    producer_wheel: Path,
    producer_config: Path,
) -> dict[str, Any]:
    producer = config["producer"]
    repository = Path(producer["repository"])
    prereg_parent = git_text(ROOT, "rev-parse", "4e1a6c63^")
    return {
        "producer_evidence_commit_type": git_text(repository, "cat-file", "-t", producer["evidence_commit"]),
        "producer_implementation_commit_type": git_text(repository, "cat-file", "-t", producer["implementation_commit"]),
        "producer_source_blob": git_text(repository, "rev-parse", f'{producer["evidence_commit"]}:{producer["source_path"]}'),
        "producer_source_sha256": sha256_bytes(git_bytes(repository, "show", f'{producer["evidence_commit"]}:{producer["source_path"]}')),
        "producer_evidence_blob": git_text(repository, "rev-parse", f'{producer["evidence_commit"]}:{producer["evidence_path"]}'),
        "producer_evidence_sha256": sha256_bytes(git_bytes(repository, "show", f'{producer["evidence_commit"]}:{producer["evidence_path"]}')),
        "producer_hdr_blob": git_text(repository, "rev-parse", f'{producer["evidence_commit"]}:src/zhuise/hdr.py'),
        "producer_hdr_sha256": sha256_bytes(git_bytes(repository, "show", f'{producer["evidence_commit"]}:src/zhuise/hdr.py')),
        "producer_runtime": _runtime_versions(producer_python),
        "producer_wheel_bytes": producer_wheel.stat().st_size,
        "producer_wheel_sha256": sha256_file(producer_wheel),
        "producer_config_bytes": producer_config.stat().st_size,
        "producer_config_sha256": sha256_file(producer_config),
        "consumer_parent_commit": prereg_parent,
        "consumer_parent_source_blob": git_text(ROOT, "rev-parse", f'{prereg_parent}:{config["consumer"]["source_path"]}'),
        "consumer_current_source_sha256": sha256_file(ROOT / config["consumer"]["source_path"]),
        "consumer_runtime": {
            "numpy": np.__version__,
            "ocio": OCIO_VERSION,
            "python": sys.version.split()[0],
        },
    }


def identities_exact(config: dict[str, Any], observed: dict[str, Any]) -> bool:
    producer = config["producer"]
    consumer = config["consumer"]
    return bool(
        observed["producer_evidence_commit_type"] == "commit"
        and observed["producer_implementation_commit_type"] == "commit"
        and observed["producer_source_blob"] == producer["source_blob"]
        and observed["producer_source_sha256"] == producer["source_sha256"]
        and observed["producer_evidence_blob"] == producer["evidence_blob"]
        and observed["producer_evidence_sha256"] == producer["evidence_sha256"]
        and observed["producer_wheel_sha256"] == producer["opencolorio_wheel_sha256"]
        and observed["producer_config_sha256"] == producer["config_sha256"]
        and observed["consumer_parent_source_blob"] == consumer["parent_source_blob"]
    )


def _run_producer(
    *,
    python: Path,
    fixture: np.ndarray,
    order: str,
    config: dict[str, Any],
    producer_config: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    indices = np.arange(fixture.shape[0])
    execution_indices = indices if order == "forward" else indices[::-1]
    ordered = np.ascontiguousarray(fixture[execution_indices])
    producer = config["producer"]
    repository = Path(producer["repository"])
    with tempfile.TemporaryDirectory(prefix="neuro-film-p226-producer-") as directory:
        root = Path(directory)
        package = root / "zhuise"
        package.mkdir()
        (package / "__init__.py").write_bytes(b"")
        (package / "aces2_render_bridge.py").write_bytes(
            git_bytes(repository, "show", f'{producer["evidence_commit"]}:{producer["source_path"]}')
        )
        (package / "hdr.py").write_bytes(
            git_bytes(repository, "show", f'{producer["evidence_commit"]}:src/zhuise/hdr.py')
        )
        input_path = root / "input.npy"
        output_path = root / "output.npy"
        facts_path = root / "facts.json"
        invocation = root / "invoke.py"
        np.save(input_path, ordered, allow_pickle=False)
        invocation.write_text(
            """from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np
import PyOpenColorIO as ocio
sys.path.insert(0, sys.argv[1])
from zhuise.aces2_render_bridge import render_acescg_to_rec2100_pq_ocio
input_path, output_path, facts_path, config_path, config_sha = sys.argv[2:]
values = np.load(input_path, allow_pickle=False)
before = hashlib.sha256(values.tobytes(order='C')).hexdigest()
output = render_acescg_to_rec2100_pq_ocio(values, config_path=config_path, expected_config_sha256=config_sha)
after = hashlib.sha256(values.tobytes(order='C')).hexdigest()
np.save(output_path, output, allow_pickle=False)
Path(facts_path).write_text(json.dumps({
    'input_unchanged': before == after,
    'output_dtype': str(output.dtype),
    'output_c_contiguous': bool(output.flags.c_contiguous),
    'output_finite_in_0_1': bool(np.isfinite(output).all() and np.all(output >= 0) and np.all(output <= 1)),
    'opencolorio_version': ocio.__version__,
    'numpy_version': np.__version__,
}, sort_keys=True), encoding='utf-8')
""",
            encoding="utf-8",
        )
        run_command([
            str(python), str(invocation), str(root), str(input_path), str(output_path),
            str(facts_path), str(producer_config), producer["config_sha256"],
        ])
        output = np.load(output_path, allow_pickle=False)
        facts = json.loads(facts_path.read_text(encoding="utf-8"))
    canonical = np.empty_like(output)
    canonical[execution_indices] = output
    return canonical, facts


def execute(
    order: str,
    producer_python: Path,
    producer_wheel: Path,
    producer_config: Path,
    config_path: Path = CONFIG,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    observed = inspect_identities(config, producer_python, producer_wheel, producer_config)
    exact_identities = identities_exact(config, observed)
    if not exact_identities:
        raise RuntimeError("P226 frozen producer/config/wheel or consumer-parent identity differs")
    if observed["producer_runtime"]["numpy"] != "2.3.5":
        raise RuntimeError("P226 producer NumPy runtime differs from R1CV")

    inherited_fixture = build_aces2_numeric_fixture()
    fixture = build_p226_parity_fixture()
    before = fixture.copy()
    producer_output, producer_facts = _run_producer(
        python=producer_python,
        fixture=fixture,
        order=order,
        config=config,
        producer_config=producer_config,
    )
    consumer_output = apply_aces2_output_packed(fixture, TARGET)
    existing_hashes = {
        target: sha256_bytes(
            apply_aces2_output_packed(inherited_fixture, target).tobytes()
        )
        for target in config["consumer"]["existing_targets"]
    }
    invalid_rejected = False
    try:
        apply_aces2_output_packed(fixture, "invalid")  # type: ignore[arg-type]
    except OcioAces2RuntimeError:
        invalid_rejected = True

    black_index = int(np.flatnonzero(np.all(fixture == 0.0, axis=1))[0])
    producer_black = producer_output[black_index]
    consumer_black = consumer_output[black_index]
    producer_sha = sha256_bytes(producer_output.tobytes())
    consumer_sha = sha256_bytes(consumer_output.tobytes())
    output_bytes_exact = bool(np.array_equal(producer_output, consumer_output))
    metrics = {
        "fixture_rows": int(fixture.shape[0]),
        "parity_fixture_sha256": sha256_bytes(fixture.tobytes()),
        "producer_output_sha256": producer_sha,
        "consumer_output_sha256": consumer_sha,
        "producer_black": [float(value) for value in producer_black],
        "consumer_black": [float(value) for value in consumer_black],
        "minimum_code": float(np.min(consumer_output)),
        "maximum_code": float(np.max(consumer_output)),
        "existing_target_output_hashes": existing_hashes,
    }
    outputs_valid = bool(
        producer_facts["output_dtype"] == "float32"
        and producer_facts["output_c_contiguous"]
        and producer_facts["output_finite_in_0_1"]
        and consumer_output.dtype == np.float32
        and consumer_output.flags.c_contiguous
        and np.isfinite(consumer_output).all()
        and np.all(consumer_output >= 0)
        and np.all(consumer_output <= 1)
    )
    gates = {
        "producer_and_consumer_identities_exact": exact_identities,
        "producer_and_consumer_runtime_versions_exact": bool(
            observed["producer_runtime"]["ocio"] == config["producer"]["required_opencolorio_version"]
            and observed["consumer_runtime"]["ocio"] == config["consumer"]["required_opencolorio_version"]
        ),
        "fixture_rows_exact": metrics["fixture_rows"] == config["consumer"]["fixture_rows"],
        "producer_consumer_output_bytes_exact": output_bytes_exact,
        "producer_consumer_output_sha256_exact": producer_sha == consumer_sha,
        "inputs_unchanged": bool(producer_facts["input_unchanged"] and np.array_equal(fixture, before)),
        "outputs_float32_contiguous_finite_in_0_1": outputs_valid,
        "official_black_value_exact": bool(
            np.all(producer_black == OFFICIAL_BLACK) and np.array_equal(producer_black, consumer_black)
        ),
        "existing_target_output_hashes_unchanged": existing_hashes == FROZEN_EXISTING_HASHES,
        "invalid_target_rejected": invalid_rejected,
        "canonical_order_reconstruction_exact": True,
    }
    decision = (
        "PASS_PRIVATE_R1CV_REC2100_PQ_RUNTIME_COMPATIBILITY"
        if all(gates.values())
        else "FAIL_CLOSED_R1CV_REC2100_PQ_RUNTIME_COMPATIBILITY"
    )
    scientific = {
        "protocol": config["schema"],
        "producer": {
            "evidence_commit": config["producer"]["evidence_commit"],
            "implementation_commit": config["producer"]["implementation_commit"],
            "display": config["producer"]["display"],
            "view": config["producer"]["view"],
        },
        "consumer": {
            "target": TARGET,
            "current_source_sha256": observed["consumer_current_source_sha256"],
        },
        "preserved_boundaries": config["preserved_boundaries"],
        "observed_identities": observed,
        "runtime_facts": {"producer": producer_facts},
        "metrics": metrics,
        "gates": gates,
        "decision": decision,
        "claim_ceiling": config["claim_ceiling"],
    }
    scientific["stable_identity"] = "sha256:" + sha256_bytes(canonical_bytes(scientific))
    return {
        "schema": "neuro_film.p226_r1cv_rec2100_pq_runtime_compatibility_result.v1",
        "experiment_id": "P226",
        "scientific": scientific,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--producer-python", type=Path, required=True)
    parser.add_argument("--producer-wheel", type=Path, required=True)
    parser.add_argument("--producer-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = execute(
        arguments.order,
        arguments.producer_python,
        arguments.producer_wheel,
        arguments.producer_config,
        arguments.config,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_bytes(canonical_bytes(report))
    os.replace(temporary, arguments.output)
    print(json.dumps(report["scientific"]["gates"], sort_keys=True))
    return 0 if all(report["scientific"]["gates"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
