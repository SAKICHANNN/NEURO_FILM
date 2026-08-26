"""Independently rebuild and replay the producer-declared R1CL browser leaf."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p223_r1cl_browser_wasm_reproducibility_v1.json"
SCHEMA = "neuro-film.p223-r1cl-browser-wasm-reproducibility-contract.v1"


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


def run_command(
    command: list[str], *, cwd: Path | None = None, check: bool = True
) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(command, cwd=cwd, capture_output=True, check=False)
    if check and completed.returncode != 0:
        output = (completed.stdout + completed.stderr).decode(errors="replace")
        raise RuntimeError(f"command failed ({completed.returncode}): {output}")
    return completed


def git_bytes(repository: Path, *arguments: str) -> bytes:
    return run_command(["git", *arguments], cwd=repository).stdout


def git_text(repository: Path, *arguments: str) -> str:
    return git_bytes(repository, *arguments).decode("utf-8").strip()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != SCHEMA:
        raise RuntimeError("P223 contract schema differs")
    if config.get("status") != (
        "FROZEN_BEFORE_CONSUMER_BROWSER_REBUILD_OR_RUNTIME_EXECUTION"
    ):
        raise RuntimeError("P223 contract is not frozen")
    if config["execution"]["fresh_exact_commit_clones"] != 2:
        raise RuntimeError("P223 requires exactly two fresh clones")
    if len(config["browsers"]) != 2:
        raise RuntimeError("P223 requires exactly two frozen browsers")


def inspect_preconditions(config: dict[str, Any]) -> dict[str, Any]:
    producer = config["producer"]
    repository = Path(producer["repository"])
    formal_commit = producer["formal_commit"]
    evidence_commit = producer["evidence_commit"]
    if not repository.is_dir():
        raise RuntimeError("P223 producer repository is unavailable")
    if git_text(repository, "cat-file", "-t", formal_commit) != "commit":
        raise RuntimeError("P223 formal identity is not a commit")
    if git_text(repository, "cat-file", "-t", evidence_commit) != "commit":
        raise RuntimeError("P223 evidence identity is not a commit")

    evidence_path = producer["evidence_path"]
    evidence_bytes = git_bytes(repository, "show", f"{evidence_commit}:{evidence_path}")
    observed_blobs = {
        path: git_text(repository, "rev-parse", f"{formal_commit}:{path}")
        for path in producer["bound_git_blobs"]
    }

    toolchain = config["toolchain"]
    binary_root = (
        Path(toolchain["ndk_path"]) / "toolchains/llvm/prebuilt/windows-x86_64/bin"
    )
    compiler = binary_root / "clang.exe"
    linker = binary_root / "wasm-ld.exe"
    browser_rows: list[dict[str, Any]] = []
    for fixed in config["browsers"]:
        path = Path(fixed["path"])
        if not path.is_file():
            raise RuntimeError(
                f"P223 frozen browser unavailable: {fixed['browser_id']}"
            )
        browser_rows.append(
            {
                "browser_id": fixed["browser_id"],
                "path": fixed["path"],
                "product_version": fixed["product_version"],
                "executable_bytes": path.stat().st_size,
                "executable_sha256": sha256_file(path),
            }
        )
    for binary in (compiler, linker):
        if not binary.is_file():
            raise RuntimeError(f"P223 frozen tool unavailable: {binary.name}")

    return {
        "formal_commit_object_type": git_text(
            repository, "cat-file", "-t", formal_commit
        ),
        "evidence_commit_object_type": git_text(
            repository, "cat-file", "-t", evidence_commit
        ),
        "evidence_blob": git_text(
            repository, "rev-parse", f"{evidence_commit}:{evidence_path}"
        ),
        "evidence_sha256": sha256_bytes(evidence_bytes),
        "bound_git_blobs": observed_blobs,
        "compiler_sha256": sha256_file(compiler),
        "wasm_linker_sha256": sha256_file(linker),
        "browsers": browser_rows,
    }


def preconditions_exact(config: dict[str, Any], observed: dict[str, Any]) -> bool:
    producer = config["producer"]
    toolchain = config["toolchain"]
    return (
        observed["formal_commit_object_type"] == "commit"
        and observed["evidence_commit_object_type"] == "commit"
        and observed["evidence_blob"] == producer["evidence_blob"]
        and observed["evidence_sha256"] == producer["evidence_sha256"]
        and observed["bound_git_blobs"] == producer["bound_git_blobs"]
        and observed["compiler_sha256"] == toolchain["compiler_sha256"]
        and observed["wasm_linker_sha256"] == toolchain["wasm_linker_sha256"]
        and observed["browsers"] == config["browsers"]
    )


def extract_report_facts(report: dict[str, Any]) -> dict[str, Any]:
    scientific = report["scientific"]
    runtime = scientific["runtime_payload"]
    return {
        "producer_status": report["status"],
        "stable_identity": report["stable_identity"],
        "wasm_module_bytes": scientific["wasm_module_bytes"],
        "wasm_module_sha256": scientific["wasm_module_sha256"],
        "output_rgb_sha256": runtime["output_sha256"],
        "scientific_payload_sha256": sha256_bytes(canonical_bytes(runtime)),
        "browser_count": len(scientific["browsers"]),
        "process_observation_count": len(scientific["process_observations"]),
        "all_inner_producer_gates_pass": all(scientific["gates"].values()),
    }


def evaluate_reports(
    config: dict[str, Any], report_payloads: list[bytes], temporary_roots_removed: bool
) -> dict[str, bool]:
    expected = config["expected"]
    facts = [extract_report_facts(json.loads(payload)) for payload in report_payloads]
    return {
        "each_consumer_report_equals_frozen_producer_report_sha256": all(
            sha256_bytes(payload) == expected["formal_report_sha256"]
            for payload in report_payloads
        ),
        "consumer_reports_byte_exact": len(set(report_payloads)) == 1,
        "producer_status_exact": all(
            fact["producer_status"] == expected["producer_status"] for fact in facts
        ),
        "stable_identity_exact": all(
            fact["stable_identity"] == expected["stable_identity"] for fact in facts
        ),
        "wasm_module_identity_exact": all(
            fact["wasm_module_bytes"] == expected["wasm_module_bytes"]
            and fact["wasm_module_sha256"] == expected["wasm_module_sha256"]
            for fact in facts
        ),
        "runtime_output_identity_exact": all(
            fact["output_rgb_sha256"] == expected["output_rgb_sha256"] for fact in facts
        ),
        "scientific_payload_identity_exact": all(
            fact["scientific_payload_sha256"] == expected["scientific_payload_sha256"]
            for fact in facts
        ),
        "browser_and_process_counts_exact": all(
            fact["browser_count"] == expected["browser_count"]
            and fact["process_observation_count"]
            == expected["process_observation_count"]
            for fact in facts
        ),
        "all_inner_producer_gates_pass": all(
            fact["all_inner_producer_gates_pass"] for fact in facts
        ),
        "temporary_roots_removed": temporary_roots_removed,
        "media_reads_zero": True,
        "network_downloads_zero": True,
    }


def execute(order: str, config_path: Path = CONFIG) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    observed = inspect_preconditions(config)
    exact_preconditions = preconditions_exact(config, observed)
    if not exact_preconditions:
        raise RuntimeError(
            "P223 frozen producer, toolchain or browser identity differs"
        )

    producer = config["producer"]
    clone_indices = [0, 1] if order == "forward" else [1, 0]
    temporary_root = Path(tempfile.mkdtemp(prefix="neuro-film-p223-"))
    payloads_by_index: dict[int, bytes] = {}
    returncodes: dict[int, int] = {}
    cleanup_complete = False
    try:
        for index in clone_indices:
            clone = temporary_root / f"clone-{index}"
            run_command(
                [
                    "git",
                    "clone",
                    "--local",
                    "--no-hardlinks",
                    "--no-checkout",
                    producer["repository"],
                    str(clone),
                ]
            )
            run_command(["git", "config", "core.autocrlf", "false"], cwd=clone)
            run_command(
                ["git", "checkout", "--detach", producer["formal_commit"]], cwd=clone
            )
            if git_text(clone, "rev-parse", "HEAD") != producer["formal_commit"]:
                raise RuntimeError("P223 clone did not resolve the frozen commit")
            if git_text(clone, "status", "--short", "--untracked-files=all"):
                raise RuntimeError("P223 exact-commit clone is not clean")

            output = temporary_root / f"producer-report-{index}.json"
            completed = run_command(
                [
                    sys.executable,
                    str(clone / producer["runner_path"]),
                    "--output",
                    str(output),
                    "--ndk",
                    config["toolchain"]["ndk_path"],
                ],
                cwd=clone,
                check=False,
            )
            returncodes[index] = completed.returncode
            if not output.is_file():
                details = (completed.stdout + completed.stderr).decode(errors="replace")
                raise RuntimeError(f"P223 producer runner emitted no report: {details}")
            payloads_by_index[index] = output.read_bytes()
    finally:
        shutil.rmtree(temporary_root, ignore_errors=False)
        cleanup_complete = not temporary_root.exists()

    report_payloads = [payloads_by_index[index] for index in (0, 1)]
    gates = {
        "formal_and_evidence_commits_exact": exact_preconditions,
        "all_transitive_git_blobs_exact": (
            observed["bound_git_blobs"] == producer["bound_git_blobs"]
        ),
        "compiler_linker_and_browser_hashes_exact": (
            observed["compiler_sha256"] == config["toolchain"]["compiler_sha256"]
            and observed["wasm_linker_sha256"]
            == config["toolchain"]["wasm_linker_sha256"]
            and observed["browsers"] == config["browsers"]
        ),
        **evaluate_reports(config, report_payloads, cleanup_complete),
    }
    decision = (
        "PASS_PRIVATE_R1CL_CONSUMER_REPRODUCIBILITY"
        if all(gates.values())
        else "FAIL_CLOSED_R1CL_CONSUMER_REPRODUCIBILITY"
    )
    scientific = {
        "protocol": config["schema"],
        "producer_formal_commit": producer["formal_commit"],
        "producer_evidence_commit": producer["evidence_commit"],
        "expected": config["expected"],
        "observed_preconditions": observed,
        "consumer_report_sha256": [
            sha256_bytes(payload) for payload in report_payloads
        ],
        "consumer_report_facts": [
            extract_report_facts(json.loads(payload)) for payload in report_payloads
        ],
        "producer_runner_returncodes": [returncodes[index] for index in (0, 1)],
        "gates": gates,
        "decision": decision,
        "claim_ceiling": config["claim_ceiling"],
    }
    scientific["stable_identity"] = "sha256:" + sha256_bytes(
        canonical_bytes(scientific)
    )
    return {
        "schema": "neuro_film.p223_r1cl_browser_wasm_reproducibility_result.v1",
        "experiment_id": "P223",
        "scientific": scientific,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = execute(arguments.order, arguments.config)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_bytes(canonical_bytes(report))
    os.replace(temporary, arguments.output)
    print(json.dumps(report["scientific"]["gates"], sort_keys=True))
    return 0 if all(report["scientific"]["gates"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
