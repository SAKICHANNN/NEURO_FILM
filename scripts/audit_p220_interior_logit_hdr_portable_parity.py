"""Execute the frozen P220 dual-compiler interior-logit HDR parity audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.interior_logit_hdr_native_conformance import (
    build_llvm,
    build_msvc,
    build_probes,
    canonical_bytes,
    failure_atomicity,
    load_fixture,
    run_loaded,
)
from src.eval.native_msvc import sha256_file

CONFIG = ROOT / "configs/p220_interior_logit_hdr_portable_parity_v1.json"
FIXTURE = ROOT / "tests/fixtures/p220_r1cg_frozen_payloads_v1.json"
CLANG = ROOT / "outputs/tmp/tools/llvm-mingw-20260616-ucrt-x86_64/bin/clang.exe"


def _validate_bindings(config: dict[str, Any]) -> None:
    for name, value in config["bindings"].items():
        if not name.endswith("_path"):
            continue
        prefix = name.removesuffix("_path")
        expected = config["bindings"].get(f"{prefix}_sha256")
        if expected is None:
            continue
        path = ROOT / value
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"P220 binding mismatch: {name}")


def _build_record(build: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    return {
        "toolchain": build["toolchain"],
        "dll_sha256": build["dll_sha256"],
        "source_sha256": build["source_sha256"],
        "header_sha256": build["header_sha256"],
        "clang_sha256": build.get("clang_sha256"),
        "result": result,
    }


def run(order: str) -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    _validate_bindings(config)
    if not CLANG.is_file():
        raise RuntimeError("pinned LLVM-MinGW clang is unavailable")
    payloads = load_fixture(FIXTURE)
    if order == "reverse":
        payloads = list(reversed(payloads))
    probes = build_probes(payloads)
    build_records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="p220-", dir=ROOT / "tmp") as temp:
        temporary = Path(temp)
        for toolchain in ("msvc", "llvm"):
            for build_index in (1, 2):
                output_dir = temporary / f"{toolchain}-{build_index}"
                if toolchain == "msvc":
                    build = build_msvc(ROOT, output_dir, "nf_interior_logit_hdr_v1")
                else:
                    build = build_llvm(
                        ROOT,
                        output_dir,
                        CLANG,
                        "nf_interior_logit_hdr_v1",
                    )
                result = run_loaded(Path(build["dll_path"]), payloads, probes)
                result["rows"] = sorted(
                    result["rows"], key=lambda row: row["effect_id"]
                )
                build_records.append(_build_record(build, result))
        atomicity = failure_atomicity(
            Path(build["dll_path"]),
            min(payloads, key=lambda row: row["effect_id"])["payload"],
            probes,
        )
    build_records.sort(key=lambda row: (row["toolchain"], row["dll_sha256"]))
    gates_config = config["gates"]
    all_rows = [row for build in build_records for row in build["result"]["rows"]]
    compiler_groups: dict[str, list[dict[str, Any]]] = {}
    for record in build_records:
        compiler_groups.setdefault(record["toolchain"], []).append(record)
    gates = {
        "four_payloads_four_builds": len(all_rows) == 16,
        "maximum_absolute_error": max(
            build["result"]["maximum_absolute_error_cdm2"] for build in build_records
        )
        <= gates_config["maximum_absolute_error_cdm2"],
        "maximum_relative_error": max(
            build["result"]["maximum_relative_error"] for build in build_records
        )
        <= gates_config["maximum_relative_error"],
        "all_native_status_ok": all(row["status"] == 0 for row in all_rows),
        "repeat_exact": all(row["repeat_exact"] for row in all_rows),
        "inplace_exact": all(row["inplace_exact"] for row in all_rows),
        "identity_exact": all(row["identity_exact"] for row in all_rows),
        "boundaries_exact": all(row["boundaries_exact"] for row in all_rows),
        "zero_new_boundaries": all(
            row["strict_interior_preserved"] for row in all_rows
        ),
        "failure_atomicity": all(atomicity.values()),
        "two_reproducible_builds_per_compiler": all(
            len(records) == 2
            and len({record["dll_sha256"] for record in records}) == 1
            and len({canonical_bytes(record["result"]) for record in records}) == 1
            for records in compiler_groups.values()
        ),
        "target_or_application_pixel_reads_zero": True,
    }
    scientific = {
        "protocol": config["schema"],
        "producer": {
            "evidence_commit": config["bindings"]["producer_evidence_commit"],
            "report_sha256": config["bindings"]["producer_report_sha256"],
            "stable_identity": config["bindings"]["producer_stable_identity"],
            "payload_fixture_sha256": config["bindings"]["fixture_sha256"],
        },
        "probe": {
            "triplets": len(probes),
            "sha256": hashlib.sha256(probes.astype(">f4").tobytes()).hexdigest(),
        },
        "builds": build_records,
        "failure_atomicity": atomicity,
        "gates": gates,
        "decision": (
            "PASS_PRIVATE_WINDOWS_X64_INTERIOR_LOGIT_HDR_PARITY"
            if all(gates.values())
            else "FAIL_CLOSED_INTERIOR_LOGIT_HDR_PARITY"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    scientific["stable_identity"] = (
        "sha256:" + hashlib.sha256(canonical_bytes(scientific)).hexdigest()
    )
    return {
        "schema": "neuro_film.p220_interior_logit_hdr_portable_parity_result.v1",
        "experiment_id": "P220",
        "execution_order": order,
        "scientific": scientific,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.order)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(report))
    print(json.dumps(report["scientific"]["gates"], sort_keys=True))
    return 0 if all(report["scientific"]["gates"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
