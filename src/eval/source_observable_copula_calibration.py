"""P4HN dual-compiler evaluation of source-observable copula calibration."""

from __future__ import annotations

import hashlib
import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.native_histogram_copula_transport import _build_llvm
from src.eval.native_msvc import build_msvc_c11_dll, sha256_file
from src.film_physics.calibrated_native_histogram_copula import (
    apply_source_observable_calibrated_copula,
)
from src.film_physics.native_histogram_copula import (
    load_native_histogram_copula_library,
)

SCHEMA = "neuro-film.u6-p4hn-source-observable-copula-calibration-contract.v1"


def _fixture(row: dict[str, Any]) -> np.ndarray:
    count = int(row["height"]) * int(row["width"])
    rng = np.random.default_rng(int(row["field_seed"]))
    latent = rng.standard_normal((count, 4))
    fields = np.empty((count, 3), dtype=np.float32)
    fields[:, 0] = latent[:, 0] + 0.19 * latent[:, 3] ** 2
    fields[:, 1] = 0.23 * latent[:, 0] + latent[:, 1] - 0.11 * latent[:, 3] ** 2
    fields[:, 2] = -0.17 * latent[:, 0] + 0.29 * latent[:, 1] + latent[:, 2]
    return fields


def _stable_payload(report: dict[str, Any]) -> dict[str, Any]:
    stable = json.loads(json.dumps(report))
    stable.pop("stable_evidence_id", None)
    for toolchain in stable["toolchains"].values():
        toolchain.pop("dll_path", None)
        toolchain.pop("dll_sha256", None)
        toolchain.pop("compiler_output", None)
    return stable


def evaluate(
    *, root: Path, contract: dict[str, Any], output_dir: Path, clang: Path
) -> dict[str, Any]:
    if contract.get("schema") != SCHEMA:
        raise ValueError("unsupported P4HN contract")
    for name in ("p4hm_evidence", "profile"):
        binding = contract["parents"][name]
        path = root / binding["path"]
        if not path.is_file() or sha256_file(path) != binding["sha256"]:
            raise ValueError(f"P4HN parent drift: {name}")
    evidence = json.loads(
        (root / contract["parents"]["p4hm_evidence"]["path"]).read_text("utf-8")
    )
    if evidence.get("decision") != contract["parents"]["p4hm_evidence"][
        "required_decision"
    ]:
        raise ValueError("P4HN P4HM decision drift")
    for key, relative in {
        "native_header_sha256": "native/film_physics/nf_histogram_copula_f32_v1.h",
        "native_source_sha256": "native/film_physics/nf_histogram_copula_f32_v1.c",
    }.items():
        if sha256_file(root / relative) != contract["parents"][key]:
            raise ValueError(f"P4HN {key} drift")
    profile = json.loads(
        (root / contract["parents"]["profile"]["path"]).read_text("utf-8")
    )
    correlation = np.asarray(profile["execution"]["correlation_matrix"], np.float64)
    candidate = contract["candidate"]
    msvc = build_msvc_c11_dll(
        root=root,
        output_dir=output_dir,
        source_relative="native/film_physics/nf_histogram_copula_f32_v1.c",
        header_relative="native/film_physics/nf_histogram_copula_f32_v1.h",
        basename="nf_histogram_copula_f32_msvc_v1",
    )
    llvm = _build_llvm(root, output_dir, clang)
    rows: dict[str, list[dict[str, Any]]] = {"msvc": [], "llvm_mingw": []}
    outputs: dict[tuple[str, int], np.ndarray] = {}
    for backend, build in {"msvc": msvc, "llvm_mingw": llvm}.items():
        library = load_native_histogram_copula_library(Path(build["dll_path"]))
        for index, fixture_row in enumerate(contract["confirmation_fixtures"]):
            fields = _fixture(fixture_row)
            first, receipt = apply_source_observable_calibrated_copula(
                library,
                fields,
                profile_correlation=correlation,
                rank_bins=int(candidate["rank_bins"]),
                iterations=int(candidate["iterations"]),
            )
            second, second_receipt = apply_source_observable_calibrated_copula(
                library,
                fields,
                profile_correlation=correlation,
                rank_bins=int(candidate["rank_bins"]),
                iterations=int(candidate["iterations"]),
            )
            outputs[(backend, index)] = first
            final_diag = receipt["final_native_diagnostics"]
            rows[backend].append(
                {
                    "fixture": fixture_row,
                    "output_sha256": hashlib.sha256(first.tobytes()).hexdigest(),
                    "repeat_exact": bool(np.array_equal(first, second)),
                    "receipt_repeat_exact": receipt == second_receipt,
                    "finite_unit_output": bool(
                        np.all(np.isfinite(first))
                        and np.all(first >= 0.0)
                        and np.all(first <= 1.0)
                    ),
                    "uniform_mean_error": float(
                        np.max(
                            np.abs(
                                np.asarray(final_diag["output_uniform_mean"], float)
                                - 0.5
                            )
                        )
                    ),
                    "workspace_bytes": int(final_diag["workspace_bytes"]),
                    **receipt,
                }
            )
    cross = max(
        float(
            np.max(
                np.abs(
                    outputs[("msvc", index)].astype(np.float64)
                    - outputs[("llvm_mingw", index)].astype(np.float64)
                )
            )
        )
        for index in range(len(contract["confirmation_fixtures"]))
    )
    flat = [row for backend_rows in rows.values() for row in backend_rows]
    gates = contract["automatic_gates"]
    checks = {
        "target_correlation": max(
            row["iteration_maximum_correlation_errors"][-1] for row in flat
        )
        <= gates["maximum_target_correlation_error"],
        "uniform_mean": max(row["uniform_mean_error"] for row in flat)
        <= gates["maximum_uniform_mean_error"],
        "cross_compiler": cross <= gates["maximum_cross_compiler_absolute_error"],
        "workspace": max(row["workspace_bytes"] for row in flat)
        <= gates["maximum_workspace_bytes"],
        "latent_eigenvalue": min(
            row["minimum_final_latent_eigenvalue"] for row in flat
        )
        >= gates["minimum_latent_correlation_eigenvalue"],
        "latent_correction": max(
            row["maximum_absolute_latent_correction"] for row in flat
        )
        <= gates["maximum_absolute_latent_correction"],
        "strict_error_reduction": all(
            all(left > right for left, right in pairwise(errors))
            for errors in (
                row["iteration_maximum_correlation_errors"] for row in flat
            )
        ),
        "repeat": all(
            row["repeat_exact"] and row["receipt_repeat_exact"] for row in flat
        ),
        "finite_unit": all(row["finite_unit_output"] for row in flat),
        "native_calls": all(
            row["native_call_count"] == int(candidate["iterations"]) for row in flat
        ),
    }
    passed = all(checks.values())
    report: dict[str, Any] = {
        "schema": "neuro-film.u6-p4hn-source-observable-copula-calibration-result.v1",
        "contract_sha256": hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "profile_bundle_sha256": contract["parents"]["profile"]["bundle_sha256"],
        "toolchains": {"msvc": msvc, "llvm_mingw": llvm},
        "results": rows,
        "maximum_cross_compiler_absolute_error": cross,
        "gates": checks,
        "automatic_pass": passed,
        "decision": contract["decision_if_pass"]
        if passed
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    report["stable_evidence_id"] = hashlib.sha256(
        json.dumps(_stable_payload(report), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return report


__all__ = ["evaluate"]
