#!/usr/bin/env python3
"""Evaluate one exact producer wheel on the frozen P44 A1/A4/A5 gates."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.color_match import (  # noqa: E402
    ContextInvarianceBatchMetrics,
    ContextInvarianceMetrics,
    PhotographicSafetyBatchMetrics,
    PhotographicSafetyMetrics,
    adjudicate_promotion,
    aggregate_known_operator_samples,
    canonical_sha256,
    evaluate_context_invariance_outputs,
    evaluate_known_operator_batch,
    evaluate_photographic_probe,
    invoke_dpct_package_v1,
    invoke_dpct_package_v2,
    load_dpct_invocation_profile_v2,
    make_context_invariance_probes,
    make_photographic_probe,
    prepare_working_image_match_view,
)
from src.color_match.dpct_invocation_profile import (  # noqa: E402
    DpctInvocationProfileV2,
)
from src.color_match.evaluation import (  # noqa: E402
    KnownOperatorSampleMetrics,
)
from src.preprocess import WorkingImage, load_working_image  # noqa: E402


PROTOCOL = "neuro-film.dpct-invocation-promotion-evaluation.v1"
SAMPLE_IDS = ("01", "02", "03", "05", "07", "09")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--invocation-profile", type=Path)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(directory: Path, sample_id: str) -> Path:
    matches = tuple(
        path for path in directory.glob(f"{sample_id}_*") if path.is_file()
    )
    if len(matches) != 1:
        raise ValueError(
            f"{directory} must contain exactly one {sample_id}_* file"
        )
    return matches[0].resolve()


def _working(source: WorkingImage, pixels: np.ndarray) -> WorkingImage:
    return WorkingImage(
        pixels=np.asarray(pixels, dtype=np.float32),
        working_space=source.working_space,
        transfer_state=source.transfer_state,
        source_transfer_state=source.source_transfer_state,
        source_profile=source.source_profile,
        hdr_metadata=dict(source.hdr_metadata),
        orientation_applied=source.orientation_applied,
        alpha_policy=source.alpha_policy,
        bit_depth_in=32,
        source_path=source.source_path,
        warnings=list(source.warnings),
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _load_progress(path: Path, contract_id: str) -> dict[str, Any]:
    if not path.is_file():
        return {
            "protocol": PROTOCOL,
            "contract_id": contract_id,
            "known_rows": {},
            "photographic_rows": {},
            "context_rows": {},
        }
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("protocol") != PROTOCOL
        or value.get("contract_id") != contract_id
    ):
        raise ValueError("existing P44 progress belongs to another contract")
    return value


def _load_invocation_profile(
    path: Path | None,
) -> DpctInvocationProfileV2 | None:
    if path is None:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invocation profile is unavailable or invalid") from exc
    return load_dpct_invocation_profile_v2(value)


def _invoke(
    source: WorkingImage,
    reference_prepared: Any,
    *,
    row_id: str,
    wheel: Path,
    python: Path,
    scratch: Path,
    invocation_profile: DpctInvocationProfileV2 | None,
) -> tuple[WorkingImage, dict[str, Any]]:
    source_prepared = prepare_working_image_match_view(source)
    arguments = {
        "source": source_prepared,
        "reference": reference_prepared,
        "intent_id": hashlib.sha256(row_id.encode("utf-8")).hexdigest(),
        "wheel_path": wheel,
        "python_executable": python,
        "scratch_directory": scratch,
    }
    if invocation_profile is None:
        outcome = invoke_dpct_package_v1(**arguments)
    else:
        outcome = invoke_dpct_package_v2(
            profile=invocation_profile,
            **arguments,
        )
    if outcome.status != "candidate" or outcome.candidate is None:
        raise RuntimeError(f"P44 row {row_id} did not yield a candidate")
    candidate = outcome.candidate
    return (
        _working(source, candidate.prepared_output.pixels),
        {
            "request_id": outcome.request_id,
            "response_id": outcome.response_id,
            "producer_bundle_id": candidate.aliases.bundle_id,
            "producer_diagnostics_id": candidate.aliases.diagnostics_id,
            "producer_result_id": candidate.aliases.result_id,
            "consumer_transform_id": candidate.transform.transform_id,
            "consumer_receipt_id": candidate.prepared_output.receipt.receipt_id,
            "output_pixel_sha256": (
                candidate.prepared_output.receipt.output_view.pixel_sha256
            ),
            "out_of_gamut_fraction": (
                candidate.diagnostics.out_of_gamut_fraction
            ),
            "clipping_fraction": candidate.diagnostics.clipping_fraction,
            "projected_fraction": candidate.diagnostics.projected_fraction,
        },
    )


def _photo_batch(
    rows: list[PhotographicSafetyMetrics],
) -> PhotographicSafetyBatchMetrics:
    failed = sum(not row.passed for row in rows)
    return PhotographicSafetyBatchMetrics(
        samples=tuple(rows),
        passed_recipe_count=len(rows) - failed,
        failed_recipe_count=failed,
        maximum_neutral_chroma_p95=max(
            row.neutral_chroma_p95 for row in rows
        ),
        maximum_tone_reversal_fraction=max(
            row.tone_reversal_fraction for row in rows
        ),
        largest_tone_reversal_delta_l=min(
            row.tone_largest_reversal_delta_l for row in rows
        ),
        maximum_tone_plateau_fraction=max(
            row.tone_plateau_fraction for row in rows
        ),
        maximum_new_boundary_fraction=max(
            row.new_boundary_fraction for row in rows
        ),
        maximum_semantic_hue_rotation_p95_degrees=max(
            row.semantic_hue_rotation_p95_degrees for row in rows
        ),
    )


def _context_batch(
    rows: list[ContextInvarianceMetrics],
) -> ContextInvarianceBatchMetrics:
    failed = sum(not row.passed for row in rows)
    return ContextInvarianceBatchMetrics(
        samples=tuple(rows),
        passed_recipe_count=len(rows) - failed,
        failed_recipe_count=failed,
        maximum_delta_e76_median=max(
            row.delta_e76_median for row in rows
        ),
        maximum_delta_e76_p95=max(row.delta_e76_p95 for row in rows),
        maximum_delta_e76=max(row.delta_e76_maximum for row in rows),
    )


def main() -> int:
    args = _parser().parse_args()
    invocation_profile = _load_invocation_profile(
        args.invocation_profile
    )
    main_root = args.main_root.resolve()
    source_dir = (
        main_root
        / "outputs/color_baseline/"
        "velvia50_rawpixls20_s0p50_gamutsafe/inputs"
    )
    target_dir = (
        main_root / "outputs/eval/baseline_current/velvia_50/after"
    )
    sources = {key: _resolve(source_dir, key) for key in SAMPLE_IDS}
    targets = {key: _resolve(target_dir, key) for key in SAMPLE_IDS}
    contract = {
        "protocol": PROTOCOL,
        "sample_ids": list(SAMPLE_IDS),
        "wheel_sha256": _sha256(args.wheel),
        "source_sha256": {key: _sha256(path) for key, path in sources.items()},
        "target_sha256": {key: _sha256(path) for key, path in targets.items()},
        "fit_semantics": "source-and-reference-per-row",
        "targets_at_inference": False,
        "thresholds": "existing-promotion-defaults",
    }
    if invocation_profile is not None:
        contract["invocation_profile_id"] = invocation_profile.profile_id
        contract["capability_id"] = invocation_profile.capability_id
        contract["producer_stable_commit"] = (
            invocation_profile.producer_stable_commit
        )
    contract_id = canonical_sha256(contract)
    progress_path = args.output.with_suffix(".progress.json")
    progress = _load_progress(progress_path, contract_id)

    for reference_id in SAMPLE_IDS:
        reference = load_working_image(targets[reference_id])
        reference_prepared = prepare_working_image_match_view(reference)
        for source_id in SAMPLE_IDS:
            if source_id == reference_id:
                continue
            row_id = f"{reference_id}->{source_id}"
            if row_id in progress["known_rows"]:
                continue
            source = load_working_image(sources[source_id])
            target = load_working_image(targets[source_id])
            candidate, invocation = _invoke(
                source,
                reference_prepared,
                row_id=f"known:{row_id}",
                wheel=args.wheel,
                python=args.python,
                scratch=args.scratch,
                invocation_profile=invocation_profile,
            )
            metrics = evaluate_known_operator_batch(
                [source],
                [target],
                [candidate],
                sample_ids=[row_id],
            ).samples[0]
            progress["known_rows"][row_id] = {
                "metrics": asdict(metrics),
                "invocation": invocation,
            }
            _atomic_json(progress_path, progress)

        if reference_id not in progress["photographic_rows"]:
            probe = make_photographic_probe()
            candidate, invocation = _invoke(
                probe,
                reference_prepared,
                row_id=f"photographic:{reference_id}",
                wheel=args.wheel,
                python=args.python,
                scratch=args.scratch,
                invocation_profile=invocation_profile,
            )
            progress["photographic_rows"][reference_id] = {
                "metrics": asdict(evaluate_photographic_probe(probe, candidate)),
                "invocation": invocation,
            }
            _atomic_json(progress_path, progress)

        if reference_id not in progress["context_rows"]:
            first, second, region = make_context_invariance_probes()
            first_candidate, first_invocation = _invoke(
                first,
                reference_prepared,
                row_id=f"context-first:{reference_id}",
                wheel=args.wheel,
                python=args.python,
                scratch=args.scratch,
                invocation_profile=invocation_profile,
            )
            second_candidate, second_invocation = _invoke(
                second,
                reference_prepared,
                row_id=f"context-second:{reference_id}",
                wheel=args.wheel,
                python=args.python,
                scratch=args.scratch,
                invocation_profile=invocation_profile,
            )
            metrics = evaluate_context_invariance_outputs(
                first_candidate,
                second_candidate,
                shared_region=region,
            )
            progress["context_rows"][reference_id] = {
                "metrics": asdict(metrics),
                "invocations": [first_invocation, second_invocation],
            }
            _atomic_json(progress_path, progress)

    known = aggregate_known_operator_samples(
        KnownOperatorSampleMetrics(**progress["known_rows"][key]["metrics"])
        for key in sorted(progress["known_rows"])
    )
    photographic = _photo_batch(
        [
            PhotographicSafetyMetrics(
                **progress["photographic_rows"][key]["metrics"]
            )
            for key in SAMPLE_IDS
        ]
    )
    context = _context_batch(
        [
            ContextInvarianceMetrics(
                **progress["context_rows"][key]["metrics"]
            )
            for key in SAMPLE_IDS
        ]
    )
    decision = adjudicate_promotion(known, photographic, context)
    stable_rows: dict[str, Any] = {}
    for group_name, group in (
        ("known", progress["known_rows"]),
        ("photographic", progress["photographic_rows"]),
        ("context", progress["context_rows"]),
    ):
        stable_group: dict[str, Any] = {}
        for key, row in group.items():
            invocations = (
                row["invocations"]
                if "invocations" in row
                else [row["invocation"]]
            )
            stable_group[key] = {
                "metrics": row["metrics"],
                "invocations": [
                    {
                        identity_key: invocation[identity_key]
                        for identity_key in (
                            "request_id",
                            "producer_bundle_id",
                            "consumer_transform_id",
                            "output_pixel_sha256",
                        )
                        if identity_key in invocation
                    }
                    for invocation in invocations
                ],
            }
        stable_rows[group_name] = stable_group
    stable_evidence = {
        "protocol": PROTOCOL,
        "contract_id": contract_id,
        "known_operator": asdict(known),
        "photographic_safety": asdict(photographic),
        "context_invariance": asdict(context),
        "promotion_decision": asdict(decision),
        "stable_rows": stable_rows,
    }
    report: dict[str, Any] = {
        "protocol": PROTOCOL,
        "report_id": "",
        "stable_evidence_id": canonical_sha256(stable_evidence),
        "contract": contract,
        "contract_id": contract_id,
        "known_operator": asdict(known),
        "photographic_safety": asdict(photographic),
        "context_invariance": asdict(context),
        "promotion_decision": asdict(decision),
        "visual_review": None,
        "rows": {
            "known": progress["known_rows"],
            "photographic": progress["photographic_rows"],
            "context": progress["context_rows"],
        },
    }
    report["report_id"] = canonical_sha256(
        {key: value for key, value in report.items() if key != "report_id"}
    )
    _atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "report_id": report["report_id"],
                "promotion_status": decision.status,
                "reasons": list(decision.reasons),
                "known_improvement_rate": known.improvement_rate,
                "known_median_improvement": (
                    known.median_improvement_fraction
                ),
                "photo_failed": photographic.failed_recipe_count,
                "context_failed": context.failed_recipe_count,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
