"""Fail-closed evaluator for the frozen U5.R2L0 strength Oracle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import fmean, median
from typing import Any, Mapping


class DensityStrengthOracleError(ValueError):
    """Raised when immutable parent evidence or Oracle structure drifts."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_hashed_json(
    root: Path,
    relative_path: str,
    expected_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise DensityStrengthOracleError("parent path escapes repository root")
    if sha256_file(path) != expected_sha256:
        raise DensityStrengthOracleError(f"parent hash mismatch: {relative_path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DensityStrengthOracleError("parent JSON must be an object")
    return path, value


def _candidate_rows(
    report: Mapping[str, Any],
    candidate_id: str,
    expected_strength: float,
    expected_witness: str,
) -> dict[str, dict[str, Any]]:
    candidate = report.get("candidates", {}).get(candidate_id)
    if not isinstance(candidate, dict):
        raise DensityStrengthOracleError(f"missing candidate: {candidate_id}")
    if str(candidate.get("witness_id")) != expected_witness:
        raise DensityStrengthOracleError(f"witness drift: {candidate_id}")
    if float(candidate.get("strength", -1.0)) != expected_strength:
        raise DensityStrengthOracleError(f"strength drift: {candidate_id}")
    rows: dict[str, dict[str, Any]] = {}
    for raw in candidate.get("per_image", []):
        if not isinstance(raw, dict):
            raise DensityStrengthOracleError("per-image row must be an object")
        sample_id = str(raw.get("sample_id", ""))
        if not sample_id or sample_id in rows:
            raise DensityStrengthOracleError(
                f"missing or duplicate sample: {candidate_id}/{sample_id}"
            )
        rows[sample_id] = dict(raw)
    return rows


def _manifest_rows(
    manifest: Mapping[str, Any],
    candidate_ids: set[str],
) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in manifest.get("records", []):
        if not isinstance(raw, dict):
            raise DensityStrengthOracleError("manifest row must be an object")
        candidate_id = str(raw.get("candidate_id", ""))
        if candidate_id not in candidate_ids:
            continue
        sample_id = str(raw.get("sample_id", ""))
        key = (candidate_id, sample_id)
        if not sample_id or key in rows:
            raise DensityStrengthOracleError(
                f"missing or duplicate manifest row: {key}"
            )
        rows[key] = dict(raw)
    return rows


def evaluate_oracle(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_sha256: str,
    software_commit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate immutable E1 evidence and evaluate the hard strength Oracle."""

    parent = config["parent"]
    report_path, parent_report = _load_hashed_json(
        root,
        str(parent["automatic_report"]),
        str(parent["automatic_report_sha256"]),
    )
    manifest_path, parent_manifest = _load_hashed_json(
        root,
        str(parent["render_manifest"]),
        str(parent["render_manifest_sha256"]),
    )
    if str(parent_manifest.get("frozen_set_sha256", "")) != str(
        parent["frozen_set_sha256"]
    ):
        raise DensityStrengthOracleError("frozen-set hash drift")

    witness = str(config["witness_id"])
    baseline_id = str(config["baseline_candidate_id"])
    challenger_id = str(config["challenger_candidate_id"])
    baseline = _candidate_rows(
        parent_report,
        baseline_id,
        float(config["baseline_strength"]),
        witness,
    )
    challenger = _candidate_rows(
        parent_report,
        challenger_id,
        float(config["challenger_strength"]),
        witness,
    )
    if baseline.keys() != challenger.keys():
        raise DensityStrengthOracleError("candidate sample membership differs")
    expected = int(config["expected_samples"])
    if len(baseline) != expected:
        raise DensityStrengthOracleError(
            f"expected {expected} samples, found {len(baseline)}"
        )

    manifest_rows = _manifest_rows(
        parent_manifest,
        {baseline_id, challenger_id},
    )
    if len(manifest_rows) != 2 * expected:
        raise DensityStrengthOracleError("candidate manifest is incomplete")

    ceiling = float(
        config["oracle"]["maximum_selected_new_hard_clipping_fraction"]
    )
    selected_rows: list[dict[str, Any]] = []
    for sample_id in sorted(baseline):
        base = baseline[sample_id]
        challenge = challenger[sample_id]
        if str(base.get("split")) != str(challenge.get("split")):
            raise DensityStrengthOracleError(f"split drift: {sample_id}")
        challenge_clip = float(challenge["new_hard_clipping_fraction"])
        selected_id = challenger_id if challenge_clip <= ceiling else baseline_id
        selected = challenge if selected_id == challenger_id else base
        manifest_row = manifest_rows[(selected_id, sample_id)]
        output_sha256 = str(manifest_row["output_sha256"])
        if output_sha256 != str(selected["output_sha256"]):
            raise DensityStrengthOracleError(
                f"metric/manifest output hash mismatch: {selected_id}/{sample_id}"
            )
        output_path = (manifest_path.parent / str(manifest_row["output"])).resolve()
        if not output_path.is_relative_to(manifest_path.parent.resolve()):
            raise DensityStrengthOracleError("output path escapes parent directory")
        if sha256_file(output_path) != output_sha256:
            raise DensityStrengthOracleError(
                f"selected output hash mismatch: {selected_id}/{sample_id}"
            )
        split = str(base["split"])
        selected_rows.append(
            {
                "sample_id": sample_id,
                "split": split,
                "selected_candidate_id": selected_id,
                "selected_strength": (
                    float(config["challenger_strength"])
                    if selected_id == challenger_id
                    else float(config["baseline_strength"])
                ),
                "selected_output": output_path.relative_to(root.resolve()).as_posix(),
                "selected_output_sha256": output_sha256,
                "selected_style_delta_e76": float(
                    selected["median_style_delta_e76"]
                ),
                "baseline_style_delta_e76": float(
                    base["median_style_delta_e76"]
                ),
                "style_gain_delta_e76": float(
                    selected["median_style_delta_e76"]
                    - base["median_style_delta_e76"]
                ),
                "selected_non_basic_residual_delta_e76": float(
                    selected["median_non_basic_residual_delta_e76"]
                ),
                "selected_new_hard_clipping_fraction": float(
                    selected["new_hard_clipping_fraction"]
                ),
                "challenger_new_hard_clipping_fraction": challenge_clip,
            }
        )

    gold = [row for row in selected_rows if row["split"] == "gold"]
    stress = [row for row in selected_rows if row["split"] == "stress"]
    if len(gold) != int(config["expected_gold_samples"]):
        raise DensityStrengthOracleError("gold sample count drift")
    if len(stress) != int(config["expected_stress_samples"]):
        raise DensityStrengthOracleError("stress sample count drift")

    gains = [row["style_gain_delta_e76"] for row in selected_rows]
    selected_challenger = sum(
        row["selected_candidate_id"] == challenger_id for row in selected_rows
    )
    aggregates = {
        "challenger_selected_count": selected_challenger,
        "baseline_fallback_count": expected - selected_challenger,
        "challenger_selection_fraction": selected_challenger / expected,
        "mean_style_gain_delta_e76": fmean(gains),
        "median_style_gain_delta_e76": median(gains),
        "gold_mean_style_gain_delta_e76": fmean(
            row["style_gain_delta_e76"] for row in gold
        ),
        "stress_mean_style_gain_delta_e76": fmean(
            row["style_gain_delta_e76"] for row in stress
        ),
        "maximum_selected_new_hard_clipping_fraction": max(
            row["selected_new_hard_clipping_fraction"] for row in selected_rows
        ),
    }
    gates = config["automatic_gates"]
    checks = {
        "challenger_selection_fraction": (
            aggregates["challenger_selection_fraction"]
            >= float(gates["minimum_challenger_selection_fraction"])
        ),
        "mean_style_gain": (
            aggregates["mean_style_gain_delta_e76"]
            >= float(gates["minimum_mean_style_gain_delta_e76"])
        ),
        "gold_mean_style_gain": (
            aggregates["gold_mean_style_gain_delta_e76"]
            >= float(gates["minimum_gold_mean_style_gain_delta_e76"])
        ),
        "median_style_gain": (
            aggregates["median_style_gain_delta_e76"]
            >= float(gates["minimum_median_style_gain_delta_e76"])
        ),
        "selected_clipping": (
            aggregates["maximum_selected_new_hard_clipping_fraction"]
            <= float(gates["maximum_selected_new_hard_clipping_fraction"])
        ),
        "parent_and_selected_output_hashes": True,
        "sample_membership_and_splits": True,
    }
    selected_manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "config_sha256": config_sha256,
        "parent_report_sha256": parent["automatic_report_sha256"],
        "parent_manifest_sha256": parent["render_manifest_sha256"],
        "records": selected_rows,
        "claim_ceiling": config["claim_ceiling"],
    }
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "config_sha256": config_sha256,
        "parent_report": report_path.relative_to(root.resolve()).as_posix(),
        "parent_report_sha256": parent["automatic_report_sha256"],
        "parent_manifest": manifest_path.relative_to(root.resolve()).as_posix(),
        "parent_manifest_sha256": parent["render_manifest_sha256"],
        "sample_count": expected,
        "gold_sample_count": len(gold),
        "stress_sample_count": len(stress),
        "aggregates": aggregates,
        "checks": checks,
        "automatic_checks_passed": all(checks.values()),
        "repeat_report_byte_identity_requires_external_run": True,
        "visual_status": (
            "required"
            if all(checks.values())
            else "forbidden_by_automatic_gate"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    return report, selected_manifest
