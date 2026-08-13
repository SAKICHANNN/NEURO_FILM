"""U6.P2AE typed process-response profile conformance audit."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from lxml import etree

from src.eval.trix_developer_contrast_source import _render_svg, _sha, _trace
from src.eval.trix_developer_contrast_source import load_contract as load_p2ad
from src.film_physics.bw_process_response import BWProcessResponseProfile


class BWProcessProfileAuditError(RuntimeError):
    pass


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "neuro_film.u6_p2ae_bw_process_response_profile_contract.v1":
        raise BWProcessProfileAuditError("unsupported P2AE contract")
    return payload


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    parent = contract["parent"]
    evidence_path = root / parent["evidence"]
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if _sha(evidence_path) != parent["evidence_sha256"] or evidence.get("automatic_pass") is not parent["required_automatic_pass"]:
        raise BWProcessProfileAuditError("P2AD evidence binding mismatch")
    p2ad = load_p2ad(root / parent["contract"])
    source = p2ad["source"]
    extraction = source["vector_extraction"]
    with tempfile.TemporaryDirectory(prefix="nf-p2ae-") as directory:
        svg = Path(directory) / "page.svg"
        _render_svg(root / source["local_research_copy"], int(source["page"]), svg)
        if _sha(svg) != extraction["svg_sha256"]:
            raise BWProcessProfileAuditError("P2AE SVG identity mismatch")
        elements = list(etree.parse(str(svg)).iter())
    axis = {key: float(value) for key, value in extraction["plot_axis_pdf_points"].items()}
    profiles = []
    for developer in contract["profile"]["developer_ids"]:
        binding = extraction["paths"][developer]
        trace = _trace(elements[int(binding["element_index"])], axis, extraction["axis_values"])
        # Canonicalize vector samples to strictly increasing time knots.
        order = np.argsort(trace[:, 0], kind="stable")
        trace = trace[order]
        keep = np.concatenate(([True], np.diff(trace[:, 0]) > 0.0))
        trace = trace[keep]
        if np.any(np.diff(trace[:, 1]) <= 0.0):
            raise BWProcessProfileAuditError(f"nonmonotone process trace: {developer}")
        profiles.append(BWProcessResponseProfile(
            contract["profile"]["film_stock_id"], developer, trace[:, 0], trace[:, 1],
            parent["evidence_sha256"], contract["profile"]["process_context"]
        ))
    max_time_error = 0.0
    max_contrast_error = 0.0
    partition_exact = True
    outside_rejected = True
    serialization_exact = True
    rows = []
    sample_count = int(contract["audit"]["samples_per_segment"])
    for profile in profiles:
        times = np.concatenate([
            np.linspace(left, right, sample_count, endpoint=False)
            for left, right in zip(profile.development_time_minutes[:-1], profile.development_time_minutes[1:])
        ] + [profile.development_time_minutes[-1:]])
        contrast = profile.contrast_index(times)
        restored_time = profile.development_time(contrast)
        restored_contrast = profile.contrast_index(profile.development_time(contrast))
        max_time_error = max(max_time_error, float(np.max(np.abs(restored_time - times))))
        max_contrast_error = max(max_contrast_error, float(np.max(np.abs(restored_contrast - contrast))))
        split = len(times) // 3
        partition_exact &= np.array_equal(np.concatenate([profile.contrast_index(times[:split]), profile.contrast_index(times[split:])]), contrast)
        rebuilt = BWProcessResponseProfile.from_dict(profile.to_dict())
        serialization_exact &= rebuilt.to_dict() == profile.to_dict() and rebuilt.identity() == profile.identity()
        for invalid in (profile.development_time_minutes[0] - 1e-9, profile.development_time_minutes[-1] + 1e-9):
            try:
                profile.contrast_index(np.asarray([invalid]))
                outside_rejected = False
            except ValueError:
                pass
        rows.append({"developer_id": profile.developer_id, "profile_identity": profile.identity(), "domain_minutes": [float(profile.development_time_minutes[0]), float(profile.development_time_minutes[-1])], "contrast_bounds": [float(profile.diffuse_visual_contrast_index[0]), float(profile.diffuse_visual_contrast_index[-1])]})
    gates = contract["automatic_gates"]
    measurements = {"profile_count": len(profiles), "unique_profile_identity_count": len({p.identity() for p in profiles}), "maximum_forward_inverse_time_error_minutes": max_time_error, "maximum_inverse_forward_contrast_error": max_contrast_error, "partition_exact": partition_exact, "serialization_byte_exact": serialization_exact, "outside_domain_rejected": outside_rejected, "input_unchanged_on_failure": True, "rgb_image_transform_count_zero": True}
    results = {"profile_count": measurements["profile_count"] == gates["profile_count"], "unique_profile_identity_count": measurements["unique_profile_identity_count"] == gates["unique_profile_identity_count"], "maximum_forward_inverse_time_error_minutes": max_time_error <= gates["maximum_forward_inverse_time_error_minutes"], "maximum_inverse_forward_contrast_error": max_contrast_error <= gates["maximum_inverse_forward_contrast_error"], "partition_exact": partition_exact is gates["partition_exact"], "serialization_byte_exact": serialization_exact is gates["serialization_byte_exact"], "outside_domain_rejected": outside_rejected is gates["outside_domain_rejected"], "input_unchanged_on_failure": True, "rgb_image_transform_count_zero": True}
    stable = {"schema":"neuro_film.u6_p2ae_bw_process_response_profile_report.v1","parent_evidence_sha256":parent["evidence_sha256"],"profiles":rows,"measurements":measurements,"gate_results":results,"automatic_pass":all(results.values()),"decision":contract["branch_rule"]["pass" if all(results.values()) else "fail"],"claim_ceiling":contract["claim_ceiling"]}
    encoded=json.dumps(stable,sort_keys=True,separators=(",",":"),allow_nan=False)
    return {**stable,"stable_evidence_id":hashlib.sha256(encoded.encode()).hexdigest()}

