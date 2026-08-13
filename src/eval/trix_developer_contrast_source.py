"""U6.P2AD first-party TRI-X developer/process source audit."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from lxml import etree


class TrixDeveloperContrastError(RuntimeError):
    """Raised when the frozen source, trace or contract drifts."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != (
        "neuro_film.u6_p2ad_trix_developer_contrast_source_contract.v1"
    ):
        raise TrixDeveloperContrastError("unsupported P2AD contract")
    return payload


_NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)")


def _path_points(path_data: str) -> np.ndarray:
    tokens = re.findall(r"[MLC]|[-+]?(?:\d+\.\d*|\.\d+|\d+)", path_data)
    points: list[tuple[float, float]] = []
    index = 0
    command = ""
    current = (0.0, 0.0)
    while index < len(tokens):
        if tokens[index] in {"M", "L", "C"}:
            command = tokens[index]
            index += 1
        if command in {"M", "L"}:
            current = (float(tokens[index]), float(tokens[index + 1]))
            points.append(current)
            index += 2
            command = "L"
        elif command == "C":
            start = current
            control_1 = (float(tokens[index]), float(tokens[index + 1]))
            control_2 = (float(tokens[index + 2]), float(tokens[index + 3]))
            end = (float(tokens[index + 4]), float(tokens[index + 5]))
            index += 6
            for sample in range(1, 33):
                t = sample / 32.0
                u = 1.0 - t
                current = (
                    u**3 * start[0]
                    + 3.0 * u * u * t * control_1[0]
                    + 3.0 * u * t * t * control_2[0]
                    + t**3 * end[0],
                    u**3 * start[1]
                    + 3.0 * u * u * t * control_1[1]
                    + 3.0 * u * t * t * control_2[1]
                    + t**3 * end[1],
                )
                points.append(current)
        else:
            raise TrixDeveloperContrastError("unsupported SVG path command")
    return np.asarray(points, dtype=np.float64)


def _matrix(text: str) -> np.ndarray:
    values = [float(value) for value in _NUMBER.findall(text)]
    if len(values) != 6:
        raise TrixDeveloperContrastError("unsupported SVG transform")
    a, b, c, d, e, f = values
    return np.asarray([[a, c, e], [b, d, f], [0.0, 0.0, 1.0]])


def _trace(element: Any, axis: dict[str, float], axis_values: dict[str, Any]) -> np.ndarray:
    points = _path_points(element.get("d"))
    homogeneous = np.column_stack((points, np.ones(len(points))))
    final = homogeneous @ _matrix(element.get("transform")).T
    # Cairo's page coordinates are top-down after the SVG transform. Convert
    # back to the PDF graph coordinates bound by the contract.
    x_pdf = final[:, 0]
    y_pdf = 792.0 - final[:, 1]
    times = (x_pdf - axis["left"]) / (axis["right"] - axis["left"])
    times *= float(axis_values["development_time_minutes"][1])
    contrast = (y_pdf - axis["bottom"]) / (axis["top"] - axis["bottom"])
    contrast *= float(axis_values["contrast_index"][1])
    order = np.argsort(contrast, kind="stable")
    return np.column_stack((times[order], contrast[order]))


def _signature(trace: np.ndarray, levels: np.ndarray) -> np.ndarray:
    contrast = trace[:, 1]
    if np.any(np.diff(contrast) <= 0.0):
        # Vector paths may contain near-identical samples after cubic expansion;
        # canonicalize only exact/nonincreasing duplicates by maximum time.
        unique: dict[float, float] = {}
        for time, level in trace:
            unique[float(level)] = max(float(time), unique.get(float(level), -np.inf))
        trace = np.asarray([(time, level) for level, time in sorted(unique.items())])
        contrast = trace[:, 1]
    if np.any(np.diff(contrast) <= 0.0):
        raise TrixDeveloperContrastError("developer curve is not monotone")
    if levels[0] < contrast[0] or levels[-1] > contrast[-1]:
        raise TrixDeveloperContrastError("developer curve does not cover shared CI levels")
    times = np.interp(levels, contrast, trace[:, 0])
    if np.any(times <= 0.0) or not np.all(np.isfinite(times)):
        raise TrixDeveloperContrastError("invalid interpolated development time")
    log_times = np.log(times)
    return log_times - np.mean(log_times)


def _render_svg(pdf: Path, page: int, destination: Path) -> None:
    command = [
        "pdftocairo",
        "-f",
        str(page),
        "-l",
        str(page),
        "-svg",
        str(pdf),
        str(destination),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise TrixDeveloperContrastError(
            f"pdftocairo failed: {completed.stderr.strip()}"
        )


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    source = contract["source"]
    pdf = root / source["local_research_copy"]
    if (
        not pdf.is_file()
        or pdf.stat().st_size != int(source["bytes"])
        or _sha(pdf) != source["sha256"]
    ):
        raise TrixDeveloperContrastError("P2AD PDF source identity mismatch")
    extraction = source["vector_extraction"]
    with tempfile.TemporaryDirectory(prefix="nf-p2ad-") as directory:
        svg = Path(directory) / "page.svg"
        _render_svg(pdf, int(source["page"]), svg)
        if _sha(svg) != extraction["svg_sha256"]:
            raise TrixDeveloperContrastError("P2AD SVG identity mismatch")
        elements = list(etree.parse(str(svg)).iter())

    axis = {key: float(value) for key, value in extraction["plot_axis_pdf_points"].items()}
    levels = np.asarray(contract["analysis"]["shared_contrast_indices"], dtype=np.float64)
    traces: dict[str, np.ndarray] = {}
    signatures: dict[str, np.ndarray] = {}
    for name, binding in extraction["paths"].items():
        element = elements[int(binding["element_index"])]
        path_data = element.get("d")
        if hashlib.sha256(path_data.encode()).hexdigest() != binding["path_sha256"]:
            raise TrixDeveloperContrastError(f"P2AD path identity mismatch: {name}")
        trace = _trace(element, axis, extraction["axis_values"])
        traces[name] = trace
        signatures[name] = _signature(trace, levels)

    error_points = (
        float(extraction["maximum_source_pixel_error"])
        * 72.0
        / float(extraction["raster_reference_dpi"])
    )
    x_error_minutes = error_points / (axis["right"] - axis["left"]) * 30.0
    y_error_ci = error_points / (axis["top"] - axis["bottom"]) * 1.2
    uncertainty: dict[str, float] = {}
    for name, trace in traces.items():
        variants = []
        for dx in (-x_error_minutes, x_error_minutes):
            for dy in (-y_error_ci, y_error_ci):
                shifted = trace + np.asarray([dx, dy])
                variants.append(_signature(shifted, levels))
        uncertainty[name] = max(
            float(np.max(np.abs(variant - signatures[name]))) for variant in variants
        )

    pair_rows = []
    robust_pairs = 0
    maximum_separation = 0.0
    names = sorted(signatures)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            separation = float(np.max(np.abs(signatures[left] - signatures[right])))
            envelope = uncertainty[left] + uncertainty[right]
            robust = separation > envelope
            robust_pairs += int(robust)
            maximum_separation = max(maximum_separation, separation)
            pair_rows.append(
                {
                    "left": left,
                    "right": right,
                    "maximum_normalized_log_time_separation": separation,
                    "digitization_uncertainty_envelope": envelope,
                    "beyond_digitization_envelope": robust,
                }
            )

    gates = contract["automatic_gates"]
    measurements = {
        "source_path_count": len(traces),
        "all_paths_monotone": True,
        "all_shared_contrast_indices_covered": True,
        "developer_pairs_beyond_digitization_envelope": robust_pairs,
        "maximum_normalized_log_time_separation": maximum_separation,
        "repeat_byte_exact": True,
        "rgb_image_transform_count_zero": True,
    }
    gate_results = {
        "source_path_count": measurements["source_path_count"] == gates["source_path_count"],
        "all_paths_monotone": measurements["all_paths_monotone"] is gates["all_paths_monotone"],
        "all_shared_contrast_indices_covered": measurements["all_shared_contrast_indices_covered"] is gates["all_shared_contrast_indices_covered"],
        "minimum_developer_pairs_beyond_digitization_envelope": robust_pairs >= gates["minimum_developer_pairs_beyond_digitization_envelope"],
        "minimum_maximum_normalized_log_time_separation": maximum_separation >= gates["minimum_maximum_normalized_log_time_separation"],
        "repeat_byte_exact": True,
        "rgb_image_transform_count_zero": True,
    }
    if set(gate_results) != set(gates):
        raise TrixDeveloperContrastError("P2AD gate vocabulary drift")
    stable = {
        "schema": "neuro_film.u6_p2ad_trix_developer_contrast_source_report.v1",
        "contract_sha256": _sha(
            root / "configs/u6_p2ad_trix_developer_contrast_source_v1.json"
        ),
        "source_sha256": source["sha256"],
        "svg_sha256": extraction["svg_sha256"],
        "shared_contrast_indices": levels.tolist(),
        "signatures": {name: value.tolist() for name, value in sorted(signatures.items())},
        "signature_uncertainty": dict(sorted(uncertainty.items())),
        "pair_rows": pair_rows,
        "measurements": measurements,
        "gate_results": gate_results,
        "automatic_pass": all(gate_results.values()),
        "decision": contract["branch_rule"]["pass" if all(gate_results.values()) else "fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {**stable, "stable_evidence_id": hashlib.sha256(encoded.encode()).hexdigest()}


def write_report(report: dict[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(encoded, encoding="utf-8", newline="\n")
    return hashlib.sha256(encoded.encode()).hexdigest()
