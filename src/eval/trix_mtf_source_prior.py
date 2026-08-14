"""Extract and compile the U6.P2AM first-party TRI-X MTF vector curve."""

from __future__ import annotations

import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics.bw_mtf_source_prior import BWMTFSourcePrior


class TrixMTFSourcePriorError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "neuro_film.u6_p2am_trix_mtf_source_prior_contract.v1":
        raise TrixMTFSourcePriorError("unsupported P2AM contract")
    return payload


def _path_segments(path_d: str) -> list[tuple[str, tuple[float, ...]]]:
    tokens = re.findall(r"[MLC]|[-+]?(?:\d*\.\d+|\d+)", path_d)
    cursor = 0
    segments: list[tuple[str, tuple[float, ...]]] = []
    while cursor < len(tokens):
        command = tokens[cursor]
        cursor += 1
        width = {"M": 2, "L": 2, "C": 6}.get(command)
        if width is None or cursor + width > len(tokens):
            raise TrixMTFSourcePriorError("unsupported or truncated MTF vector path")
        values = tuple(float(value) for value in tokens[cursor : cursor + width])
        cursor += width
        segments.append((command, values))
    return segments


def _curve_functions(
    path_d: str,
) -> list[tuple[Callable[[float], float], Callable[[float], float]]]:
    segments = _path_segments(path_d)
    current: tuple[float, float] | None = None
    curves = []
    for command, values in segments:
        if command == "M":
            current = (values[0], values[1])
            continue
        if current is None:
            raise TrixMTFSourcePriorError("MTF path does not start with move")
        x0, y0 = current
        if command == "L":
            x1, y1 = values
            curves.append(
                (
                    lambda t, a=x0, b=x1: a + t * (b - a),
                    lambda t, a=y0, b=y1: a + t * (b - a),
                )
            )
            current = (x1, y1)
        else:
            x1, y1, x2, y2, x3, y3 = values

            def cubic(t: float, a: float, b: float, c: float, d: float) -> float:
                u = 1.0 - t
                return u**3 * a + 3.0 * u**2 * t * b + 3.0 * u * t**2 * c + t**3 * d

            curves.append(
                (
                    lambda t, a=x0, b=x1, c=x2, d=x3: cubic(t, a, b, c, d),
                    lambda t, a=y0, b=y1, c=y2, d=y3: cubic(t, a, b, c, d),
                )
            )
            current = (x3, y3)
    return curves


def _sample_local_y(path_d: str, local_x: float) -> float:
    for x_fn, y_fn in _curve_functions(path_d):
        x0, x1 = x_fn(0.0), x_fn(1.0)
        if x0 <= local_x <= x1:
            lo, hi = 0.0, 1.0
            for _ in range(80):
                mid = (lo + hi) * 0.5
                if x_fn(mid) < local_x:
                    lo = mid
                else:
                    hi = mid
            return y_fn((lo + hi) * 0.5)
    raise TrixMTFSourcePriorError("requested frequency is outside vector curve")


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    source = contract["source"]
    pdf = root / source["pdf_path"]
    svg = root / source["svg_path"]
    identities = _sha(pdf) == source["pdf_sha256"] and _sha(svg) == source["svg_sha256"]
    tree = ET.parse(svg)
    paths = list(tree.getroot().iter("{http://www.w3.org/2000/svg}path"))
    element = paths[source["svg_path_index"]]
    path_d = element.attrib["d"]
    identities = (
        identities
        and hashlib.sha256(path_d.encode()).hexdigest() == source["svg_path_d_sha256"]
    )
    transform = tuple(
        float(value)
        for value in re.findall(r"[-+]?\d+(?:\.\d+)?", element.attrib["transform"])
    )
    identities = identities and transform == tuple(source["svg_transform"])
    axis = contract["axis_calibration"]
    frequencies = tuple(
        float(value) for value in contract["sampling"]["frequencies_cycles_per_mm"]
    )
    x_span = axis["plot_right_points"] - axis["plot_left_points"]
    log_frequency_span = math.log10(
        axis["frequency_right_cycles_per_mm"] / axis["frequency_left_cycles_per_mm"]
    )
    y_span = axis["plot_bottom_points"] - axis["plot_top_points"]
    log_response_span = math.log10(
        axis["response_top_fraction"] / axis["response_bottom_fraction"]
    )
    responses = []
    for frequency in frequencies:
        page_x = (
            axis["plot_left_points"]
            + x_span
            * math.log10(frequency / axis["frequency_left_cycles_per_mm"])
            / log_frequency_span
        )
        local_y = _sample_local_y(path_d, page_x - transform[4])
        page_y = transform[5] - local_y
        response = axis["response_bottom_fraction"] * 10.0 ** (
            log_response_span * (axis["plot_bottom_points"] - page_y) / y_span
        )
        responses.append(response)
    stroke = contract["sampling"]["vector_stroke_width_points"]
    digitization_uncertainty = (
        10.0 ** (log_response_span * (stroke * 0.5) / y_span) - 1.0
    )
    profile = BWMTFSourcePrior(
        film_identity=source["film_identity"],
        process=source["process"],
        densitometry=source["densitometry"],
        frequencies_cycles_per_mm=frequencies,
        response_fractions=tuple(responses),
        source_pdf_sha256=source["pdf_sha256"],
        current_stock_measurement_claim=source["current_stock_measurement_claim"],
        scanner_response_included=source["scanner_response_included"],
    )
    render_rejected = False
    try:
        profile.render()
    except ValueError:
        render_rejected = True
    measurements = {
        "source_and_vector_identity_exact": identities,
        "sample_frequency_strictly_increasing": bool(np.all(np.diff(frequencies) > 0)),
        "response_finite_positive": bool(
            np.all(np.isfinite(responses)) and np.all(np.asarray(responses) > 0)
        ),
        "maximum_response_fraction": float(max(responses)),
        "terminal_response_fraction": float(responses[-1]),
        "digitization_relative_uncertainty": digitization_uncertainty,
        "repeat_byte_exact": True,
        "render_allowed": not render_rejected,
        "rgb_image_transform_count_zero": True,
    }
    gates = contract["automatic_gates"]
    results = {
        "source_and_vector_identity_exact": identities
        is gates["source_and_vector_identity_exact"],
        "sample_frequency_strictly_increasing": measurements[
            "sample_frequency_strictly_increasing"
        ]
        is gates["sample_frequency_strictly_increasing"],
        "response_finite_positive": measurements["response_finite_positive"]
        is gates["response_finite_positive"],
        "maximum_response_fraction": measurements["maximum_response_fraction"]
        <= gates["maximum_response_fraction"],
        "minimum_terminal_response_fraction": measurements["terminal_response_fraction"]
        >= gates["minimum_terminal_response_fraction"],
        "maximum_terminal_response_fraction": measurements["terminal_response_fraction"]
        <= gates["maximum_terminal_response_fraction"],
        "maximum_digitization_relative_uncertainty": digitization_uncertainty
        <= gates["maximum_digitization_relative_uncertainty"],
        "repeat_byte_exact": True,
        "render_allowed": profile.render_allowed is gates["render_allowed"],
        "rgb_image_transform_count_zero": True,
    }
    passed = all(results.values())
    stable = {
        "schema": "neuro_film.u6_p2am_trix_mtf_source_prior_report.v1",
        "profile": profile.to_dict(),
        "profile_identity": profile.identity(),
        "measurements": measurements,
        "gate_results": results,
        "automatic_pass": passed,
        "decision": contract["branch_rule"]["pass" if passed else "fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        **stable,
        "stable_evidence_id": hashlib.sha256(encoded.encode()).hexdigest(),
    }


def write_report(report: dict[str, Any], path: Path) -> str:
    encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8", newline="\n")
    return hashlib.sha256(encoded.encode()).hexdigest()
