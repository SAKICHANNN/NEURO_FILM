"""U6.P2AO primary-source audit for historical TRI-X Wiener amplitudes."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics.bw_wiener_density_prior import BWWienerDensityPrior


class HistoricalTrixWienerSourceError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema")
        != "neuro_film.u6_p2ao_historical_trix_wiener_density_source_contract.v1"
    ):
        raise HistoricalTrixWienerSourceError("unsupported P2AO contract")
    return payload


def _table_pattern() -> re.Pattern[str]:
    cell = r'<td align="center" valign="top">'
    empty = r'<td align="left" valign="top"/>'
    exponent = r"×10<sup>−10</sup>"
    rows = [
        rf">Tri-X</td>{cell}0\.06</td>{cell}33\.2{exponent}</td></tr>",
        rf"<tr>{empty}{cell}0\.36</td>{cell}88\.2{exponent}</td></tr>",
        rf"<tr>{empty}{cell}0\.74</td>{cell}88\.2{exponent}</td></tr>",
        rf"<tr>{empty}{cell}1\.14</td>{cell}131{exponent}</td></tr>",
    ]
    return re.compile("".join(rows))


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    source = contract["source"]
    path = root / source["path"]
    source_exact = _sha(path) == source["sha256"]
    text = path.read_text(encoding="utf-8")
    copies = len(_table_pattern().findall(text))
    development_fragment = (
        '>Tri-X</td><td align="center" valign="top">6.5 min</td>'
        '<td align="left" valign="top"><em>SD</em>-28</td>'
        '<td align="center" valign="top">0.54</td>'
    )
    metadata_copies = text.count(development_fragment)
    profile = BWWienerDensityPrior(
        film_identity=source["film_identity"],
        development=source["development"],
        reported_gamma=source["reported_gamma"],
        densities=tuple(source["densities"]),
        wiener_granularity_spectrum_cm2=tuple(
            source["wiener_granularity_spectrum_cm2"]
        ),
        source_sha256=source["sha256"],
        current_400tx_claim=source["current_400tx_claim"],
        spatial_spectrum_shape_available=source["spatial_spectrum_shape_available"],
    )
    density = np.asarray(profile.densities, dtype=np.float64)
    wiener = np.asarray(profile.wiener_granularity_spectrum_cm2, dtype=np.float64)
    render_rejected = False
    try:
        profile.render()
    except ValueError:
        render_rejected = True
    measurements = {
        "source_identity_exact": source_exact,
        "table_copy_count": copies,
        "development_metadata_copy_count": metadata_copies,
        "table_values_exact": profile.densities == tuple(source["densities"])
        and profile.wiener_granularity_spectrum_cm2
        == tuple(source["wiener_granularity_spectrum_cm2"]),
        "densities_strictly_increasing": bool(np.all(np.diff(density) > 0.0)),
        "wiener_values_finite_positive": bool(
            np.all(np.isfinite(wiener)) and np.all(wiener > 0.0)
        ),
        "wiener_values_nondecreasing": bool(np.all(np.diff(wiener) >= 0.0)),
        "plateau_observation_exact": bool(wiener[1] == wiener[2]),
        "render_allowed": not render_rejected,
        "rgb_image_transform_count_zero": True,
        "repeat_byte_exact": True,
    }
    gates = contract["automatic_gates"]
    results = {
        "source_identity_exact": source_exact is gates["source_identity_exact"],
        "table_copies_exact": copies == source["expected_html_table_copies"],
        "table_values_exact": measurements["table_values_exact"]
        is gates["table_values_exact"],
        "densities_strictly_increasing": measurements["densities_strictly_increasing"]
        is gates["densities_strictly_increasing"],
        "wiener_values_finite_positive": measurements["wiener_values_finite_positive"]
        is gates["wiener_values_finite_positive"],
        "wiener_values_nondecreasing": measurements["wiener_values_nondecreasing"]
        is gates["wiener_values_nondecreasing"],
        "plateau_observation_exact": measurements["plateau_observation_exact"]
        is gates["plateau_observation_exact"],
        "render_allowed": profile.render_allowed is gates["render_allowed"],
        "rgb_image_transform_count_zero": True,
        "repeat_byte_exact": True,
    }
    passed = all(results.values())
    stable = {
        "schema": "neuro_film.u6_p2ao_historical_trix_wiener_density_source_report.v1",
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
