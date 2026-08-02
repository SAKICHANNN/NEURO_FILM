from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.kodak_250d_granularity_amplitude_profile import (
    GranularityAmplitudeCompilerError,
    compile_and_evaluate,
    load_contract,
)
from src.film_physics.granularity_amplitude import GranularityAmplitudeProfile
from src.film_physics.manufacturer_characteristic import (
    ManufacturerCharacteristicPrior,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4ax_kodak_250d_granularity_amplitude_profile_v1.json"
PARENT_REPORT = (
    ROOT
    / "outputs/experiments/u6_p4aw_kodak_250d_same_sheet_granularity_v1/report_a.json"
)


def test_contract_rejects_spatial_fields(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["compiler"]["spatial_psf_or_nps_fields_allowed"] = True
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(GranularityAmplitudeCompilerError, match="contract drift"):
        load_contract(path)


@pytest.mark.skipif(not PARENT_REPORT.is_file(), reason="P4AW report unavailable")
def test_exact_amplitude_profile_compiles_and_replays() -> None:
    first_bundle, first_report = compile_and_evaluate(load_contract(CONTRACT), ROOT)
    second_bundle, second_report = compile_and_evaluate(load_contract(CONTRACT), ROOT)
    assert first_bundle == second_bundle
    assert first_report == second_report
    assert first_report["automatic_pass"]
    assert first_report["decision"] == "retain_amplitude_only_profile"
    assert all(first_report["gate_results"].values())
    assert first_report["maximum_absolute_sigma_replay_error"] <= 1e-15
    assert first_report["profile_id"] == (
        "d8d6d15fa08b21e2fa34b72d73c6393c19015267ec72c0bdcb1dba93a1c9cd25"
    )
    assert first_report["stable_evidence_id"] == (
        "9595540018f9dd6cf7b2c288eb4d8d356df4c7778267713ce41b23e66ca7c722"
    )
    encoded = json.dumps(first_bundle, sort_keys=True).lower()
    assert all(token not in encoded for token in ("psf", "mtf", "nps", "scanner"))


def test_profile_rejects_wrong_prior_and_domain() -> None:
    bundle_payload = json.loads(
        (ROOT / "outputs/u6_p2q_kodak_250d_characteristic_prior/bundle_run1.json").read_text(
            encoding="utf-8"
        )
    )
    prior = ManufacturerCharacteristicPrior.from_dict(bundle_payload["prior"])
    profile = GranularityAmplitudeProfile(
        characteristic_prior_identity=prior.identity(),
        source_evidence_id="1" * 64,
        channel_floor_variance={"red": 1e-5, "green": 2e-5, "blue": 3e-5},
        shared_amplitude=0.001,
    )
    with pytest.raises(ValueError, match="outside"):
        profile.evaluate_channel(
            prior, "red", np.asarray([prior.curves[0].domain[0] - 0.1])
        )
    wrong = prior.to_dict()
    wrong["source_evidence_id"] = "2" * 64
    with pytest.raises(ValueError, match="identity mismatch"):
        profile.evaluate_channel(
            ManufacturerCharacteristicPrior.from_dict(wrong),
            "red",
            np.asarray([prior.curves[0].domain[0]]),
        )


def test_profile_roundtrip_preserves_identity() -> None:
    profile = GranularityAmplitudeProfile(
        characteristic_prior_identity="a" * 64,
        source_evidence_id="b" * 64,
        channel_floor_variance={"red": 1e-5, "green": 2e-5, "blue": 3e-5},
        shared_amplitude=0.001,
    )
    canonical_payload = json.loads(json.dumps(profile.to_dict(), sort_keys=True))
    replay = GranularityAmplitudeProfile.from_dict(canonical_payload)
    assert replay.to_dict() == profile.to_dict()
    assert replay.identity() == profile.identity()
