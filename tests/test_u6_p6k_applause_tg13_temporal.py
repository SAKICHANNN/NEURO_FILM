from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from src.eval.applause_tg13_temporal import (
    ApplauseTg13Error,
    extract_step_vector_from_array,
    open_fits_image,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (ROOT / "configs/u6_p6k_applause_tg13_temporal_v1.json").read_text(
        encoding="utf-8"
    )
)
DECISION = json.loads(
    (ROOT / "configs/u6_p6k_applause_tg13_temporal_decision_v1.json").read_text(
        encoding="utf-8"
    )
)


def _fits_bytes(values: np.ndarray, *, bzero: int = 65535) -> bytes:
    cards = [
        "SIMPLE  =                    T",
        "BITPIX  =                   16",
        "NAXIS   =                    2",
        f"NAXIS1  = {values.shape[1]:20d}",
        f"NAXIS2  = {values.shape[0]:20d}",
        "BSCALE  =         1.000000E+00",
        f"BZERO   = {bzero:20d}",
        "END",
    ]
    header = b"".join(card.ljust(80).encode("ascii") for card in cards)
    header += b" " * ((-len(header)) % 2880)
    payload = np.asarray(values, dtype=">i2").tobytes()
    return header + payload + b"\0" * ((-len(payload)) % 2880)


def test_contract_freezes_unseen_confirmatory_rows_and_claim_ceiling() -> None:
    validate_config(CONFIG)
    confirmatory = [
        row for row in CONFIG["rows"] if row["role"].startswith("confirmatory_")
    ]
    assert len(confirmatory) == 6
    assert all("sha256" not in row for row in confirmatory)
    assert CONFIG["scanner_calibration_allowed"] is False
    assert CONFIG["stock_or_emulsion_fitting_allowed"] is False


def test_step_measurement_recovers_monotone_plateaus() -> None:
    image = np.zeros((8, 40), dtype=np.int16)
    image[:, :10] = -100
    image[:, 10:20] = -50
    image[:, 20:30] = 0
    image[:, 30:] = 100
    vector, span = extract_step_vector_from_array(
        image,
        edges=[0, 10, 20, 30, 40],
        row_start=1,
        row_stop=7,
        half_width=2,
        bzero=65535.0,
    )
    assert span == 200.0
    assert np.allclose(vector, [0.0, 0.25, 0.5, 1.0])


def test_minimal_fits_parser_is_fail_closed(tmp_path: Path) -> None:
    values = np.arange(12, dtype=np.int16).reshape(3, 4)
    path = tmp_path / "small.fits"
    path.write_bytes(_fits_bytes(values))
    contract = {
        "bitpix": 16,
        "naxis1": 4,
        "naxis2": 3,
        "bscale": 1.0,
        "bzero": 65535.0,
    }
    image, header = open_fits_image(path, contract)
    assert header["BZERO"] == 65535
    assert np.array_equal(np.asarray(image), values)
    bad = dict(contract, naxis1=5)
    with pytest.raises(ApplauseTg13Error, match="NAXIS1"):
        open_fits_image(path, bad)


def test_contract_rejects_support_and_download_drift() -> None:
    drifted = deepcopy(CONFIG)
    drifted["rows"] = drifted["rows"][:-1]
    with pytest.raises(ApplauseTg13Error, match="fourteen"):
        validate_config(drifted)
    drifted = deepcopy(CONFIG)
    drifted["download"]["maximum_total_bytes"] = 300_000_000
    with pytest.raises(ApplauseTg13Error, match="download boundary"):
        validate_config(drifted)


def test_decision_binds_exact_contract_and_keeps_calibration_closed() -> None:
    contract_bytes = (
        ROOT / "configs/u6_p6k_applause_tg13_temporal_v1.json"
    ).read_bytes()
    assert (
        hashlib.sha256(contract_bytes).hexdigest()
        == DECISION["evidence"]["contract"]["sha256"]
    )
    assert DECISION["automatic_pass"] is True
    assert DECISION["support"]["confirmatory_files"] == 6
    assert DECISION["metrics"]["maximum_confirmatory_correct_prototype_rmse"] < 0.02
    assert any(
        "scanner calibration" in item for item in DECISION["forbidden"]
    )
    assert "film stock, emulsion, exposure or development response" in DECISION[
        "forbidden"
    ]
