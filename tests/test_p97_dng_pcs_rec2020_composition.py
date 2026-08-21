from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.dng_camera_to_pcs_native_conformance import build_probes
from src.preprocess.prophoto_icc import (
    ProPhotoICCError,
    d50_xyz_to_linear_rec2020,
    decode_prophoto_rgb16_to_linear_rec2020,
)


def test_existing_official_romm_decoder_bytes_are_unchanged() -> None:
    profile = Path("data/wide_gamut/icc_official/ISO22028-2_ROMM-RGB.icc").read_bytes()
    codes = np.arange(65536, dtype=np.uint16)
    encoded = np.stack(
        (codes, codes[::-1], np.bitwise_xor(codes, 0x5A5A)), axis=1
    ).reshape(1, -1, 3)
    output = decode_prophoto_rgb16_to_linear_rec2020(encoded, profile)
    assert hashlib.sha256(output.tobytes()).hexdigest() == (
        "379c29f2a231c4297e34ac46abef05380b076739436ccd00c08d1f1bc58aa752"
    )


def test_p94_staged_and_composed_matrices_agree() -> None:
    report = json.loads(
        Path(
            "outputs/eval/p94_dng_forward_matrix_mechanics/outer1_forward.json"
        ).read_text(encoding="utf-8")
    )
    probes = build_probes()
    for row in report["rows"]:
        camera_to_pcs = np.asarray(row["camera_to_pcs"], dtype=np.float64)
        staged = d50_xyz_to_linear_rec2020(probes @ camera_to_pcs.T)
        basis = d50_xyz_to_linear_rec2020(np.eye(3, dtype=np.float64))
        direct = probes @ (basis.T @ camera_to_pcs).T
        np.testing.assert_allclose(staged, direct, atol=5e-15, rtol=0)


def test_pcs_primitive_fails_closed_on_invalid_inputs() -> None:
    with pytest.raises(TypeError, match="numpy ndarray"):
        d50_xyz_to_linear_rec2020([[0.0, 0.0, 0.0]])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="final dimension"):
        d50_xyz_to_linear_rec2020(np.zeros((2, 4), dtype=np.float64))
    invalid = np.zeros((2, 3), dtype=np.float64)
    invalid[1, 2] = np.nan
    with pytest.raises(ProPhotoICCError, match="non-finite"):
        d50_xyz_to_linear_rec2020(invalid)
