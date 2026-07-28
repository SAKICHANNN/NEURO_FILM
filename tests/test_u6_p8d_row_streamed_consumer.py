from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.film_physics.profile_consumer import (
    _linear_srgb_to_encoded_row_staged,
    compile_standalone_profile_artifact,
    render_working_image,
    render_working_image_fully_row_streamed,
    render_working_image_row_streamed,
)
from src.film_physics.display_look import (
    build_density_source_context_row_staged,
    build_source_context_display_look,
    build_source_context_display_look_row_streamed,
)
from src.eval.density_witness_frontier import (
    encoded_srgb_to_linear,
    linear_srgb_to_encoded,
)
from src.roll2film.density_domain import (
    DensityDomainNegativePrintOperator,
)
from scripts.pipeline_color_baseline import build_safe_lab_source_context
from src.preprocess.types import SourceProfile, WorkingImage


ROOT = Path(__file__).resolve().parents[1]
P8B = ROOT / "configs/u6_p8b_artifact_only_cpu_consumer_v1.json"


def _working(pixels: np.ndarray) -> WorkingImage:
    return WorkingImage(
        pixels=np.asarray(pixels, dtype=np.float32),
        working_space="linear_srgb_d65",
        transfer_state="scene_linear",
        source_transfer_state="scene_linear",
        source_profile=SourceProfile("raw_metadata", "synthetic"),
        hdr_metadata={},
        orientation_applied=True,
        alpha_policy="absent",
        bit_depth_in=32,
        source_path=Path("synthetic.scene-linear"),
    )


def test_row_staged_scene_oetf_is_float_exact() -> None:
    scene = np.random.default_rng(2026072917).random(
        (257, 131, 3), dtype=np.float32
    )
    reference = linear_srgb_to_encoded(scene.astype(np.float64))
    for tile_rows in (1, 31, 128, 509):
        staged = _linear_srgb_to_encoded_row_staged(
            scene, tile_rows=tile_rows
        )
        assert np.array_equal(reference, staged)


def test_artifact_consumer_row_stream_is_float_exact() -> None:
    config = json.loads(P8B.read_text(encoding="utf-8"))
    artifact = compile_standalone_profile_artifact(
        root=ROOT, config=config
    )
    working = _working(
        np.random.default_rng(2026072909).random(
            (129, 131, 3), dtype=np.float32
        )
    )
    reference, _ = render_working_image(artifact, working)
    forward, forward_receipt = render_working_image_row_streamed(
        artifact, working, tile_rows=31, order="forward"
    )
    reverse, reverse_receipt = render_working_image_row_streamed(
        artifact, working, tile_rows=47, order="reverse"
    )
    assert np.array_equal(reference, forward)
    assert np.array_equal(reference, reverse)
    assert (
        forward_receipt["output"]["array_sha256"]
        == reverse_receipt["output"]["array_sha256"]
    )
    assert forward_receipt["output"]["array_sha256"] == (
        hashlib.sha256(
            memoryview(np.ascontiguousarray(forward)).cast("B")
        ).hexdigest()
    )
    assert forward_receipt["execution"]["seam_rows"] == [31, 62, 93, 124]
    assert reverse_receipt["execution"]["seam_rows"] == [47, 94]


def test_display_look_base_and_residual_row_stream_are_float_exact() -> None:
    config = json.loads(P8B.read_text(encoding="utf-8"))
    artifact = compile_standalone_profile_artifact(
        root=ROOT, config=config
    )
    payload = artifact["component_payloads"][
        "ao6-source-context-display-look"
    ]
    source = np.random.default_rng(2026072910).random((67, 71, 3))
    input_values = np.random.default_rng(2026072911).random(
        (67, 71, 3)
    )
    reference = build_source_context_display_look(
        payload, source
    )(input_values)
    streamed = build_source_context_display_look_row_streamed(
        payload, source, tile_rows=17
    )(input_values)
    assert np.array_equal(reference, streamed)


def test_fully_row_streamed_profile_is_float_exact() -> None:
    config = json.loads(P8B.read_text(encoding="utf-8"))
    artifact = compile_standalone_profile_artifact(
        root=ROOT, config=config
    )
    working = _working(
        np.random.default_rng(2026072912).random(
            (129, 131, 3), dtype=np.float32
        )
    )
    reference, _ = render_working_image(artifact, working)
    forward, forward_receipt = (
        render_working_image_fully_row_streamed(
            artifact, working, tile_rows=31, order="forward"
        )
    )
    reverse, reverse_receipt = (
        render_working_image_fully_row_streamed(
            artifact, working, tile_rows=47, order="reverse"
        )
    )
    assert np.array_equal(reference, forward)
    assert np.array_equal(reference, reverse)
    assert (
        forward_receipt["output"]["array_sha256"]
        == reverse_receipt["output"]["array_sha256"]
    )


def test_density_source_context_row_staging_is_exact() -> None:
    config = json.loads(P8B.read_text(encoding="utf-8"))
    artifact = compile_standalone_profile_artifact(
        root=ROOT, config=config
    )
    payload = artifact["component_payloads"][
        "ao6-source-context-display-look"
    ]
    source = np.random.default_rng(2026072913).random((67, 71, 3))
    base = payload["base"]
    operator = DensityDomainNegativePrintOperator.from_dict(
        base["density_operator"]
    )
    full_density = linear_srgb_to_encoded(
        operator.apply(
            encoded_srgb_to_linear(
                np.asarray(source, dtype=np.float32).astype(np.float64)
            ),
            strength=float(base["density_strength"]),
        )
    )
    reference = build_safe_lab_source_context(full_density)
    staged = build_density_source_context_row_staged(
        payload, source, tile_rows=17
    )
    assert staged == reference
