from __future__ import annotations

import json
from pathlib import Path
import struct
import zlib

from jsonschema import Draft202012Validator, ValidationError
import numpy as np
import pytest
import tifffile

from src.color_match import (
    REFERENCE_FILE_OUTPUT_METADATA_POLICY_ID,
    attest_reference_file_output_metadata,
    match_reference_files,
    reference_file_output_capabilities,
    reference_file_output_metadata_policy_payload,
)
from src.preprocess import (
    SourceProfile,
    WorkingImage,
    save_rec2020_16_png,
    save_srgb8,
    save_srgb16_png,
    save_srgb16_tiff,
)
from src.preprocess.output_encode import _png_chunk, srgb_icc_profile


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_file_output_metadata_policy_v1.schema.json"
)


def _pixels() -> np.ndarray:
    return np.random.default_rng(16801).uniform(
        0.02,
        0.98,
        size=(19, 23, 3),
    ).astype(np.float32)


def _encode(path: Path, working_space: str, bit_depth: int) -> None:
    pixels = _pixels()
    if working_space == "linear_rec2020":
        save_rec2020_16_png(
            WorkingImage(
                pixels=pixels,
                working_space="linear_rec2020",
                transfer_state="display_linear",
                source_transfer_state="display_referred",
                source_profile=SourceProfile("cicp", "P168 fixture"),
                hdr_metadata={},
                orientation_applied=True,
                alpha_policy="absent",
                bit_depth_in=16,
                source_path=path,
                warnings=[],
            ),
            path,
        )
    elif bit_depth == 8:
        save_srgb8(pixels, path)
    elif path.suffix.casefold() == ".png":
        save_srgb16_png(pixels, path)
    else:
        save_srgb16_tiff(pixels, path)


def _all_advertised_tuples() -> list[tuple[str, int, str, str]]:
    return [
        (
            capability.working_space,
            capability.output_bit_depth,
            extension,
            capability.encoding_profile,
        )
        for capability in reference_file_output_capabilities()
        for extension in capability.extensions
    ]


def test_p168_policy_payload_is_strict_and_schema_valid() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    payload = reference_file_output_metadata_policy_payload()
    validator.validate(payload)
    assert payload["schema_id"] == REFERENCE_FILE_OUTPUT_METADATA_POLICY_ID
    assert payload["source_metadata_copy"] == "none"

    mutated = json.loads(json.dumps(payload))
    mutated["source_metadata_copy"] = "best-effort"
    with pytest.raises(ValidationError):
        validator.validate(mutated)


@pytest.mark.parametrize(
    ("working_space", "bit_depth", "extension", "encoding_profile"),
    _all_advertised_tuples(),
)
def test_p168_every_advertised_encoder_output_attests(
    tmp_path: Path,
    working_space: str,
    bit_depth: int,
    extension: str,
    encoding_profile: str,
) -> None:
    output = tmp_path / f"output{extension}"
    _encode(output, working_space, bit_depth)
    attestation = attest_reference_file_output_metadata(
        output,
        encoding_profile=encoding_profile,
    )
    assert attestation.accepted
    assert attestation.sensitive_metadata_absent
    assert attestation.orientation == 1
    assert attestation.failure_code is None
    assert attestation.file_sha256 is not None


def _inject_png_before_idat(payload: bytes, chunk: bytes) -> bytes:
    offset = 8
    while offset < len(payload):
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        if payload[offset + 4 : offset + 8] == b"IDAT":
            return payload[:offset] + chunk + payload[offset:]
        offset += 12 + length
    raise AssertionError("fixture PNG has no IDAT")


def test_p168_rejects_png_text_and_private_metadata(tmp_path: Path) -> None:
    output = tmp_path / "output.png"
    save_srgb16_png(_pixels(), output)
    for name, chunk_type, payload in (
        ("xmp", b"iTXt", b"XML:com.adobe.xmp\x00\x00\x00\x00\x00private"),
        ("text", b"tEXt", b"Comment\x00source text"),
        ("private", b"vpAg", b"application-private"),
    ):
        mutated = tmp_path / f"{name}.png"
        mutated.write_bytes(
            _inject_png_before_idat(
                output.read_bytes(),
                _png_chunk(chunk_type, payload),
            )
        )
        decision = attest_reference_file_output_metadata(
            mutated,
            encoding_profile="srgb-icc.v1",
        )
        assert not decision.accepted
        assert not decision.sensitive_metadata_absent


def _jpeg_segment(marker: int, payload: bytes) -> bytes:
    return b"\xff" + bytes([marker]) + struct.pack(">H", len(payload) + 2) + payload


def test_p168_rejects_jpeg_exif_xmp_iptc_and_comment(tmp_path: Path) -> None:
    output = tmp_path / "output.jpg"
    save_srgb8(_pixels(), output)
    original = output.read_bytes()
    for name, segment in (
        ("exif", _jpeg_segment(0xE1, b"Exif\x00\x00private")),
        ("xmp", _jpeg_segment(0xE1, b"http://ns.adobe.com/xap/1.0/\x00private")),
        ("iptc", _jpeg_segment(0xED, b"Photoshop 3.0\x00private")),
        ("comment", _jpeg_segment(0xFE, b"source comment")),
    ):
        mutated = tmp_path / f"{name}.jpg"
        mutated.write_bytes(original[:2] + segment + original[2:])
        decision = attest_reference_file_output_metadata(
            mutated,
            encoding_profile="srgb-icc.v1",
        )
        assert not decision.accepted
        assert not decision.sensitive_metadata_absent

    private_jfif = tmp_path / "private-jfif.jpg"
    private_jfif.write_bytes(
        original.replace(
            b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00",
            b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x01",
            1,
        )
    )
    assert not attest_reference_file_output_metadata(
        private_jfif,
        encoding_profile="srgb-icc.v1",
    ).accepted


def test_p168_rejects_tiff_source_metadata_and_orientation(tmp_path: Path) -> None:
    profile = srgb_icc_profile()
    encoded = np.rint(_pixels() * 255.0).astype(np.uint8)
    cases = (
        ("description", [(270, "s", 0, "source description", False)]),
        ("artist", [(315, "s", 0, "source artist", False)]),
        ("xmp", [(700, "B", 7, b"private", False)]),
        ("orientation", [(274, "H", 1, 6, False)]),
    )
    for name, extra in cases:
        output = tmp_path / f"{name}.tiff"
        tifffile.imwrite(
            output,
            encoded,
            photometric="rgb",
            planarconfig="contig",
            metadata=None,
            extratags=[
                (34675, "B", len(profile), profile, False),
                *extra,
            ],
        )
        decision = attest_reference_file_output_metadata(
            output,
            encoding_profile="srgb-icc.v1",
        )
        assert not decision.accepted
        assert not decision.sensitive_metadata_absent

    software = tmp_path / "software.tiff"
    tifffile.imwrite(
        software,
        encoded,
        photometric="rgb",
        planarconfig="contig",
        metadata=None,
        software="private source tool",
        extratags=[(34675, "B", len(profile), profile, False)],
    )
    assert not attest_reference_file_output_metadata(
        software,
        encoding_profile="srgb-icc.v1",
    ).accepted


def test_p168_rejects_wrong_or_duplicate_color_binding(tmp_path: Path) -> None:
    srgb_png = tmp_path / "srgb.png"
    save_srgb16_png(_pixels(), srgb_png)
    assert not attest_reference_file_output_metadata(
        srgb_png,
        encoding_profile="bt2020-sdr-cicp-1-1-0-1.v1",
    ).accepted

    duplicate = tmp_path / "duplicate.png"
    duplicate.write_bytes(
        _inject_png_before_idat(
            srgb_png.read_bytes(),
            _png_chunk(b"iCCP", b"duplicate\x00\x00" + zlib.compress(profile := srgb_icc_profile())),
        )
    )
    assert profile
    assert not attest_reference_file_output_metadata(
        duplicate,
        encoding_profile="srgb-icc.v1",
    ).accepted


def test_p168_attestation_is_bound_to_stable_file_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.color_match import output_metadata_policy

    output = tmp_path / "output.png"
    save_srgb16_png(_pixels(), output)
    hashes = iter(("a" * 64, "b" * 64))
    monkeypatch.setattr(
        output_metadata_policy,
        "sha256_file",
        lambda _path: next(hashes),
    )
    decision = output_metadata_policy.attest_reference_file_output_metadata(
        output,
        encoding_profile="srgb-icc.v1",
    )
    assert not decision.accepted
    assert decision.file_sha256 == "a" * 64
    assert decision.failure_code == "output changed during metadata attestation"


@pytest.mark.parametrize("extension", [".png", ".jpg", ".tiff"])
def test_p168_product_file_path_does_not_copy_source_exif(
    tmp_path: Path,
    extension: str,
) -> None:
    reference = tmp_path / "reference.jpg"
    source = tmp_path / "source.jpg"
    reference_pixels = np.rint(_pixels() * 255.0).astype(np.uint8)
    source_pixels = np.flip(reference_pixels, axis=1).copy()
    Image = pytest.importorskip("PIL.Image")
    exif = Image.Exif()
    exif[270] = "private source description"
    exif[274] = 6
    exif[315] = "private source artist"
    exif[37510] = b"private source comment"
    Image.fromarray(reference_pixels, mode="RGB").save(reference)
    Image.fromarray(source_pixels, mode="RGB").save(source, exif=exif)

    output = tmp_path / f"output{extension}"
    result = match_reference_files(
        reference,
        [source],
        [output],
        output_bit_depth=8,
    )
    decision = attest_reference_file_output_metadata(
        output,
        encoding_profile="srgb-icc.v1",
    )
    assert result.outputs[0].output_path == output
    assert decision.accepted
    assert decision.orientation == 1
    assert decision.sensitive_metadata_absent
    assert decision.file_sha256 == result.outputs[0].output_sha256


def test_p168_product_transaction_rejects_encoder_metadata_regression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.color_match import files

    reference = tmp_path / "reference.png"
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    recipe = tmp_path / "recipe.json"
    pixels = np.rint(_pixels() * 255.0).astype(np.uint8)
    Image = pytest.importorskip("PIL.Image")
    Image.fromarray(pixels, mode="RGB").save(reference)
    Image.fromarray(np.flip(pixels, axis=0).copy(), mode="RGB").save(source)
    real_encode = files._encode_working_image

    def encode_with_private_text(image, destination, *, output_bit_depth):
        result = real_encode(
            image,
            destination,
            output_bit_depth=output_bit_depth,
        )
        destination.write_bytes(
            _inject_png_before_idat(
                destination.read_bytes(),
                _png_chunk(b"tEXt", b"Comment\x00private source text"),
            )
        )
        return result

    monkeypatch.setattr(files, "_encode_working_image", encode_with_private_text)
    with pytest.raises(
        files.ReferenceMatchContractError,
        match="metadata-minimization policy",
    ):
        match_reference_files(
            reference,
            [source],
            [output],
            recipe_path=recipe,
            output_bit_depth=16,
        )
    assert not output.exists()
    assert not recipe.exists()
    assert not list(tmp_path.glob("*reference-match-stage*"))
