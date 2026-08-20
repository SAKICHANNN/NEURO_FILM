from __future__ import annotations

import io
import pickle
import struct
import zipfile

import numpy as np
from PIL import Image

from src.real_film.ppisp_capture_pair_source_lock import parse_central_directory
from src.real_film.rgb2raw_capture_pair_pixel_preflight import (
    _pickle_facts,
    extract_member,
)


def _png() -> bytes:
    stream = io.BytesIO()
    values = np.arange(64, dtype=np.uint8).reshape(8, 8)
    rgb = np.stack((values, values, values), axis=2)
    Image.fromarray(rgb).save(stream, format="PNG")
    return stream.getvalue()


def test_row_metrics_decode_without_executing_pickle() -> None:
    raw = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4) / 64.0
    raw_stream = io.BytesIO()
    np.save(raw_stream, raw, allow_pickle=False)
    metadata = pickle.dumps({"iso": 100, "exposure": 0.01}, protocol=4)
    facts = _pickle_facts(metadata)
    assert facts["valid"] is True
    assert facts["global_symbols"] == []


def test_extract_member_validates_crc_and_size() -> None:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("member.bin", b"payload")
    payload = stream.getvalue()
    eocd = payload.rfind(b"PK\x05\x06")
    values = struct.unpack_from("<4s4H2IH", payload, eocd)
    central = payload[values[6] : values[6] + values[5]]
    member = parse_central_directory(central)[0]

    def read_range(_url: str, start: int, end: int, size: int) -> bytes:
        assert size == len(payload)
        return payload[start : end + 1]

    extracted, consumed = extract_member("unused", member, len(payload), read_range)
    assert extracted == b"payload"
    assert consumed == 30 + member.compressed_size
