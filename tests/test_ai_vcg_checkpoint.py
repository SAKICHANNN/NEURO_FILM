import io
import pickle
import time
import zipfile

import numpy as np
import pytest
from safetensors.numpy import load

from scripts.acquire_ai_vcg_checkpoint import (
    BoundedReader,
    copy_group,
    header_and_offsets,
    request_groups,
)
from scripts.inspect_ai_vcg_checkpoint import (
    DescriptorUnpickler,
    FloatStorage,
    RemoteMetadata,
    tensor_descriptor,
)


def test_unknown_pickle_global_rejected_without_execution():
    with pytest.raises(ValueError, match="forbidden pickle global"):
        DescriptorUnpickler(io.BytesIO(pickle.dumps(eval))).load()


def test_whole_contiguous_tensor_descriptor_and_rejections():
    storage = {"dtype": "F32", "elements": 6, "storage": "7"}
    result = tensor_descriptor(storage, 0, (2, 3), (3, 1), False, {})
    assert result["shape"] == [2, 3]
    for offset, shape, stride in [(1, (2, 3), (3, 1)), (0, (3, 2), (1, 3))]:
        with pytest.raises(ValueError):
            tensor_descriptor(storage, offset, shape, stride, False, {})


def test_persistent_storage_validation():
    reader = DescriptorUnpickler(io.BytesIO())
    assert reader.persistent_load(("storage", FloatStorage, "7", "cuda:0", 6)) == {
        "dtype": "F32",
        "elements": 6,
        "storage": "7",
    }
    for value in [
        (),
        ("storage", FloatStorage, "../7", "cpu", 6),
        ("storage", FloatStorage, "7", "cpu", -1),
    ]:
        with pytest.raises(ValueError):
            reader.persistent_load(value)


def test_budget_and_invalid_seek_fail_before_network(monkeypatch):
    with RemoteMetadata({"id": "unused", "size": 100}, budget=10) as stream:
        monkeypatch.setattr(
            stream.session, "get", lambda *a, **k: pytest.fail("network")
        )
        with pytest.raises(ValueError, match="budget"):
            stream.read(11)
        with pytest.raises(ValueError, match="seek"):
            stream.seek(-1)
        stream.seek(100)
        assert stream.read() == b""


def test_selected_zip_storage_to_safetensors_exact_and_crc_rejection():
    original = np.arange(12, dtype="<f4").reshape(3, 4)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as writer:
        writer.writestr("checkpoint/data/4", original.tobytes())
    payload = archive.getvalue()
    with zipfile.ZipFile(io.BytesIO(payload)) as reader:
        entry = reader.infolist()[0]
    row = {
        "key": "layer.weight",
        "shape": [3, 4],
        "bytes": original.nbytes,
        "member": entry.filename,
        "crc32": f"{entry.CRC:08x}",
        "local_header_offset": entry.header_offset,
    }
    assert len(request_groups([row], len(payload))) == 1
    header, offsets, _ = header_and_offsets([row])
    output = io.BytesIO(header)
    bounded = BoundedReader(io.BytesIO(payload), len(payload), time.monotonic() + 5)
    hashes = copy_group(bounded, 0, [row], output, offsets, len(header))
    assert len(hashes) == 1 and bounded.remaining == 0
    assert np.array_equal(load(output.getvalue())["layer.weight"], original)
    bounded = BoundedReader(io.BytesIO(payload), len(payload), time.monotonic() + 5)
    with pytest.raises(ValueError, match="CRC"):
        copy_group(
            bounded,
            0,
            [{**row, "crc32": "00000000"}],
            io.BytesIO(header),
            offsets,
            len(header),
        )


def test_transfer_timeout_and_short_read():
    reader = BoundedReader(io.BytesIO(b"x"), 2, time.monotonic() + 5)
    with pytest.raises(ValueError, match="truncated"):
        reader.read(2)
    reader = BoundedReader(io.BytesIO(b"x"), 1, time.monotonic() - 1)
    with pytest.raises(TimeoutError):
        reader.read(1)
