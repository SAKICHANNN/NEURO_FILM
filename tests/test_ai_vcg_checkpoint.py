import io
import pickle

import pytest

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
