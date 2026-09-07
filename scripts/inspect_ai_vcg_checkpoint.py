"""Bounded official VCG checkpoint metadata inspection; never executes torch pickle.

The restricted decoder recognizes only tensor descriptors and OrderedDict. It
does not import producer code, instantiate tensors, or read tensor storage.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import math
import pickle
import zipfile
from pathlib import Path

import requests

SOURCES = (
    {
        "name": "step1.ckpt",
        "id": "1WIVo09KWjIH2LVfk-uHz2MEuDODU419H",
        "size": 6866987641,
        "pickle_sha256": "eb379d5a7510b9c99c066213b0e7315fdeb51b274005cc11db892efaf55672b1",
        "select": "referencenet_state_dict",
    },
    {
        "name": "step2.ckpt",
        "id": "1VnE0ezMAPs8E4q4eZeA1ljlyuhNvzitt",
        "size": 3438391685,
        "pickle_sha256": "efde07a8e71edbfef4f0127053198931a41fe2f662aa2e1995d9c26f03576152",
        "select": "unet_state_dict",
    },
)


class RemoteMetadata(io.RawIOBase):
    def __init__(self, source: dict, budget: int = 8 * 1024 * 1024):
        self.source, self.budget = source, budget
        self.pos = self.used = 0
        self.session = requests.Session()

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        bases = {0: 0, 1: self.pos, 2: self.source["size"]}
        if whence not in bases:
            raise ValueError("invalid whence")
        pos = bases[whence] + offset
        if not 0 <= pos <= self.source["size"]:
            raise ValueError("invalid seek")
        self.pos = pos
        return pos

    def read(self, size=-1):
        available = self.source["size"] - self.pos
        size = min(available, size if size >= 0 else available)
        if size == 0:
            return b""
        if self.used + size > self.budget:
            raise ValueError("metadata transfer budget exceeded")
        end = self.pos + size - 1
        with self.session.get(
            "https://drive.usercontent.google.com/download",
            params={"id": self.source["id"], "export": "download", "confirm": "t"},
            headers={"Range": f"bytes={self.pos}-{end}"},
            stream=True,
            timeout=30,
        ) as response:
            expected = f"bytes {self.pos}-{end}/{self.source['size']}"
            if (
                response.status_code != 206
                or response.headers.get("Content-Range") != expected
            ):
                raise ValueError("server did not honor exact bounded range")
            payload = response.raw.read(size + 1)
            if len(payload) != size:
                raise ValueError("wrong range body length")
        self.pos += size
        self.used += size
        return payload

    def close(self):
        self.session.close()
        super().close()


class FloatStorage:
    """Identity token only, never torch storage."""


def tensor_descriptor(storage, offset, shape, stride, requires_grad, hooks):
    if not isinstance(storage, dict) or storage.get("dtype") != "F32":
        raise ValueError("invalid storage descriptor")
    if requires_grad not in (True, False) or hooks:
        raise ValueError("unsupported tensor metadata")
    if not isinstance(shape, tuple) or not isinstance(stride, tuple):
        raise TypeError("invalid dimensions")
    if len(shape) != len(stride) or len(shape) > 8:
        raise ValueError("invalid rank")
    expected = 1
    for dim, step in reversed(tuple(zip(shape, stride))):
        if (
            type(dim) is not int
            or dim <= 0
            or type(step) is not int
            or step != expected
        ):
            raise ValueError("only contiguous positive tensors are supported")
        expected *= dim
    if offset != 0 or math.prod(shape) != storage["elements"]:
        raise ValueError("only whole-storage tensors are supported")
    return {**storage, "shape": list(shape)}


class DescriptorUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ("collections", "OrderedDict"): collections.OrderedDict,
            ("torch", "FloatStorage"): FloatStorage,
            ("torch._utils", "_rebuild_tensor_v2"): tensor_descriptor,
        }
        if (module, name) not in allowed:
            raise ValueError(f"forbidden pickle global: {module}.{name}")
        return allowed[(module, name)]

    def persistent_load(self, value):
        if not isinstance(value, tuple) or len(value) != 5:
            raise ValueError("invalid persistent descriptor")
        kind, dtype, key, location, elements = value
        if kind != "storage" or dtype is not FloatStorage:
            raise ValueError("unsupported storage")
        if not isinstance(key, str) or not key.isdecimal():
            raise ValueError("invalid storage key")
        if type(elements) is not int or not 0 < elements <= 2**31:
            raise ValueError("invalid storage size")
        if not isinstance(location, str):
            raise TypeError("invalid location")
        return {"storage": key, "elements": elements, "dtype": "F32"}


def inspect_archive(stream, source):
    with zipfile.ZipFile(stream) as archive:
        members = archive.infolist()
        names = [entry.filename for entry in members]
        if len(set(names)) != len(names):
            raise ValueError("duplicate ZIP names")
        candidates = [
            entry for entry in members if entry.filename.endswith("/data.pkl")
        ]
        if len(candidates) != 1 or candidates[0].file_size > 4 * 1024 * 1024:
            raise ValueError("invalid pickle member")
        raw = archive.read(candidates[0])  # ZipFile independently verifies CRC.
        if hashlib.sha256(raw).hexdigest() != source["pickle_sha256"]:
            raise ValueError("checkpoint metadata identity mismatch")
        root = DescriptorUnpickler(io.BytesIO(raw)).load()
        selected = root[source["select"]]
        if not isinstance(selected, dict) or not selected or len(selected) > 2000:
            raise ValueError("invalid selected state")
        prefix = candidates[0].filename.removesuffix("data.pkl")
        byteorder_name = prefix + "byteorder"
        if byteorder_name in names:
            if archive.read(byteorder_name) != b"little":
                raise ValueError("only little-endian float storage is supported")
            byteorder = "explicit-little"
        else:
            # Torch 1.13 checkpoint format predates the byteorder record. Its
            # published CUDA/Linux training host is little endian; retain this
            # format assumption instead of inventing an embedded declaration.
            byteorder = "legacy-torch-host-little-assumed"
        rows = []
        seen = set()
        for key, tensor in selected.items():
            if not isinstance(key, str) or not isinstance(tensor, dict):
                raise TypeError("invalid parameter")
            member = archive.getinfo(prefix + "data/" + tensor["storage"])
            if member.compress_type != zipfile.ZIP_STORED or member.flag_bits & 1:
                raise ValueError("only unencrypted stored tensors supported")
            if member.file_size != tensor["elements"] * 4 or member.filename in seen:
                raise ValueError("storage mismatch or alias")
            seen.add(member.filename)
            rows.append(
                {
                    "key": key,
                    **tensor,
                    "member": member.filename,
                    "bytes": member.file_size,
                    "crc32": f"{member.CRC:08x}",
                    "local_header_offset": member.header_offset,
                }
            )
        # Local headers are deliberately left for acquisition; directory offsets
        # are not a claim that their payload has already been read or validated.
        return {
            "source": source,
            "archive_members": len(members),
            "selected_tensors": rows,
            "selected_storage_bytes": sum(row["bytes"] for row in rows),
            "metadata_only": True,
            "tensor_storage_reads": 0,
            "byteorder": byteorder,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reports = []
    for source in SOURCES:
        with RemoteMetadata(source) as stream:
            report = inspect_archive(stream, source)
            report["transferred_metadata_bytes"] = stream.used
            reports.append(report)
            print(source["name"], report["selected_storage_bytes"], flush=True)
    payload = {"schema": "ai-vcg-checkpoint-metadata.v1", "sources": reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
