"""Stream only source-locked VCG float tensors, with ZIP CRC verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import sys
import time
import zlib
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.film_physics.create_only_file import publish_create_only


def header_and_offsets(rows):
    header, offsets, cursor = {}, {}, 0
    for row in rows:
        size = row["bytes"]
        header[row["key"]] = {
            "dtype": "F32",
            "shape": row["shape"],
            "data_offsets": [cursor, cursor + size],
        }
        offsets[row["key"]] = cursor
        cursor += size
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    return struct.pack("<Q", len(raw)) + raw, offsets, cursor


def request_groups(rows, total_size):
    """Conservative local-header envelope, merged only across small gaps."""
    groups = []
    for row in sorted(rows, key=lambda item: item["local_header_offset"]):
        start = row["local_header_offset"]
        end = min(
            total_size, start + 30 + len(row["member"].encode()) + 65535 + row["bytes"]
        )
        if groups and start <= groups[-1][1] + 65536:
            groups[-1][1] = max(end, groups[-1][1])
            groups[-1][2].append(row)
        else:
            groups.append([start, end, [row]])
    return groups


class BoundedReader:
    def __init__(self, raw, remaining, deadline):
        self.raw, self.remaining, self.deadline = raw, remaining, deadline
        self.used = 0

    def read(self, size):
        if time.monotonic() > self.deadline:
            raise TimeoutError("acquisition wall-time cap")
        if not 0 <= size <= min(self.remaining, 1024 * 1024):
            raise ValueError("invalid bounded read")
        data = self.raw.read(size)
        if len(data) != size:
            raise ValueError("truncated HTTP range")
        self.remaining -= size
        self.used += size
        return data

    def skip(self, size):
        if size < 0:
            raise ValueError("overlapping or invalid ZIP offsets")
        while size:
            count = min(size, 1024 * 1024)
            self.read(count)
            size -= count


def copy_group(reader, start, rows, output, offsets, header_size):
    hashes = {}
    for row in rows:
        reader.skip(row["local_header_offset"] - start - reader.used)
        local = struct.unpack("<IHHHHHIIIHH", reader.read(30))
        signature, _, flags, compression, _, _, _, _, _, name_len, extra_len = local
        if signature != 0x04034B50 or compression != 0 or flags & 1:
            raise ValueError("unsupported local ZIP header")
        if reader.read(name_len).decode("utf-8") != row["member"]:
            raise ValueError("local/central name mismatch")
        reader.skip(extra_len)
        output.seek(header_size + offsets[row["key"]])
        digest, crc, left = hashlib.sha256(), 0, row["bytes"]
        while left:
            block = reader.read(min(left, 1024 * 1024))
            digest.update(block)
            crc = zlib.crc32(block, crc)
            output.write(block)
            left -= len(block)
        if f"{crc:08x}" != row["crc32"]:
            raise ValueError(f"storage CRC mismatch: {row['key']}")
        hashes[row["key"]] = digest.hexdigest()
    reader.skip(reader.remaining)
    return hashes


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/ai_vcg_acquisition_v1.json"
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    metadata = (ROOT / config["metadata_path"]).read_bytes()
    if hashlib.sha256(metadata).hexdigest() != config["metadata_sha256"]:
        raise ValueError("selection manifest identity mismatch")
    source_reports = json.loads(metadata)["sources"]
    planned = [
        request_groups(s["selected_tensors"], s["source"]["size"])
        for s in source_reports
    ]
    network_bytes = sum(end - start for groups in planned for start, end, _ in groups)
    if network_bytes > config["max_network_bytes"]:
        raise ValueError("planned ranges exceed transfer cap")
    print(
        json.dumps(
            {"network_bytes": network_bytes, "requests": list(map(len, planned))}
        ),
        flush=True,
    )
    if not args.execute:
        return
    root = ROOT / config["output_root"]
    if not root.resolve().is_relative_to((ROOT / "data/ai_models").resolve()):
        raise ValueError("destination outside project model namespace")
    if root.exists():
        raise FileExistsError(
            "use a fresh destination; partials are not verified weights"
        )
    if shutil.disk_usage(root.parent).free < config["minimum_free_bytes"]:
        raise ValueError("insufficient free space")
    root.mkdir()
    deadline = time.monotonic() + config["max_seconds"]
    results = []
    with requests.Session() as session:
        for report, groups in zip(source_reports, planned):
            source, rows = report["source"], report["selected_tensors"]
            header, offsets, storage_size = header_and_offsets(rows)
            destination = root / (source["select"] + ".safetensors")
            stage = destination.with_suffix(".partial")
            hashes = {}
            with stage.open("xb") as output:
                output.write(header)
                output.truncate(len(header) + storage_size)
                for index, (start, end, group_rows) in enumerate(groups):
                    with session.get(
                        "https://drive.usercontent.google.com/download",
                        params={
                            "id": source["id"],
                            "export": "download",
                            "confirm": "t",
                        },
                        headers={"Range": f"bytes={start}-{end - 1}"},
                        stream=True,
                        timeout=60,
                    ) as response:
                        expected = f"bytes {start}-{end - 1}/{source['size']}"
                        if (
                            response.status_code != 206
                            or response.headers.get("Content-Range") != expected
                        ):
                            raise ValueError("server range identity mismatch")
                        reader = BoundedReader(response.raw, end - start, deadline)
                        hashes.update(
                            copy_group(
                                reader, start, group_rows, output, offsets, len(header)
                            )
                        )
                    output.flush()
                    print(
                        source["name"],
                        index + 1,
                        "/",
                        len(groups),
                        len(hashes),
                        "tensors verified",
                        flush=True,
                    )
            from safetensors import safe_open

            with safe_open(stage, framework="numpy") as handle:
                if set(handle.keys()) != set(offsets):
                    raise ValueError("saved key mismatch")
                for row in rows:
                    if handle.get_slice(row["key"]).get_shape() != row["shape"]:
                        raise ValueError("saved shape mismatch")
            sha = file_sha(stage)
            # Both resolved paths must be siblings within the checked namespace.
            if (
                stage.resolve().parent != root.resolve()
                or destination.resolve().parent != root.resolve()
            ):
                raise ValueError("publication escaped model namespace")
            publish_create_only(stage, destination)
            results.append(
                {
                    "source": source,
                    "file": destination.name,
                    "sha256": sha,
                    "bytes": destination.stat().st_size,
                    "tensor_sha256": hashes,
                }
            )
    with (root / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(
            {
                "config": config,
                "network_bytes": network_bytes,
                "models": results,
                "complete_archive_sha256_verified": False,
                "verification": "source-locked ZIP storage CRCs and local tensor/container SHA256",
            },
            handle,
            indent=2,
        )
        handle.write("\n")


if __name__ == "__main__":
    main()
