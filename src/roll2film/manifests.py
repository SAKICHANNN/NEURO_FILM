"""Fail-closed FilmSet evidence, pair-blind manifests, and lockbox roles."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError


FILMSET_MANIFEST_SCHEMA = "roll2film.filmset_manifest.v1"
FILMSET_DATASET_ID = "xuhangc/filmset"
FILMSET_ARCHIVE_VERSION = "kaggle-v1-2023-11-29"
FILMSET_DOMAINS = {
    "input": "input",
    "cinema": "Cinema",
    "classneg": "ClassNeg",
    "velvia": "Velvia",
}
TRAINING_ROLES = frozenset({"source_train", "target_train"})
LOCKBOX_ROLES = frozenset({"internal_dev_lockbox", "final_628_lockbox"})
DEFAULT_EXPECTED_TRAIN = 4657
DEFAULT_EXPECTED_TEST = 628
DEFAULT_EXPECTED_FILES = 21140
DEFAULT_EXPECTED_BYTES = 11262805356


class FilmSetManifestError(ValueError):
    """Raised when FilmSet evidence or access roles violate the contract."""


@dataclass(frozen=True)
class FilmSetEvidenceConfig:
    root: Path
    output_dir: Path
    expected_train: int = DEFAULT_EXPECTED_TRAIN
    expected_test: int = DEFAULT_EXPECTED_TEST
    expected_files: int | None = DEFAULT_EXPECTED_FILES
    expected_bytes: int | None = DEFAULT_EXPECTED_BYTES
    split_seed: int = 20260715
    projection_seed: int = 2026071501
    embedding_dimensions: int = 32
    dhash_threshold: int = 4
    embedding_cosine_threshold: float = 0.995
    source_fraction: float = 0.45
    target_fraction: float = 0.45
    software_commit: str = "unknown"

    def __post_init__(self) -> None:
        if self.expected_train < 3 or self.expected_test < 1:
            raise FilmSetManifestError("expected split counts are too small")
        if not 0 <= self.dhash_threshold <= 4:
            raise FilmSetManifestError("dhash_threshold must be in [0, 4]")
        if not 0.0 < self.embedding_cosine_threshold <= 1.0:
            raise FilmSetManifestError("embedding_cosine_threshold must be in (0, 1]")
        if self.embedding_dimensions < 4:
            raise FilmSetManifestError("embedding_dimensions must be at least four")
        if self.source_fraction <= 0 or self.target_fraction <= 0:
            raise FilmSetManifestError("training pool fractions must be positive")
        if self.source_fraction + self.target_fraction >= 1.0:
            raise FilmSetManifestError("source and target fractions must leave a dev lockbox")


@dataclass(frozen=True)
class FilmSetEvidenceResult:
    report: dict[str, Any]
    manifest_paths: dict[str, Path]


@dataclass(frozen=True)
class PairBlindFilmSetView:
    """Training-only view that cannot accept evaluator lockbox manifests."""

    source_rows: tuple[dict[str, Any], ...]
    target_rows: tuple[dict[str, Any], ...]

    @classmethod
    def from_manifests(cls, source_path: Path, target_path: Path) -> "PairBlindFilmSetView":
        source = _load_role_manifest(source_path, "source_train")
        target = _load_role_manifest(target_path, "target_train")
        source_ids = {str(row["content_id"]) for row in source}
        target_ids = {str(row["content_id"]) for row in target}
        source_clusters = {str(row["duplicate_cluster_id"]) for row in source}
        target_clusters = {str(row["duplicate_cluster_id"]) for row in target}
        if source_ids & target_ids:
            raise FilmSetManifestError("source and target training content IDs overlap")
        if source_clusters & target_clusters:
            raise FilmSetManifestError("source and target duplicate clusters overlap")
        if {row["domain"] for row in source} != {"input"}:
            raise FilmSetManifestError("source manifest must expose only input images")
        if "input" in {row["domain"] for row in target}:
            raise FilmSetManifestError("target manifest must not expose input images")
        return cls(tuple(source), tuple(target))


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            winner, loser = sorted((left_root, right_root))
            self.parent[loser] = winner


def _sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _dhash64(image: Image.Image) -> int:
    values = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    bits = values[:, :-1] > values[:, 1:]
    result = 0
    for bit in bits.ravel():
        result = (result << 1) | int(bit)
    return result


def _projection(config: FilmSetEvidenceConfig) -> tuple[np.ndarray, str]:
    rng = np.random.default_rng(config.projection_seed)
    matrix = rng.normal(0.0, 1.0, size=(16 * 16 * 3, config.embedding_dimensions))
    matrix /= np.sqrt(matrix.shape[0])
    return matrix, _sha256_bytes(matrix.astype("<f8", copy=False).tobytes())


def _visual_embedding(image: Image.Image, projection: np.ndarray) -> np.ndarray:
    values = np.asarray(
        image.convert("RGB").resize((16, 16), Image.Resampling.BILINEAR),
        dtype=np.float64,
    )
    values = values / 255.0
    values = (values - values.mean(axis=(0, 1), keepdims=True)) / np.maximum(
        values.std(axis=(0, 1), keepdims=True),
        1e-4,
    )
    embedding = values.reshape(-1) @ projection
    norm = float(np.linalg.norm(embedding))
    if norm <= 1e-12:
        return np.zeros(projection.shape[1], dtype=np.float64)
    return embedding / norm


def _decoded_canaries(path: Path, projection: np.ndarray) -> dict[str, Any]:
    try:
        with Image.open(path) as image:
            image.load()
            icc = image.info.get("icc_profile")
            embedding = _visual_embedding(image, projection)
            return {
                "width": int(image.width),
                "height": int(image.height),
                "icc_sha256": _sha256_bytes(icc) if isinstance(icc, bytes) else None,
                "dhash64": f"{_dhash64(image):016x}",
                "visual_embedding": embedding.tolist(),
                "visual_embedding_hash": _sha256_bytes(
                    embedding.astype("<f8", copy=False).tobytes()
                ),
            }
    except (OSError, UnidentifiedImageError) as exc:
        raise FilmSetManifestError(f"cannot decode train input canary {path}: {exc}") from exc


def _read_cache(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FilmSetManifestError(f"{path}:{line_number}: invalid cache JSON") from exc
        records[str(row["path"])] = row
    return records


def _cached_file_record(
    *,
    root: Path,
    path: Path,
    decode_canaries: bool,
    projection: np.ndarray,
    cache: dict[str, dict[str, Any]],
    cache_handle: Any,
) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    stat = path.stat()
    previous = cache.get(relative)
    required_canaries = not decode_canaries or (
        previous is not None
        and previous.get("dhash64") is not None
        and previous.get("visual_embedding") is not None
    )
    if (
        previous is not None
        and int(previous.get("bytes", -1)) == stat.st_size
        and int(previous.get("mtime_ns", -1)) == stat.st_mtime_ns
        and previous.get("sha256")
        and required_canaries
    ):
        return previous
    row: dict[str, Any] = {
        "path": relative,
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256_file(path),
        "width": None,
        "height": None,
        "icc_sha256": None,
        "dhash64": None,
        "visual_embedding": None,
        "visual_embedding_hash": None,
    }
    if decode_canaries:
        row.update(_decoded_canaries(path, projection))
    cache_handle.write(json.dumps(row, sort_keys=True) + "\n")
    cache_handle.flush()
    cache[relative] = row
    return row


def _domain_files(root: Path, distributed_split: str, domain_dir: str) -> list[Path]:
    directory = root / distributed_split / domain_dir
    if not directory.is_dir():
        raise FilmSetManifestError(f"missing FilmSet directory: {directory}")
    files = sorted((path for path in directory.iterdir() if path.is_file()), key=lambda path: path.name)
    if not files:
        raise FilmSetManifestError(f"empty FilmSet directory: {directory}")
    return files


def _validate_tree(config: FilmSetEvidenceConfig) -> dict[str, dict[str, list[Path]]]:
    tree: dict[str, dict[str, list[Path]]] = {"train": {}, "test": {}}
    expected = {"train": config.expected_train, "test": config.expected_test}
    content_ids: dict[str, set[str]] = {}
    for distributed_split in ("train", "test"):
        for domain, directory in FILMSET_DOMAINS.items():
            files = _domain_files(config.root, distributed_split, directory)
            if len(files) != expected[distributed_split]:
                raise FilmSetManifestError(
                    f"{distributed_split}/{directory}: expected {expected[distributed_split]} files, got {len(files)}"
                )
            tree[distributed_split][domain] = files
            names = {path.name.casefold() for path in files}
            if len(names) != len(files):
                raise FilmSetManifestError(f"case-folded basename collision in {distributed_split}/{directory}")
            key = f"{distributed_split}:{domain}"
            content_ids[key] = names
        reference = content_ids[f"{distributed_split}:input"]
        for domain in FILMSET_DOMAINS:
            if content_ids[f"{distributed_split}:{domain}"] != reference:
                raise FilmSetManifestError(f"basename parity failed for {distributed_split}/{domain}")
    return tree


def _dhash_pairs(values: dict[str, int], threshold: int) -> set[tuple[str, str]]:
    buckets: dict[tuple[int, int], list[str]] = {}
    for content_id, value in values.items():
        for chunk_index in range(4):
            chunk = (value >> (16 * chunk_index)) & 0xFFFF
            buckets.setdefault((chunk_index, chunk), []).append(content_id)
    pairs: set[tuple[str, str]] = set()
    seen: set[tuple[str, str]] = set()
    for candidates in buckets.values():
        for index, left in enumerate(candidates):
            for right in candidates[index + 1 :]:
                pair = tuple(sorted((left, right)))
                if pair in seen:
                    continue
                seen.add(pair)
                if (values[left] ^ values[right]).bit_count() <= threshold:
                    pairs.add(pair)
    return pairs


def _embedding_pairs(
    content_ids: list[str],
    embeddings: np.ndarray,
    threshold: float,
    block_size: int = 256,
) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for start in range(0, len(content_ids), block_size):
        similarities = embeddings[start : start + block_size] @ embeddings.T
        for local_index, matches in enumerate(similarities):
            left_index = start + local_index
            for right_index in np.flatnonzero(matches >= threshold):
                if right_index > left_index:
                    pairs.add((content_ids[left_index], content_ids[int(right_index)]))
    return pairs


def _clusters(
    input_records: dict[str, dict[str, Any]],
    config: FilmSetEvidenceConfig,
) -> tuple[dict[str, str], dict[str, Any]]:
    content_ids = sorted(input_records)
    union = _UnionFind(content_ids)
    exact_groups: dict[str, list[str]] = {}
    for content_id, row in input_records.items():
        exact_groups.setdefault(str(row["sha256"]), []).append(content_id)
    exact_pairs = 0
    for group in exact_groups.values():
        for other in group[1:]:
            union.union(group[0], other)
            exact_pairs += 1

    dhashes = {content_id: int(input_records[content_id]["dhash64"], 16) for content_id in content_ids}
    dhash_pairs = _dhash_pairs(dhashes, config.dhash_threshold)
    for left, right in dhash_pairs:
        union.union(left, right)

    embeddings = np.asarray(
        [input_records[content_id]["visual_embedding"] for content_id in content_ids],
        dtype=np.float64,
    )
    embedding_pairs = _embedding_pairs(
        content_ids,
        embeddings,
        config.embedding_cosine_threshold,
    )
    for left, right in embedding_pairs:
        union.union(left, right)

    members: dict[str, list[str]] = {}
    for content_id in content_ids:
        members.setdefault(union.find(content_id), []).append(content_id)
    cluster_ids: dict[str, str] = {}
    for group in members.values():
        cluster_id = "cluster:" + hashlib.sha256("\n".join(sorted(group)).encode()).hexdigest()[:16]
        for content_id in group:
            cluster_ids[content_id] = cluster_id
    return cluster_ids, {
        "cluster_count": len(set(cluster_ids.values())),
        "largest_cluster": max(len(group) for group in members.values()),
        "exact_pairs": exact_pairs,
        "dhash_pairs": len(dhash_pairs),
        "embedding_pairs": len(embedding_pairs),
    }


def _assign_pools(
    cluster_ids: dict[str, str],
    config: FilmSetEvidenceConfig,
) -> dict[str, str]:
    members: dict[str, list[str]] = {}
    for content_id, cluster_id in cluster_ids.items():
        members.setdefault(cluster_id, []).append(content_id)
    ordered = sorted(
        members,
        key=lambda cluster_id: hashlib.sha256(
            f"{config.split_seed}:{cluster_id}".encode()
        ).hexdigest(),
    )
    total = len(cluster_ids)
    source_target = round(total * config.source_fraction)
    target_target = round(total * config.target_fraction)
    pool_by_cluster: dict[str, str] = {}
    counts = {"source_train": 0, "target_train": 0, "internal_dev_lockbox": 0}
    for cluster_id in ordered:
        size = len(members[cluster_id])
        if counts["source_train"] < source_target:
            pool = "source_train"
        elif counts["target_train"] < target_target:
            pool = "target_train"
        else:
            pool = "internal_dev_lockbox"
        pool_by_cluster[cluster_id] = pool
        counts[pool] += size
    if any(value == 0 for value in counts.values()):
        raise FilmSetManifestError(f"pair-blind pool assignment produced an empty pool: {counts}")
    return {content_id: pool_by_cluster[cluster_id] for content_id, cluster_id in cluster_ids.items()}


def _manifest_row(
    *,
    distributed_split: str,
    domain: str,
    content_id: str,
    cluster_id: str,
    research_pool: str,
    file_record: dict[str, Any],
) -> dict[str, Any]:
    decoded_allowed = distributed_split == "train" and domain == "input"
    return {
        "schema_version": FILMSET_MANIFEST_SCHEMA,
        "dataset_id": FILMSET_DATASET_ID,
        "archive_version": FILMSET_ARCHIVE_VERSION,
        "distributed_split": distributed_split,
        "domain": domain,
        "content_id": content_id,
        "duplicate_cluster_id": cluster_id,
        "research_pool": research_pool,
        "path": file_record["path"],
        "bytes": file_record["bytes"],
        "mtime_ns": file_record["mtime_ns"],
        "sha256": file_record["sha256"],
        "width": file_record["width"] if decoded_allowed else None,
        "height": file_record["height"] if decoded_allowed else None,
        "icc_sha256": file_record["icc_sha256"] if decoded_allowed else None,
        "dhash64": file_record["dhash64"] if decoded_allowed else None,
        "visual_embedding_hash": file_record["visual_embedding_hash"] if decoded_allowed else None,
        "allowed_use": "internal_research_only",
        "redistributable": False,
        "license_snapshot": "kaggle_metadata_mit_per_image_rights_unverified",
        "payload_access": {
            "source_train": "training_source_only",
            "target_train": "training_target_only",
            "internal_dev_lockbox": "internal_evaluator_only",
            "final_628_lockbox": "final_evaluator_only",
        }[research_pool],
    }


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    encoded = payload.encode("utf-8")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)
    return _sha256_bytes(encoded)


def _atomic_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    encoded = text.encode("utf-8")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)
    return _sha256_bytes(encoded)


def _load_role_manifest(path: Path, expected_role: str) -> list[dict[str, Any]]:
    if expected_role not in TRAINING_ROLES:
        raise FilmSetManifestError("training loader accepts only source_train or target_train")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("schema_version") != FILMSET_MANIFEST_SCHEMA:
            raise FilmSetManifestError(f"{path}:{line_number}: unsupported schema")
        role = str(row.get("research_pool"))
        if role in LOCKBOX_ROLES:
            raise FilmSetManifestError(f"{path}:{line_number}: lockbox role rejected by training loader")
        if role != expected_role:
            raise FilmSetManifestError(f"{path}:{line_number}: expected role {expected_role}, got {role}")
        rows.append(row)
    if not rows:
        raise FilmSetManifestError(f"{path}: empty training manifest")
    return rows


def build_filmset_evidence(
    config: FilmSetEvidenceConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> FilmSetEvidenceResult:
    """Hash FilmSet, freeze pair-blind manifests, and leave source bytes untouched."""
    root = config.root.resolve()
    output_dir = config.output_dir.resolve()
    if not root.is_dir():
        raise FilmSetManifestError(f"FilmSet root does not exist: {root}")
    tree = _validate_tree(config)
    projection, projection_hash = _projection(config)
    cache_path = output_dir / "inventory_cache.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = _read_cache(cache_path)
    records: dict[str, dict[str, Any]] = {}
    ordered_files: list[tuple[str, str, Path]] = []
    for distributed_split in ("train", "test"):
        for domain in FILMSET_DOMAINS:
            ordered_files.extend(
                (distributed_split, domain, path) for path in tree[distributed_split][domain]
            )
    with cache_path.open("a", encoding="utf-8") as cache_handle:
        for index, (distributed_split, domain, path) in enumerate(ordered_files, start=1):
            relative = path.relative_to(root).as_posix()
            records[relative] = _cached_file_record(
                root=root,
                path=path,
                decode_canaries=distributed_split == "train" and domain == "input",
                projection=projection,
                cache=cache,
                cache_handle=cache_handle,
            )
            if progress and (index % 500 == 0 or index == len(ordered_files)):
                progress(f"FilmSet evidence {index}/{len(ordered_files)}")

    total_bytes = sum(int(row["bytes"]) for row in records.values())
    if config.expected_files is not None and len(records) != config.expected_files:
        raise FilmSetManifestError(
            f"expected {config.expected_files} files, got {len(records)}"
        )
    if config.expected_bytes is not None and total_bytes != config.expected_bytes:
        raise FilmSetManifestError(
            f"expected {config.expected_bytes} bytes, got {total_bytes}"
        )

    train_input_records = {
        f"train:{path.name.casefold()}": records[path.relative_to(root).as_posix()]
        for path in tree["train"]["input"]
    }
    train_input_hashes = {str(row["sha256"]) for row in train_input_records.values()}
    test_input_hashes = {
        str(records[path.relative_to(root).as_posix()]["sha256"])
        for path in tree["test"]["input"]
    }
    exact_train_test_overlap = train_input_hashes & test_input_hashes
    if exact_train_test_overlap:
        raise FilmSetManifestError(
            "distributed train/test input payloads overlap exactly: "
            f"{len(exact_train_test_overlap)} SHA-256 values"
        )
    cluster_ids, duplicate_report = _clusters(train_input_records, config)
    pool_by_content = _assign_pools(cluster_ids, config)
    rows_by_role: dict[str, list[dict[str, Any]]] = {
        "source_train": [],
        "target_train": [],
        "internal_dev_lockbox": [],
        "final_628_lockbox": [],
    }
    train_paths = {
        domain: {f"train:{path.name.casefold()}": path for path in paths}
        for domain, paths in tree["train"].items()
    }
    for content_id in sorted(train_input_records):
        pool = pool_by_content[content_id]
        domains = {
            "source_train": ("input",),
            "target_train": ("cinema", "classneg", "velvia"),
            "internal_dev_lockbox": tuple(FILMSET_DOMAINS),
        }[pool]
        for domain in domains:
            path = train_paths[domain][content_id]
            rows_by_role[pool].append(
                _manifest_row(
                    distributed_split="train",
                    domain=domain,
                    content_id=content_id,
                    cluster_id=cluster_ids[content_id],
                    research_pool=pool,
                    file_record=records[path.relative_to(root).as_posix()],
                )
            )

    test_paths = {
        domain: {f"test:{path.name.casefold()}": path for path in paths}
        for domain, paths in tree["test"].items()
    }
    for content_id in sorted(test_paths["input"]):
        cluster_id = "final:" + hashlib.sha256(content_id.encode()).hexdigest()[:16]
        for domain in FILMSET_DOMAINS:
            path = test_paths[domain][content_id]
            rows_by_role["final_628_lockbox"].append(
                _manifest_row(
                    distributed_split="test",
                    domain=domain,
                    content_id=content_id,
                    cluster_id=cluster_id,
                    research_pool="final_628_lockbox",
                    file_record=records[path.relative_to(root).as_posix()],
                )
            )

    source_ids = {row["content_id"] for row in rows_by_role["source_train"]}
    target_ids = {row["content_id"] for row in rows_by_role["target_train"]}
    dev_ids = {row["content_id"] for row in rows_by_role["internal_dev_lockbox"]}
    source_clusters = {cluster_ids[value] for value in source_ids}
    target_clusters = {cluster_ids[value] for value in target_ids}
    dev_clusters = {cluster_ids[value] for value in dev_ids}
    leakage = {
        "content_id_intersections": {
            "source_target": len(source_ids & target_ids),
            "source_dev": len(source_ids & dev_ids),
            "target_dev": len(target_ids & dev_ids),
        },
        "duplicate_cluster_intersections": {
            "source_target": len(source_clusters & target_clusters),
            "source_dev": len(source_clusters & dev_clusters),
            "target_dev": len(target_clusters & dev_clusters),
        },
        "exact_cross_pool": 0,
        "dhash_cross_pool": 0,
        "embedding_cross_pool": 0,
    }
    if any(leakage["content_id_intersections"].values()) or any(
        leakage["duplicate_cluster_intersections"].values()
    ):
        raise FilmSetManifestError(f"pair-blind leakage detected: {leakage}")

    manifest_names = {
        "source_train": "source_train.jsonl",
        "target_train": "target_train.jsonl",
        "internal_dev_lockbox": "internal_dev_lockbox.jsonl",
        "final_628_lockbox": "final_628_lockbox.jsonl",
    }
    manifest_paths = {role: output_dir / name for role, name in manifest_names.items()}
    manifest_hashes = {
        role: _atomic_jsonl(manifest_paths[role], rows_by_role[role])
        for role in manifest_paths
    }
    PairBlindFilmSetView.from_manifests(
        manifest_paths["source_train"],
        manifest_paths["target_train"],
    )

    tree_lines = [
        f"{path}\0{records[path]['bytes']}\0{records[path]['sha256']}\n"
        for path in sorted(records)
    ]
    tree_hash = _sha256_bytes("".join(tree_lines).encode())
    pool_identity_counts = {
        "source_train": len(source_ids),
        "target_train": len(target_ids),
        "internal_dev_lockbox": len(dev_ids),
        "final_628_lockbox": config.expected_test,
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "dataset_id": FILMSET_DATASET_ID,
        "archive_version": FILMSET_ARCHIVE_VERSION,
        "source_root": "data/raw/filmset/FilmSet",
        "software_commit": config.software_commit,
        "archive_observed": {
            "train_per_domain": config.expected_train,
            "test_per_domain": config.expected_test,
            "paper_reported_test_n": 638,
            "image_files": len(records),
            "image_bytes": total_bytes,
            "tree_sha256": tree_hash,
            "basename_parity": True,
            "train_test_basename_overlap": len(
                {path.name.casefold() for path in tree["train"]["input"]}
                & {path.name.casefold() for path in tree["test"]["input"]}
            ),
            "train_test_exact_input_sha256_overlap": 0,
        },
        "split_contract": {
            "split_seed": config.split_seed,
            "source_fraction": config.source_fraction,
            "target_fraction": config.target_fraction,
            "pool_identity_counts": pool_identity_counts,
        },
        "duplicate_canaries": {
            **duplicate_report,
            "dhash_threshold": config.dhash_threshold,
            "embedding_cosine_threshold": config.embedding_cosine_threshold,
            "projection_seed": config.projection_seed,
            "projection_sha256": projection_hash,
            "embedding_dimensions": config.embedding_dimensions,
        },
        "leakage": leakage,
        "manifest_rows": {role: len(rows) for role, rows in rows_by_role.items()},
        "manifest_sha256": manifest_hashes,
        "final_lockbox_contract": {
            "identities": config.expected_test,
            "rows": len(rows_by_role["final_628_lockbox"]),
            "payload_access": "final_evaluator_only",
            "decoded_test_payloads": 0,
            "logical_separation_not_os_acl": True,
        },
        "rights": {
            "allowed_use": "internal_research_only",
            "redistributable": False,
            "kaggle_metadata_license": "MIT",
            "per_image_rights_manifest": "absent",
            "public_weights_or_examples": "release_review_required",
        },
        "decision": "pass",
        "claim_boundary": "FilmSet paired-blind recipe research only; not physical film or stock calibration.",
    }
    report_path = output_dir / "report.json"
    _atomic_json(report_path, report)
    return FilmSetEvidenceResult(report=report, manifest_paths=manifest_paths)
