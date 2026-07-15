"""Fail-closed FilmSet access and equal-image sampling for CT5."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.preprocess.pipeline import load_working_image
from src.preprocess.types import WorkingImage

from .manifests import FILMSET_MANIFEST_SCHEMA, FilmSetManifestError


@dataclass(frozen=True)
class CT5DataContract:
    project_root: Path
    dataset_root: Path
    evidence_dir: Path
    experiment_id: str
    config_sha256: str
    seed: int
    pixels_per_training_image: int
    pixels_per_dev_image: int
    pilot_fraction: float
    domains: tuple[str, ...]
    manifest_hashes: dict[str, str]

    @classmethod
    def from_config(cls, config_path: Path, project_root: Path) -> "CT5DataContract":
        resolved_config = config_path.resolve()
        root = project_root.resolve()
        payload = json.loads(resolved_config.read_text(encoding="utf-8"))
        dataset = payload["dataset"]
        sampling = payload["sampling"]
        pilot_fraction = float(sampling["pilot_fraction"])
        if not 0.0 < pilot_fraction < 1.0:
            raise FilmSetManifestError("CT5 pilot fraction must be inside (0, 1)")
        domains = tuple(str(value) for value in dataset["domains"])
        if not domains or "input" in domains or len(set(domains)) != len(domains):
            raise FilmSetManifestError("CT5 target domains must be unique and exclude input")
        hashes = {
            "source_train": str(dataset["source_manifest_sha256"]),
            "target_train": str(dataset["target_manifest_sha256"]),
            "internal_dev_lockbox": str(dataset["internal_dev_manifest_sha256"]),
            "final_628_lockbox": str(dataset["forbidden_final_manifest_sha256"]),
        }
        return cls(
            project_root=root,
            dataset_root=(root / dataset["root"]).resolve(),
            evidence_dir=(root / dataset["evidence_dir"]).resolve(),
            experiment_id=str(payload["experiment_id"]),
            config_sha256=_sha256_file(resolved_config),
            seed=int(payload["seed"]),
            pixels_per_training_image=int(sampling["pixels_per_training_image"]),
            pixels_per_dev_image=int(sampling["pixels_per_dev_image"]),
            pilot_fraction=pilot_fraction,
            domains=domains,
            manifest_hashes=hashes,
        )

    def verify_final_lockbox_sealed(self) -> None:
        path = self.evidence_dir / "final_628_lockbox.jsonl"
        _verify_hash(path, self.manifest_hashes["final_628_lockbox"])


@dataclass(frozen=True)
class CT5TrainingSamples:
    source_ids: tuple[str, ...]
    source_pixels: np.ndarray
    target_ids: dict[str, tuple[str, ...]]
    target_pixels: dict[str, np.ndarray]


@dataclass(frozen=True)
class CT5DevSamples:
    content_ids: tuple[str, ...]
    cluster_ids: tuple[str, ...]
    folds: tuple[str, ...]
    input_pixels: np.ndarray
    target_pixels: dict[str, np.ndarray]


def load_ct5_training_rows(
    contract: CT5DataContract,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    source = _read_role_manifest(
        contract.evidence_dir / "source_train.jsonl",
        contract.manifest_hashes["source_train"],
        "source_train",
    )
    target = _read_role_manifest(
        contract.evidence_dir / "target_train.jsonl",
        contract.manifest_hashes["target_train"],
        "target_train",
    )
    if {str(row["domain"]) for row in source} != {"input"}:
        raise FilmSetManifestError("CT5 source training manifest must contain input only")
    target_by_domain = {
        domain: sorted(
            (row for row in target if str(row["domain"]) == domain),
            key=lambda row: str(row["content_id"]),
        )
        for domain in contract.domains
    }
    if any(not rows for rows in target_by_domain.values()):
        raise FilmSetManifestError("CT5 target training manifest is missing a frozen domain")
    if set(map(str, (row["domain"] for row in target))) != set(contract.domains):
        raise FilmSetManifestError("CT5 target manifest contains an undeclared domain")
    source_ids = {str(row["content_id"]) for row in source}
    target_ids = {str(row["content_id"]) for row in target}
    source_clusters = {str(row["duplicate_cluster_id"]) for row in source}
    target_clusters = {str(row["duplicate_cluster_id"]) for row in target}
    if source_ids & target_ids or source_clusters & target_clusters:
        raise FilmSetManifestError("CT5 source/target training pools overlap")
    contract.verify_final_lockbox_sealed()
    return sorted(source, key=lambda row: str(row["content_id"])), target_by_domain


def sample_ct5_training(
    contract: CT5DataContract,
    *,
    progress: Callable[[str], None] | None = None,
) -> CT5TrainingSamples:
    source_rows, target_rows = load_ct5_training_rows(contract)
    source_ids, source_pixels = _sample_rows(
        contract,
        source_rows,
        contract.pixels_per_training_image,
        progress=progress,
        label="source_train",
    )
    target_ids: dict[str, tuple[str, ...]] = {}
    target_pixels: dict[str, np.ndarray] = {}
    for domain, rows in target_rows.items():
        identifiers, pixels = _sample_rows(
            contract,
            rows,
            contract.pixels_per_training_image,
            progress=progress,
            label=f"target_train:{domain}",
        )
        target_ids[domain] = identifiers
        target_pixels[domain] = pixels
    return CT5TrainingSamples(source_ids, source_pixels, target_ids, target_pixels)


def sample_ct5_internal_dev(
    contract: CT5DataContract,
    *,
    progress: Callable[[str], None] | None = None,
) -> CT5DevSamples:
    """Evaluator-only decoder. Training code has no manifest-path parameter."""
    rows = load_ct5_internal_dev_rows(contract)
    by_content: dict[str, dict[str, dict[str, Any]]] = {}
    clusters: dict[str, str] = {}
    for row in rows:
        content_id = str(row["content_id"])
        domain = str(row["domain"])
        by_content.setdefault(content_id, {})[domain] = row
        clusters[content_id] = str(row["duplicate_cluster_id"])
    required = {"input", *contract.domains}
    for content_id, domain_rows in by_content.items():
        if set(domain_rows) != required:
            raise FilmSetManifestError(f"internal dev {content_id} does not contain all domains")
    content_ids = tuple(sorted(by_content))
    input_samples: list[np.ndarray] = []
    target_samples = {domain: [] for domain in contract.domains}
    folds: list[str] = []
    cluster_ids: list[str] = []
    for index, content_id in enumerate(content_ids, start=1):
        domain_rows = by_content[content_id]
        decoded = {
            domain: _load_pixels(contract, row)
            for domain, row in domain_rows.items()
        }
        shape = decoded["input"].shape
        if any(values.shape != shape for values in decoded.values()):
            raise FilmSetManifestError(f"paired dev dimensions differ for {content_id}")
        indices = _sample_indices(
            len(decoded["input"]),
            contract.pixels_per_dev_image,
            contract.seed,
            content_id,
        )
        input_samples.append(decoded["input"][indices])
        for domain in contract.domains:
            target_samples[domain].append(decoded[domain][indices])
        cluster_id = clusters[content_id]
        cluster_ids.append(cluster_id)
        folds.append(_fold(contract, cluster_id))
        if progress and (index % 50 == 0 or index == len(content_ids)):
            progress(f"internal_dev {index}/{len(content_ids)}")
    return CT5DevSamples(
        content_ids=content_ids,
        cluster_ids=tuple(cluster_ids),
        folds=tuple(folds),
        input_pixels=np.stack(input_samples),
        target_pixels={domain: np.stack(values) for domain, values in target_samples.items()},
    )


def _read_role_manifest(path: Path, expected_hash: str, expected_role: str) -> list[dict[str, Any]]:
    _verify_hash(path, expected_hash)
    if expected_role == "final_628_lockbox":
        raise FilmSetManifestError("CT5 refuses to parse the final 628 lockbox")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        row = json.loads(line)
        if row.get("schema_version") != FILMSET_MANIFEST_SCHEMA:
            raise FilmSetManifestError(f"{path}:{line_number}: unsupported schema")
        if row.get("research_pool") != expected_role:
            raise FilmSetManifestError(f"{path}:{line_number}: unexpected role")
        expected_access = {
            "source_train": "training_source_only",
            "target_train": "training_target_only",
            "internal_dev_lockbox": "internal_evaluator_only",
        }[expected_role]
        if row.get("payload_access") != expected_access:
            raise FilmSetManifestError(f"{path}:{line_number}: unexpected payload access")
        rows.append(row)
    if not rows:
        raise FilmSetManifestError(f"{path}: empty CT5 manifest")
    return rows


def _sample_rows(
    contract: CT5DataContract,
    rows: list[dict[str, Any]],
    pixel_count: int,
    *,
    progress: Callable[[str], None] | None,
    label: str,
) -> tuple[tuple[str, ...], np.ndarray]:
    identifiers: list[str] = []
    samples: list[np.ndarray] = []
    for index, row in enumerate(rows, start=1):
        content_id = str(row["content_id"])
        pixels = _load_pixels(contract, row)
        indices = _sample_indices(len(pixels), pixel_count, contract.seed, f"{label}:{content_id}")
        identifiers.append(content_id)
        samples.append(pixels[indices])
        if progress and (index % 100 == 0 or index == len(rows)):
            progress(f"{label} {index}/{len(rows)}")
    return tuple(identifiers), np.stack(samples)


def _load_pixels(contract: CT5DataContract, row: dict[str, Any]) -> np.ndarray:
    return load_ct5_working_image(contract, row).pixels.reshape(-1, 3)


def load_ct5_internal_dev_rows(contract: CT5DataContract) -> list[dict[str, Any]]:
    rows = _read_role_manifest(
        contract.evidence_dir / "internal_dev_lockbox.jsonl",
        contract.manifest_hashes["internal_dev_lockbox"],
        "internal_dev_lockbox",
    )
    contract.verify_final_lockbox_sealed()
    return rows


def load_ct5_working_image(
    contract: CT5DataContract,
    row: dict[str, Any],
) -> WorkingImage:
    relative = Path(str(row["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise FilmSetManifestError("manifest path escapes the FilmSet root")
    path = (contract.dataset_root / relative).resolve()
    try:
        path.relative_to(contract.dataset_root)
    except ValueError as exc:
        raise FilmSetManifestError("manifest path escapes the FilmSet root") from exc
    if _sha256_file(path) != str(row["sha256"]):
        raise FilmSetManifestError(f"payload hash mismatch: {relative.as_posix()}")
    working = load_working_image(path)
    if working.working_space != "linear_srgb" or working.transfer_state != "display_linear":
        raise FilmSetManifestError(f"unexpected decoded colour state: {relative.as_posix()}")
    if any(warning.code == "icc_convert_failed" for warning in working.warnings):
        raise FilmSetManifestError(f"ICC conversion failed: {relative.as_posix()}")
    if not np.all(np.isfinite(working.pixels)):
        raise FilmSetManifestError(f"non-finite decoded pixels: {relative.as_posix()}")
    return working


def _sample_indices(total: int, count: int, seed: int, identity: str) -> np.ndarray:
    if count < 16 or total < count:
        raise FilmSetManifestError(f"cannot sample {count} unique pixels from {total}")
    digest = hashlib.sha256(f"{seed}:{identity}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return np.sort(rng.choice(total, size=count, replace=False))


def _fold(contract: CT5DataContract, cluster_id: str) -> str:
    digest = hashlib.sha256(f"{contract.seed}:{cluster_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return "pilot" if value < contract.pilot_fraction else "confirmatory"


def _verify_hash(path: Path, expected: str) -> None:
    observed = _sha256_file(path)
    if observed.lower() != expected.lower():
        raise FilmSetManifestError(f"manifest hash mismatch for {path.name}: {observed}")


def _sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
