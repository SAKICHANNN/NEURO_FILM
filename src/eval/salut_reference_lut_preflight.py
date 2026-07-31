"""Audit the pinned SA-LUT release before any quality comparison.

This module deliberately does not repair or vendor SA-LUT.  It checks the
published checkpoint and the execution-critical source contracts and records
whether the official CPU/GPU implementation is a valid reproducible baseline.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import torch


SCHEMA = "neuro_film.u5_r2bl12_salut_reference_lut_preflight_report.v1"


class SalutPreflightError(RuntimeError):
    """Raised when the frozen acquisition or report contract drifts."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def cuda_forward_max_index(
    *, dim: int, context: float, red: float, green: float, blue: float
) -> int:
    """Maximum LUT-plane index used by the published CUDA forward kernel."""

    del context
    binsize = 1.000001 / (dim - 1)
    r_id = int(red // binsize)
    g_id = int(green // binsize)
    b_id = int(blue // binsize)
    return dim**3 + (r_id + 1) + (g_id + 1) * dim + (b_id + 1) * dim**2


def cpu_forward_max_index(
    *, dim: int, context: float, red: float, green: float, blue: float
) -> int:
    """Maximum LUT-plane index used by the published CPU forward kernel."""

    binsize = 1.000001 / (dim - 1)
    context_id = int(context // binsize)
    r_id = int(red // binsize)
    g_id = int(green // binsize)
    b_id = int(blue // binsize)
    return (
        context_id
        + 1
        + (r_id + 1) * dim
        + (g_id + 1) * dim**2
        + (b_id + 1) * dim**3
    )


def dense_attention_bytes(*, height: int, width: int, stride: int = 2) -> int:
    """Bytes in the published float32 QK attention matrix."""

    tokens = ((height + stride - 1) // stride) * (
        (width + stride - 1) // stride
    )
    return tokens * tokens * 4


def _git(repo: Path, *args: str) -> str:
    safe = repo.as_posix()
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={safe}", "-C", str(repo), *args],
        text=True,
    ).strip()


def run_salut_preflight(
    *, root: Path, config: Mapping[str, Any], output_dir: Path
) -> dict[str, Any]:
    acquisition = config["acquisition"]
    source = config["primary_source"]
    acquired = root / acquisition["root"]
    repo = acquired / acquisition["repository_subdir"]
    checkpoint = acquired / acquisition["checkpoint_subpath"]

    if not repo.is_dir() or not checkpoint.is_file():
        raise SalutPreflightError("pinned SA-LUT acquisition is incomplete")
    commit = _git(repo, "rev-parse", "HEAD")
    dirty = _git(repo, "status", "--porcelain")
    checkpoint_bytes = checkpoint.stat().st_size
    checkpoint_sha = _sha256_file(checkpoint)
    if commit != source["repository_commit"] or dirty:
        raise SalutPreflightError("repository identity or cleanliness drift")
    if checkpoint_bytes != source["checkpoint_expected_bytes"]:
        raise SalutPreflightError("checkpoint byte count drift")

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise SalutPreflightError("checkpoint is not a state mapping")
    model_state = {
        key.removeprefix("vlog2stylenet."): value
        for key, value in state.items()
        if key.startswith("vlog2stylenet.")
    }
    id_lut = model_state.get("id_lut")
    if not isinstance(id_lut, torch.Tensor) or tuple(id_lut.shape) != (
        3,
        2,
        17,
        17,
        17,
    ):
        raise SalutPreflightError("checkpoint 4D LUT identity drift")

    dim = 17
    plane_elements = 2 * dim**3
    probe = {"context": 0.5, "red": 0.5, "green": 0.5, "blue": 0.5}
    cuda_max = cuda_forward_max_index(dim=dim, **probe)
    cpu_max = cpu_forward_max_index(dim=dim, **probe)
    cpu_index_in_bounds = cpu_max < plane_elements
    cuda_index_in_bounds = cuda_max < plane_elements
    attention_bytes = dense_attention_bytes(height=512, width=512)

    gates = {
        "repository_exact_and_clean": True,
        "checkpoint_exact_bytes": True,
        "checkpoint_loads_weights_only": True,
        "checkpoint_has_expected_4d_lut": True,
        "published_cuda_probe_in_bounds": cuda_index_in_bounds,
        "published_cpu_probe_in_bounds": cpu_index_in_bounds,
        "published_cpu_cuda_index_semantics_equal": cpu_max == cuda_max,
        "published_dense_attention_fits_local_12gib": attention_bytes
        <= 12 * 1024**3,
        "cpu_smoke_pass": False,
        "cuda_smoke_pass": False,
        "repeat_output_identity_pass": False,
    }
    automatic_gate_pass = all(gates.values())
    stable = {
        "schema": SCHEMA,
        "experiment_id": config["experiment_id"],
        "repository_commit": commit,
        "repository_license_sha256": _sha256_file(repo / "LICENSE"),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_bytes": checkpoint_bytes,
        "checkpoint_key_count": len(state),
        "model_key_count": len(model_state),
        "id_lut_shape": list(id_lut.shape),
        "published_interpolation_probe": {
            **probe,
            "dim": dim,
            "lut_plane_elements": plane_elements,
            "cuda_max_index": cuda_max,
            "cpu_max_index": cpu_max,
        },
        "published_attention": {
            "input_height": 512,
            "input_width": 512,
            "tokens": 65536,
            "float32_dense_matrix_bytes": attention_bytes,
            "local_vram_bytes": 12 * 1024**3,
        },
        "gates": gates,
        "automatic_gate_pass": automatic_gate_pass,
        "decision": (
            "open_bounded_quality_comparison"
            if automatic_gate_pass
            else "close_direct_reproduction_retain_literature_baseline"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    stable_id = hashlib.sha256(_canonical_bytes(stable)).hexdigest()
    report = {**stable, "stable_evidence_id": stable_id}
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.json").write_bytes(_canonical_bytes(report))
    return report


__all__ = [
    "SalutPreflightError",
    "cpu_forward_max_index",
    "cuda_forward_max_index",
    "dense_attention_bytes",
    "run_salut_preflight",
]
