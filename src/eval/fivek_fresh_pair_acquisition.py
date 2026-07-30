"""Crash-resumable owned acquisition for the frozen fresh FiveK pairs."""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

import tifffile

from src.preprocess import load_working_image


class FiveKFreshPairAcquisitionError(ValueError):
    """Raised when ownership, acquisition, or integrity validation fails."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _load_hashed_json(
    root: Path, path: str, expected_sha256: str
) -> dict[str, Any]:
    resolved = root / path
    if not resolved.is_file() or _sha256(resolved) != expected_sha256:
        raise FiveKFreshPairAcquisitionError(
            f"frozen evidence drift: {path}"
        )
    return json.loads(resolved.read_text(encoding="utf-8"))


def validate_contract(
    root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    if config.get("status") != "contract_frozen_implementation_ready":
        raise FiveKFreshPairAcquisitionError(
            "acquisition contract is not frozen"
        )
    if (
        config.get("training_allowed")
        or config.get("operator_fitting_allowed")
        or config.get("production_integration_allowed")
        or config["ownership"].get("foreign_resource_mutation_allowed")
        or config["ownership"].get("move_or_delete_unrelated_files_allowed")
    ):
        raise FiveKFreshPairAcquisitionError("ownership boundary drift")
    preflight = config["preflight"]
    manifest = _load_hashed_json(
        root, preflight["manifest"], preflight["manifest_sha256"]
    )
    report = _load_hashed_json(
        root, preflight["report"], preflight["report_sha256"]
    )
    if (
        report.get("automatic_pass")
        is not preflight["required_automatic_pass"]
        or len(manifest.get("rows", [])) != preflight["expected_pairs"]
        or manifest.get("expected_bytes") != preflight["expected_bytes"]
    ):
        raise FiveKFreshPairAcquisitionError("preflight is not eligible")
    for key in ("license", "file_list"):
        payload = config["license_inputs"]
        path = root / str(payload[key])
        if not path.is_file() or _sha256(path) != payload[f"{key}_sha256"]:
            raise FiveKFreshPairAcquisitionError(
                f"license input drift: {key}"
            )
    return {"preflight_manifest": manifest}


def ensure_owned_root(path: Path, ownership: Mapping[str, Any]) -> Path:
    root = path.resolve()
    marker = root / ".neuro_film_owner.json"
    expected = {
        "owner": ownership["owner"],
        "purpose": ownership["purpose"],
        "root": root.as_posix(),
    }
    if root.exists():
        if not root.is_dir():
            raise FiveKFreshPairAcquisitionError(
                "owned root is not a directory"
            )
        entries = list(root.iterdir())
        if entries and not marker.is_file():
            raise FiveKFreshPairAcquisitionError(
                "existing external root lacks ownership marker"
            )
        if marker.is_file():
            observed = json.loads(marker.read_text(encoding="utf-8"))
            if observed != expected:
                raise FiveKFreshPairAcquisitionError(
                    "external root ownership mismatch"
                )
    else:
        root.mkdir(parents=True)
    if not marker.exists():
        marker.write_bytes(_canonical_bytes(expected))
    return root


def _download_one(
    *,
    url: str,
    destination: Path,
    expected_bytes: int,
    timeout: int,
    user_agent: str,
    resume: bool,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == expected_bytes:
        return {
            "path": destination.as_posix(),
            "bytes": expected_bytes,
            "sha256": _sha256(destination),
            "reused": True,
        }
    part = destination.with_name(destination.name + ".part")
    offset = part.stat().st_size if resume and part.is_file() else 0
    headers = {"User-Agent": user_agent}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            append = offset > 0 and int(response.status) == 206
            mode = "ab" if append else "wb"
            with part.open(mode) as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
    except (OSError, urllib.error.URLError) as exc:
        raise FiveKFreshPairAcquisitionError(
            f"download failed for {url}: {type(exc).__name__}"
        ) from exc
    if part.stat().st_size != expected_bytes:
        raise FiveKFreshPairAcquisitionError(
            f"download length mismatch for {url}: "
            f"{part.stat().st_size} != {expected_bytes}"
        )
    part.replace(destination)
    return {
        "path": destination.as_posix(),
        "bytes": expected_bytes,
        "sha256": _sha256(destination),
        "reused": False,
    }


def run_acquisition(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
    software_commit: str,
) -> dict[str, Any]:
    validated = validate_contract(root, config)
    ownership = config["ownership"]
    external_root = ensure_owned_root(
        Path(str(ownership["external_root"])), ownership
    )
    download = config["download"]
    free_bytes = shutil.disk_usage(external_root).free
    if free_bytes < int(download["minimum_free_bytes_before_start"]):
        raise FiveKFreshPairAcquisitionError(
            "insufficient free space on owned acquisition volume"
        )
    preflight_rows = validated["preflight_manifest"]["rows"]
    tasks = []
    for row in preflight_rows:
        source_name = str(row["source_name"])
        tasks.extend(
            [
                {
                    "source_name": source_name,
                    "kind": "dng",
                    "url": row["dng_url"],
                    "expected_bytes": int(
                        row["dng_head"]["content_length"]
                    ),
                    "destination": external_root
                    / "dng"
                    / f"{source_name}.dng",
                },
                {
                    "source_name": source_name,
                    "kind": "expert_c",
                    "url": row["expert_c_url"],
                    "expected_bytes": int(
                        row["expert_c_head"]["content_length"]
                    ),
                    "destination": external_root
                    / "expert_c"
                    / f"{source_name}.tif",
                },
            ]
        )
    if (
        len(tasks) != config["preflight"]["expected_assets"]
        or sum(task["expected_bytes"] for task in tasks)
        != config["preflight"]["expected_bytes"]
        or sum(task["expected_bytes"] for task in tasks)
        > download["maximum_total_bytes"]
    ):
        raise FiveKFreshPairAcquisitionError("download inventory drift")
    acquired: dict[tuple[str, str], dict[str, Any]] = {}
    with ThreadPoolExecutor(
        max_workers=int(download["maximum_workers"])
    ) as executor:
        futures = {
            executor.submit(
                _download_one,
                url=task["url"],
                destination=task["destination"],
                expected_bytes=task["expected_bytes"],
                timeout=int(download["timeout_seconds"]),
                user_agent=str(download["user_agent"]),
                resume=bool(download["resume_partial_files"]),
            ): task
            for task in tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            acquired[(task["source_name"], task["kind"])] = future.result()
    license_dir = external_root / "legal"
    license_dir.mkdir(exist_ok=True)
    license_copies = 0
    for key in ("license", "file_list"):
        source = root / str(config["license_inputs"][key])
        destination = license_dir / source.name
        if destination.exists():
            if _sha256(destination) != config["license_inputs"][
                f"{key}_sha256"
            ]:
                raise FiveKFreshPairAcquisitionError(
                    f"owned license copy drift: {destination}"
                )
        else:
            shutil.copy2(source, destination)
        license_copies += 1
    rows = []
    decode_failures = 0
    dimension_mismatches = 0
    for preflight_row in preflight_rows:
        source_name = str(preflight_row["source_name"])
        dng = acquired[(source_name, "dng")]
        expert = acquired[(source_name, "expert_c")]
        try:
            working = load_working_image(Path(dng["path"]))
            source_shape = [int(value) for value in working.pixels.shape]
            with tifffile.TiffFile(expert["path"]) as tiff:
                page = tiff.pages[0]
                target_shape = [int(value) for value in page.shape]
                target_dtype = str(page.dtype)
            if (
                len(target_shape) != 3
                or target_shape[-1] < 3
                or target_dtype != "uint16"
            ):
                decode_failures += 1
        except (OSError, ValueError, tifffile.TiffFileError):
            decode_failures += 1
            source_shape = []
            target_shape = []
            target_dtype = ""
        if (
            source_shape
            and target_shape
            and source_shape[:2] != target_shape[:2]
        ):
            dimension_mismatches += 1
        rows.append(
            {
                "source_name": source_name,
                "license_partition": preflight_row["license_partition"],
                "dng_url": preflight_row["dng_url"],
                "dng_path": dng["path"],
                "dng_bytes": dng["bytes"],
                "dng_sha256": dng["sha256"],
                "dng_reused": dng["reused"],
                "expert_c_url": preflight_row["expert_c_url"],
                "expert_c_path": expert["path"],
                "expert_c_bytes": expert["bytes"],
                "expert_c_sha256": expert["sha256"],
                "expert_c_reused": expert["reused"],
                "source_shape": source_shape,
                "target_shape": target_shape,
                "target_dtype": target_dtype,
                "working_space": (
                    working.working_space if source_shape else None
                ),
                "transfer_state": (
                    working.transfer_state if source_shape else None
                ),
            }
        )
        if source_shape:
            del working
    observed = {
        "pairs": len(rows),
        "assets": len(acquired),
        "downloaded_bytes": sum(
            value["bytes"] for value in acquired.values()
        ),
        "length_failures": 0,
        "hash_failures": 0,
        "decode_failures": decode_failures,
        "dimension_mismatches": dimension_mismatches,
        "license_copies": license_copies,
    }
    gates = {
        key: observed[key] == expected
        for key, expected in config["pass_gates"].items()
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "preflight_manifest_sha256": config["preflight"][
            "manifest_sha256"
        ],
        "external_root": external_root.as_posix(),
        "owner": ownership["owner"],
        "rows": rows,
        "observed": observed,
        "claim_ceiling": config["claim_ceiling"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_bytes(_canonical_bytes(manifest))
    stable_payload = {
        "manifest_sha256": _sha256(manifest_path),
        "observed": observed,
        "gates": gates,
    }
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "config_sha256": _sha256(config_path),
        **stable_payload,
        "automatic_pass": all(gates.values()),
        "stable_evidence_id": hashlib.sha256(
            _canonical_bytes(stable_payload)
        ).hexdigest(),
        "claim_ceiling": config["claim_ceiling"],
    }
    report_path = output_dir / "report.json"
    report_path.write_bytes(_canonical_bytes(report))
    return {
        "manifest_path": manifest_path,
        "manifest_sha256": report["manifest_sha256"],
        "report_path": report_path,
        "report_sha256": _sha256(report_path),
        "report": report,
    }


__all__ = [
    "FiveKFreshPairAcquisitionError",
    "ensure_owned_root",
    "run_acquisition",
    "validate_contract",
]
