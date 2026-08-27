"""Train and evaluate the frozen P302 WildRelight development leaf."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.wildrelight_spatial_envmap_explicit import (
    apply_log_gain_grid,
    central_crop,
    environment_difference_features,
    fit_ridge,
    log_grid_descriptor,
    predict_ridge,
    source_dct_features,
    spherical_harmonic_coefficients,
    target_log_gain_grid,
)

ROOT = Path(__file__).resolve().parents[1]


class P302Error(RuntimeError):
    """Raised when a frozen P302 execution boundary differs."""


def _sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray, dtype: str = "<f4") -> str:
    return _sha256(np.ascontiguousarray(values, dtype=dtype).tobytes())


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _load_manifest(config: dict[str, Any]) -> dict[str, Any]:
    binding = config["source"]["member_manifest"]
    if not isinstance(binding, dict):
        raise P302Error("P302 member manifest is not frozen")
    path = ROOT / str(binding["path"])
    body = path.read_bytes()
    if len(body) != int(binding["bytes"]) or _sha256(body) != str(binding["sha256"]):
        raise P302Error("P302 member manifest identity differs")
    return json.loads(body)


def _verify_bindings(config: dict[str, Any]) -> list[dict[str, object]]:
    bindings = config.get("bindings")
    if not isinstance(bindings, list) or not bindings:
        raise P302Error("P302 execution bindings are not frozen")
    states: list[dict[str, object]] = []
    for binding in bindings:
        if not isinstance(binding, dict):
            raise P302Error("P302 execution binding is invalid")
        relative = Path(str(binding.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise P302Error("P302 execution binding path is invalid")
        body = (ROOT / relative).read_bytes()
        state = {
            "bytes": len(body),
            "path": relative.as_posix(),
            "sha256": _sha256(body),
        }
        if state["bytes"] != int(binding.get("bytes", -1)) or state["sha256"] != str(
            binding.get("sha256", "")
        ):
            raise P302Error("P302 execution binding differs")
        states.append(state)
    return states


def _local_root(config: dict[str, Any]) -> Path:
    relative = Path(str(config["source"]["local_root"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise P302Error("P302 local root is invalid")
    return ROOT / relative


def _local_path(root: Path, remote_path: str) -> Path:
    prefix = "small-aligned/"
    if not remote_path.startswith(prefix):
        raise P302Error("P302 member path is invalid")
    relative = Path(remote_path.removeprefix(prefix))
    if relative.is_absolute() or ".." in relative.parts:
        raise P302Error("P302 member path escapes source root")
    return root / relative


def _members_by_path(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    members = {str(member["path"]): member for member in manifest["members"]}
    if len(members) != len(manifest["members"]):
        raise P302Error("P302 member manifest contains duplicate paths")
    return members


def _verify_roles(
    config: dict[str, Any], manifest: dict[str, Any], roles: tuple[str, ...]
) -> list[dict[str, object]]:
    root = _local_root(config)
    states: list[dict[str, object]] = []
    for member in manifest["members"]:
        if member["role"] not in roles:
            continue
        path = _local_path(root, str(member["path"]))
        exists = path.is_file()
        size = path.stat().st_size if exists else -1
        digest = _sha256_file(path) if exists and size == int(member["bytes"]) else ""
        states.append(
            {
                "bytes": size,
                "path": str(member["path"]),
                "sha256": digest,
                "valid": bool(
                    exists
                    and size == int(member["bytes"])
                    and digest == str(member["sha256"])
                ),
            }
        )
    if not states or not all(state["valid"] for state in states):
        raise P302Error("P302 required source members are absent or differ")
    return states


def _decode_rgb(path: Path, openexr: Any) -> np.ndarray:
    with openexr.File(str(path)) as exr_file:
        channels = exr_file.channels()
        if "RGB" in channels:
            pixels = channels["RGB"].pixels.copy()
        elif all(channel in channels for channel in ("R", "G", "B")):
            pixels = np.stack(
                [channels[channel].pixels.copy() for channel in ("R", "G", "B")],
                axis=-1,
            )
        else:
            raise P302Error("P302 EXR does not expose RGB channels")
        if len(exr_file.parts) != 1:
            raise P302Error("P302 EXR must contain exactly one part")
    values = np.ascontiguousarray(pixels, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != 3 or not np.isfinite(values).all():
        raise P302Error("P302 EXR RGB payload is invalid")
    if np.any(values < 0.0):
        raise P302Error("P302 frozen nonnegative radiance gate failed")
    return values


def _paths(scene: str, source_index: int, target_index: int) -> dict[str, str]:
    base = f"small-aligned/{scene}"
    cyclic = {1: 3, 3: 5, 5: 1}[target_index]
    return {
        "source_env": f"{base}/envmap/time{source_index}_envmap.exr",
        "target_env": f"{base}/envmap/time{target_index}_envmap.exr",
        "cyclic_env": f"{base}/envmap/time{cyclic}_envmap.exr",
        "source_photo": f"{base}/photo/time{source_index}_hdr.exr",
        "target_photo": f"{base}/photo/time{target_index}_hdr.exr",
    }


def _metric(
    candidate: np.ndarray, target: np.ndarray, source: np.ndarray, cfg: dict[str, Any]
) -> tuple[float, float]:
    fraction = float(cfg["operator"]["crop_fraction"])
    epsilon = float(cfg["operator"]["epsilon"])
    candidate_crop = central_crop(candidate, fraction)
    target_crop = central_crop(target, fraction)
    source_crop = central_crop(source, fraction)
    valid = (source_crop > 0.0) & (target_crop > 0.0)
    valid_fraction = float(np.mean(valid))
    if not np.any(valid):
        raise P302Error("P302 row has no valid score support")
    delta = np.log2(candidate_crop[valid] + epsilon) - np.log2(
        target_crop[valid] + epsilon
    )
    return float(np.sqrt(np.mean(delta * delta))), valid_fraction


def _gradient_ratio(candidate: np.ndarray, source: np.ndarray, epsilon: float) -> float:
    candidate_grad = np.concatenate(
        (
            np.abs(np.diff(candidate.astype(np.float64), axis=0)).reshape(-1),
            np.abs(np.diff(candidate.astype(np.float64), axis=1)).reshape(-1),
        )
    )
    source_grad = np.concatenate(
        (
            np.abs(np.diff(source.astype(np.float64), axis=0)).reshape(-1),
            np.abs(np.diff(source.astype(np.float64), axis=1)).reshape(-1),
        )
    )
    return float(
        np.percentile(candidate_grad, 95) / max(np.percentile(source_grad, 95), epsilon)
    )


def _new_boundary_fraction(candidate: np.ndarray, source: np.ndarray) -> float:
    candidate_boundary = (candidate == 0.0) | (candidate == 1.0)
    source_boundary = (source == 0.0) | (source == 1.0)
    return float(np.mean(candidate_boundary & ~source_boundary))


def _features(
    source: np.ndarray,
    source_env: np.ndarray,
    target_env: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    op = config["operator"]
    crop = central_crop(source, float(op["crop_fraction"]))
    grid = log_grid_descriptor(
        crop,
        grid_rows=int(op["grid_rows"]),
        grid_columns=int(op["grid_columns"]),
        epsilon=float(op["epsilon"]),
    )
    common = grid.reshape(-1)
    candidate = np.concatenate(
        (
            common,
            environment_difference_features(
                source_env, target_env, float(op["epsilon"])
            ),
        )
    )
    image_only = np.concatenate((common, source_dct_features(grid)))
    return candidate, image_only


def _model_bytes(candidate_state: Any, image_state: Any) -> bytes:
    parts = []
    for state in (candidate_state, image_state):
        for value in (state.feature_mean, state.feature_scale, state.coefficients):
            array = np.ascontiguousarray(value, dtype="<f8")
            parts.append(np.asarray(array.shape, dtype="<i8").tobytes())
            parts.append(array.tobytes())
    return b"".join(parts)


def _train(
    config: dict[str, Any], members: dict[str, dict[str, Any]], root: Path, openexr: Any
) -> tuple[Any, Any, str, list[dict[str, object]]]:
    op = config["operator"]
    candidate_features: list[np.ndarray] = []
    image_features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    for scene in config["roles"]["training"]:
        for source_index, target_index in config["roles"]["directed_pairs"]:
            paths = _paths(scene, source_index, target_index)
            if any(path not in members for path in paths.values()):
                raise P302Error("P302 training member is not source-locked")
            source = _decode_rgb(_local_path(root, paths["source_photo"]), openexr)
            target = _decode_rgb(_local_path(root, paths["target_photo"]), openexr)
            source_env = _decode_rgb(_local_path(root, paths["source_env"]), openexr)
            target_env = _decode_rgb(_local_path(root, paths["target_env"]), openexr)
            candidate, image = _features(source, source_env, target_env, config)
            gain = target_log_gain_grid(
                central_crop(source, float(op["crop_fraction"])),
                central_crop(target, float(op["crop_fraction"])),
                grid_rows=int(op["grid_rows"]),
                grid_columns=int(op["grid_columns"]),
                epsilon=float(op["epsilon"]),
                minimum_log2_gain=float(op["minimum_log2_gain"]),
                maximum_log2_gain=float(op["maximum_log2_gain"]),
            )
            candidate_features.append(candidate)
            image_features.append(image)
            targets.append(gain.reshape(-1))
            rows.append(
                {"scene": scene, "source": source_index, "target": target_index}
            )
    x_candidate = np.stack(candidate_features)
    x_image = np.stack(image_features)
    y = np.stack(targets)
    candidate_state = fit_ridge(x_candidate, y, float(op["ridge_lambda"]))
    image_state = fit_ridge(x_image, y, float(op["ridge_lambda"]))
    model_sha = _sha256(_model_bytes(candidate_state, image_state))
    return candidate_state, image_state, model_sha, rows


def _evaluate_development(
    config: dict[str, Any],
    members: dict[str, dict[str, Any]],
    root: Path,
    openexr: Any,
    *,
    reverse: bool,
) -> dict[str, object]:
    candidate_state, image_state, model_sha, training_rows = _train(
        config, members, root, openexr
    )
    op = config["operator"]
    gates_config = config["gates"]
    scenes = list(config["roles"]["development"])
    if reverse:
        scenes.reverse()
    rows: list[dict[str, Any]] = []
    target_reads_before_model_lock = 0
    for scene in scenes:
        pairs = list(config["roles"]["directed_pairs"])
        if reverse:
            pairs.reverse()
        for source_index, target_index in pairs:
            paths = _paths(scene, source_index, target_index)
            if any(path not in members for path in paths.values()):
                raise P302Error("P302 development member is not source-locked")
            source = _decode_rgb(_local_path(root, paths["source_photo"]), openexr)
            source_env = _decode_rgb(_local_path(root, paths["source_env"]), openexr)
            target_env = _decode_rgb(_local_path(root, paths["target_env"]), openexr)
            cyclic_env = _decode_rgb(_local_path(root, paths["cyclic_env"]), openexr)
            candidate_features, image_features = _features(
                source, source_env, target_env, config
            )
            cyclic_features, _ = _features(source, source_env, cyclic_env, config)
            candidate_grid = predict_ridge(candidate_state, candidate_features).reshape(
                4, 4, 3
            )
            image_grid = predict_ridge(image_state, image_features).reshape(4, 4, 3)
            cyclic_grid = predict_ridge(candidate_state, cyclic_features).reshape(
                4, 4, 3
            )
            candidate = apply_log_gain_grid(
                source,
                candidate_grid,
                minimum_log2_gain=float(op["minimum_log2_gain"]),
                maximum_log2_gain=float(op["maximum_log2_gain"]),
            )
            image_only = apply_log_gain_grid(
                source,
                image_grid,
                minimum_log2_gain=float(op["minimum_log2_gain"]),
                maximum_log2_gain=float(op["maximum_log2_gain"]),
            )
            cyclic = apply_log_gain_grid(
                source,
                cyclic_grid,
                minimum_log2_gain=float(op["minimum_log2_gain"]),
                maximum_log2_gain=float(op["maximum_log2_gain"]),
            )
            source_mean = spherical_harmonic_coefficients(
                source_env, float(op["epsilon"])
            )[0]
            target_mean = spherical_harmonic_coefficients(
                target_env, float(op["epsilon"])
            )[0]
            global_gain = (
                source.astype(np.float64) * (target_mean / source_mean)
            ).astype(np.float32)
            hashes_before_target = {
                "candidate": _array_sha256(candidate),
                "cyclic": _array_sha256(cyclic),
                "global_gain": _array_sha256(global_gain),
                "identity": _array_sha256(source),
                "image_only": _array_sha256(image_only),
            }
            target = _decode_rgb(_local_path(root, paths["target_photo"]), openexr)
            candidate_error, valid_fraction = _metric(candidate, target, source, config)
            image_error, _ = _metric(image_only, target, source, config)
            global_error, _ = _metric(global_gain, target, source, config)
            identity_error, _ = _metric(source, target, source, config)
            cyclic_error, _ = _metric(cyclic, target, source, config)
            oracle_grid = target_log_gain_grid(
                central_crop(source, float(op["crop_fraction"])),
                central_crop(target, float(op["crop_fraction"])),
                grid_rows=4,
                grid_columns=4,
                epsilon=float(op["epsilon"]),
                minimum_log2_gain=float(op["minimum_log2_gain"]),
                maximum_log2_gain=float(op["maximum_log2_gain"]),
            )
            oracle = apply_log_gain_grid(
                source,
                oracle_grid,
                minimum_log2_gain=float(op["minimum_log2_gain"]),
                maximum_log2_gain=float(op["maximum_log2_gain"]),
            )
            oracle_error, _ = _metric(oracle, target, source, config)
            control_error = min(image_error, global_error, identity_error)
            reduction = (
                100.0 * (control_error - candidate_error) / max(control_error, 1e-12)
            )
            cyclic_reduction = (
                100.0 * (cyclic_error - candidate_error) / max(cyclic_error, 1e-12)
            )
            oracle_denominator = control_error - oracle_error
            recovery = (
                (control_error - candidate_error) / oracle_denominator
                if oracle_denominator > 1e-12
                else 0.0
            )
            rows.append(
                {
                    "candidate_error": candidate_error,
                    "candidate_hash_before_target_read": hashes_before_target[
                        "candidate"
                    ],
                    "control_error": control_error,
                    "cyclic_error": cyclic_error,
                    "cyclic_reduction_percent": cyclic_reduction,
                    "global_error": global_error,
                    "gradient_ratio": _gradient_ratio(
                        candidate, source, float(op["epsilon"])
                    ),
                    "identity_error": identity_error,
                    "image_only_error": image_error,
                    "new_boundary_fraction": _new_boundary_fraction(candidate, source),
                    "oracle_error": oracle_error,
                    "oracle_gain_recovery": recovery,
                    "output_hashes_before_target_read": hashes_before_target,
                    "reduction_percent": reduction,
                    "scene": scene,
                    "source": source_index,
                    "target": target_index,
                    "valid_fraction": valid_fraction,
                }
            )
    rows.sort(key=lambda row: (row["scene"], row["source"], row["target"]))
    reductions = np.asarray([row["reduction_percent"] for row in rows])
    cyclic_reductions = np.asarray([row["cyclic_reduction_percent"] for row in rows])
    recoveries = np.asarray([row["oracle_gain_recovery"] for row in rows])
    scene_medians = {
        scene: float(
            np.median(
                [row["reduction_percent"] for row in rows if row["scene"] == scene]
            )
        )
        for scene in sorted(config["roles"]["development"])
    }
    summary = {
        "candidate_control_rate": float(np.mean(reductions > 0.0)),
        "candidate_median_reduction_percent": float(np.median(reductions)),
        "candidate_worst_reduction_percent": float(np.min(reductions)),
        "cyclic_control_rate": float(np.mean(cyclic_reductions > 0.0)),
        "cyclic_median_reduction_percent": float(np.median(cyclic_reductions)),
        "maximum_gradient_ratio": float(max(row["gradient_ratio"] for row in rows)),
        "maximum_new_boundary_fraction": float(
            max(row["new_boundary_fraction"] for row in rows)
        ),
        "median_oracle_gain_recovery": float(np.median(recoveries)),
        "minimum_valid_fraction": float(min(row["valid_fraction"] for row in rows)),
        "positive_scene_medians": int(
            sum(value > 0.0 for value in scene_medians.values())
        ),
        "scene_medians": scene_medians,
    }
    gates = {
        "candidate_control_rate": summary["candidate_control_rate"]
        >= float(gates_config["minimum_candidate_control_rate"]),
        "candidate_median_reduction": summary["candidate_median_reduction_percent"]
        >= float(gates_config["minimum_candidate_median_reduction_percent"]),
        "candidate_worst_reduction": summary["candidate_worst_reduction_percent"]
        >= float(gates_config["minimum_candidate_worst_reduction_percent"]),
        "cyclic_control_rate": summary["cyclic_control_rate"]
        >= float(gates_config["minimum_cyclic_control_rate"]),
        "cyclic_median_reduction": summary["cyclic_median_reduction_percent"]
        >= float(gates_config["minimum_cyclic_median_reduction_percent"]),
        "gradient_safety": summary["maximum_gradient_ratio"]
        <= float(gates_config["maximum_candidate_source_p95_gradient_ratio"]),
        "new_boundary": summary["maximum_new_boundary_fraction"]
        <= float(gates_config["maximum_new_boundary_fraction"]),
        "oracle_gain_recovery": summary["median_oracle_gain_recovery"]
        >= float(gates_config["minimum_median_oracle_gain_recovery"]),
        "positive_scene_medians": summary["positive_scene_medians"]
        >= int(gates_config["minimum_positive_scene_medians"]),
        "target_unread_before_model_lock": target_reads_before_model_lock == 0,
        "valid_support": summary["minimum_valid_fraction"]
        >= float(gates_config["minimum_valid_component_fraction"]),
    }
    return {
        "decision": "PASS_PRIVATE_P302_DEVELOPMENT"
        if all(gates.values())
        else "FAIL_CLOSED_P302_DEVELOPMENT",
        "gates": gates,
        "model_bytes": len(_model_bytes(candidate_state, image_state)),
        "model_sha256": model_sha,
        "rows": rows,
        "summary": summary,
        "target_reads_before_model_lock": target_reads_before_model_lock,
        "training_rows": training_rows,
    }


def _worker(config_path: Path, site: Path, *, reverse: bool) -> dict[str, object]:
    sys.path.insert(0, str(site))
    import OpenEXR  # type: ignore[import-not-found]

    config_body = config_path.read_bytes()
    config = json.loads(config_body)
    binding_states = _verify_bindings(config)
    manifest = _load_manifest(config)
    members = _members_by_path(manifest)
    source_states = _verify_roles(config, manifest, ("training", "development"))
    result = _evaluate_development(
        config, members, _local_root(config), OpenEXR, reverse=reverse
    )
    return {
        **result,
        "binding_states": binding_states,
        "config_bytes": len(config_body),
        "config_sha256": _sha256(config_body),
        "confirmation_member_reads": 0,
        "experiment_id": "P302",
        "openexr_version": OpenEXR.__version__,
        "reserve_member_reads": 0,
        "schema": "neuro-film.p302-wildrelight-development-result.v1",
        "source_file_count": len(source_states),
        "source_files_exact": all(state["valid"] for state in source_states),
    }


def execute(config_path: Path, *, reverse: bool) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    wheel = ROOT / "data/vendor/openexr-3.4.15/openexr-3.4.15-cp312-cp312-win_amd64.whl"
    if (
        not wheel.is_file()
        or wheel.stat().st_size != 740197
        or _sha256_file(wheel)
        != "7aa145813acfe10a83d6e89340349379828190b321608e69f2aee426e10c890b"
    ):
        raise P302Error("P302 OpenEXR wheel identity differs")
    temporary = Path(tempfile.mkdtemp(prefix="p302-openexr-"))
    site = temporary / "site"
    worker_output = temporary / "worker.json"
    cleanup = False
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-index",
                "--no-deps",
                "--target",
                str(site),
                str(wheel),
            ],
            check=True,
            capture_output=True,
        )
        command = [
            sys.executable,
            "-m",
            "scripts.run_p302_wildrelight_spatial_envmap_explicit_operator",
            "--worker",
            "--config",
            str(config_path),
            "--site",
            str(site),
            "--output",
            str(worker_output),
        ]
        if reverse:
            command.append("--reverse")
        subprocess.run(command, cwd=ROOT, check=True, capture_output=True)
        report = json.loads(worker_output.read_text(encoding="utf-8"))
    finally:
        shutil.rmtree(temporary)
        cleanup = not temporary.exists()
    report["candidate_count"] = "2/3"
    report["claim_ceiling"] = config["claim_ceiling"]
    report["network_bytes"] = 0
    report["temporary_cleanup"] = cleanup
    report["gates"]["runtime_source_cleanup"] = bool(
        report["source_files_exact"] and cleanup
    )
    if not report["gates"]["runtime_source_cleanup"]:
        report["decision"] = "FAIL_CLOSED_P302_DEVELOPMENT"
    report["scientific_identity"] = "sha256:" + _sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--site", type=Path)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        if args.site is None:
            raise P302Error("P302 worker site is required")
        report = _worker(
            args.config.resolve(), args.site.resolve(), reverse=args.reverse
        )
    else:
        report = execute(args.config.resolve(), reverse=args.reverse)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(report))


if __name__ == "__main__":
    main()
