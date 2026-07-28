"""Audit fail-closed batch atomicity for exact unsupported media fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.color_match import files as match_files


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("node") != "P164":
        raise ValueError("config must be the P164 schema v1 contract")
    cases = config.get("cases")
    if not isinstance(cases, list) or [row.get("case_id") for row in cases] != [
        "transparent-alpha-second-source",
        "multipage-tiff-second-source",
        "libultrahdr-mpo-second-source",
    ]:
        raise ValueError("config must freeze the exact three-case matrix")
    if config.get("repeat_count_per_case") != 2:
        raise ValueError("P164 requires exactly two repeats per case")
    return config


def _rgb(seed: int, width: int, height: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def _generate_inputs(
    config: dict[str, Any],
    case: dict[str, Any],
    input_dir: Path,
) -> tuple[Path, tuple[Path, Path]]:
    image = config["image"]
    width = int(image["width"])
    height = int(image["height"])
    reference = input_dir / "reference.png"
    first_source = input_dir / "valid-source.png"
    invalid_source = input_dir / "invalid-source"
    Image.fromarray(
        _rgb(int(image["reference_seed"]), width, height),
        mode="RGB",
    ).save(reference)
    Image.fromarray(
        _rgb(int(image["valid_source_seed"]), width, height),
        mode="RGB",
    ).save(first_source)

    kind = case["fixture_kind"]
    invalid_rgb = _rgb(
        int(image["invalid_source_seed"]),
        width,
        height,
    )
    if kind == "generated-rgba-png":
        invalid_source = invalid_source.with_suffix(".png")
        alpha = np.full((height, width, 1), 255, dtype=np.uint8)
        alpha[height // 2, width // 2, 0] = 0
        Image.fromarray(
            np.concatenate((invalid_rgb, alpha), axis=2),
            mode="RGBA",
        ).save(invalid_source)
    elif kind == "generated-two-page-rgb-tiff":
        invalid_source = invalid_source.with_suffix(".tiff")
        first = Image.fromarray(invalid_rgb, mode="RGB")
        second = Image.fromarray(np.flip(invalid_rgb, axis=1), mode="RGB")
        first.save(invalid_source, save_all=True, append_images=[second])
    elif kind == "pinned-libultrahdr-reference":
        fixture = ROOT / case["fixture_path"]
        if _sha256(fixture) != case["fixture_sha256"]:
            raise ValueError("pinned libultrahdr fixture hash mismatch")
        invalid_source = invalid_source.with_suffix(".jpg")
        shutil.copyfile(fixture, invalid_source)
    else:  # pragma: no cover - strict config validation path.
        raise ValueError(f"unsupported fixture kind: {kind}")
    return reference, (first_source, invalid_source)


def _residue(run_dir: Path) -> list[str]:
    markers = (
        ".reference-match-stage",
        ".reference-match-backup",
        ".tmp",
    )
    return sorted(
        str(path.relative_to(run_dir)).replace("\\", "/")
        for path in run_dir.rglob("*")
        if path.is_file() and any(marker in path.name for marker in markers)
    )


def run_case(
    config: dict[str, Any],
    case: dict[str, Any],
    repeat: int,
    output_root: Path,
) -> dict[str, Any]:
    run_dir = output_root / f"{case['case_id']}-{repeat}"
    input_dir = run_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=False)
    reference, sources = _generate_inputs(config, case, input_dir)
    targets = (
        run_dir / "output-1.png",
        run_dir / "output-2.png",
        run_dir / "recipe.json",
        run_dir / "report.json",
    )
    for index, target in enumerate(targets):
        target.write_bytes(f"P164-SENTINEL-{index}\n".encode())
    before = {target.name: _sha256(target) for target in targets}

    encode_call_count = 0
    original_encode = match_files._encode_working_image

    def observed_encode(*args: Any, **kwargs: Any) -> tuple[str, float]:
        nonlocal encode_call_count
        encode_call_count += 1
        return original_encode(*args, **kwargs)

    match_files._encode_working_image = observed_encode
    exception_type: str | None = None
    exception_message: str | None = None
    try:
        match_files.match_reference_files(
            reference,
            sources,
            targets[:2],
            recipe_path=targets[2],
            report_path=targets[3],
            output_bit_depth=16,
        )
    except Exception as exc:  # noqa: BLE001 - the contract records exact failure.
        exception_type = type(exc).__name__
        exception_message = str(exc)
    finally:
        match_files._encode_working_image = original_encode

    after = {target.name: _sha256(target) for target in targets}
    result = {
        "case_id": case["case_id"],
        "repeat": repeat,
        "exception_type": exception_type,
        "exception_message": exception_message,
        "encode_call_count": encode_call_count,
        "input_sha256": {
            "reference": _sha256(reference),
            "valid_source": _sha256(sources[0]),
            "invalid_source": _sha256(sources[1]),
        },
        "target_sha256_before": before,
        "target_sha256_after": after,
        "target_hashes_unchanged": before == after,
        "residual_artifacts": _residue(run_dir),
    }
    result["automatic_pass"] = bool(
        exception_type == config["gates"]["expected_exception_type"]
        and case["expected_rejection_substring"] in (exception_message or "")
        and encode_call_count == 1
        and result["target_hashes_unchanged"]
        and not result["residual_artifacts"]
    )
    return result


def evaluate_runs(
    config: dict[str, Any],
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    case_results: dict[str, Any] = {}
    for case in config["cases"]:
        case_id = case["case_id"]
        selected = [row for row in runs if row["case_id"] == case_id]
        normalized = [
            {
                key: value
                for key, value in row.items()
                if key != "repeat"
            }
            for row in selected
        ]
        exact_replay = bool(
            len(normalized) == config["repeat_count_per_case"]
            and len({_canonical_sha256(row) for row in normalized}) == 1
        )
        case_results[case_id] = {
            "complete": len(selected) == config["repeat_count_per_case"],
            "exact_replay": exact_replay,
            "automatic_pass": bool(
                exact_replay
                and all(row["automatic_pass"] for row in selected)
            ),
        }
    return {
        "case_results": case_results,
        "automatic_pass": bool(
            all(row["automatic_pass"] for row in case_results.values())
        ),
        "claim_ceiling": config["claim_ceiling"],
    }


def _git_text(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def run_audit(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = load_config(config_path)
    candidate = config["candidate_commit"]
    if subprocess.run(
        ["git", "diff", "--quiet", candidate, "--", "src"],
        cwd=ROOT,
        check=False,
    ).returncode != 0:
        raise RuntimeError("current product source differs from P164 candidate")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    runs = [
        run_case(config, case, repeat, output_dir)
        for case in config["cases"]
        for repeat in range(1, int(config["repeat_count_per_case"]) + 1)
    ]
    report = {
        "schema_version": 1,
        "node": "P164",
        "config_sha256": _sha256(config_path),
        "candidate_commit": candidate,
        "execution_head": _git_text("rev-parse", "HEAD"),
        "product_source_diff_from_candidate": False,
        "runs": runs,
        "gate_result": evaluate_runs(config, runs),
        "claim_ceiling": config["claim_ceiling"],
    }
    _write_json(output_dir / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs"
        / "reference_match_unsupported_media_batch_atomicity_v1.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_audit(args.config.resolve(), args.output_dir.resolve())
    print(json.dumps(report["gate_result"], indent=2))
    return 0 if report["gate_result"]["automatic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
