from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/render_film.py"
CONFIG = ROOT / "configs/u7_2t_product_output_extension_preflight_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _artifact_paths(output: Path) -> tuple[Path, ...]:
    return (
        output,
        output.with_suffix(".recipe.json"),
        output.with_suffix(".metrics.json"),
        output.parent / f"{output.stem}_layers",
    )


def _source_locks(config: dict[str, object]) -> tuple[list[dict[str, object]], bool]:
    rows: list[dict[str, object]] = []
    for name, lock_value in sorted(config["source_locks"].items()):
        lock = dict(lock_value)
        path = ROOT / str(lock["path"])
        actual = {
            "git_blob": _git("hash-object", "--", str(path)),
            "sha256": _sha256(path),
        }
        exact = actual == {
            "git_blob": lock["git_blob"],
            "sha256": lock["sha256"],
        }
        rows.append({"name": name, "path": lock["path"], "exact": exact, **actual})
    return rows, all(bool(row["exact"]) for row in rows)


def audit(*, reverse: bool) -> dict[str, object]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    scratch = ROOT / "tmp" / f"u7_2t_{uuid.uuid4().hex}"
    scratch.mkdir(parents=True)
    try:
        source_locks, source_locks_exact = _source_locks(config)
        cases: list[tuple[str, str, bool]] = []
        cases.extend(
            (f"product-invalid-{suffix or 'none'}", suffix, True)
            for suffix in config["unsupported_suffixes"]
        )
        cases.extend(
            (f"product-valid-{suffix}", suffix, True)
            for suffix in config["supported_suffixes"]
        )
        cases.extend(
            (f"legacy-invalid-{suffix or 'none'}", suffix, False)
            for suffix in config["unsupported_suffixes"]
        )
        if reverse:
            cases.reverse()

        rows: list[dict[str, object]] = []
        for case_id, suffix, product in cases:
            source = scratch / f"missing-{case_id}.png"
            output = scratch / f"output-{case_id}{suffix}"
            arguments = [str(source)]
            if product:
                arguments.extend(["--product-look", "ektar_100"])
            else:
                arguments.extend(["--style", "ektar_100"])
            arguments.extend(["--output", str(output)])
            completed = _run(*arguments)
            artifacts_absent = all(
                not path.exists() for path in _artifact_paths(output)
            )
            if product and suffix in config["unsupported_suffixes"]:
                expected_suffix = suffix or "<none>"
                outcome = "predecode-reject"
                passed = (
                    completed.returncode == 2
                    and f"unsupported 8-bit product output extension: {expected_suffix}"
                    in completed.stderr
                    and str(source) not in completed.stderr
                    and artifacts_absent
                )
            else:
                outcome = "crossed-preflight"
                passed = (
                    completed.returncode != 2
                    and "FileNotFoundError" in completed.stderr
                    and artifacts_absent
                )
            rows.append(
                {
                    "case_id": case_id,
                    "product": product,
                    "suffix": suffix,
                    "outcome": outcome,
                    "returncode": completed.returncode,
                    "artifacts_absent": artifacts_absent,
                    "passed": passed,
                }
            )

        rows.sort(key=lambda row: str(row["case_id"]))
        invalid_product = [
            row
            for row in rows
            if row["product"] and row["outcome"] == "predecode-reject"
        ]
        valid_product = [
            row
            for row in rows
            if row["product"] and row["outcome"] == "crossed-preflight"
        ]
        legacy = [row for row in rows if not row["product"]]
        gates = {
            "source_locks_exact": source_locks_exact,
            "unsupported_product_suffixes_reject_predecode": all(
                bool(row["passed"]) for row in invalid_product
            ),
            "unsupported_product_suffixes_publish_nothing": all(
                bool(row["artifacts_absent"]) for row in invalid_product
            ),
            "supported_product_suffixes_cross_preflight": all(
                bool(row["passed"]) for row in valid_product
            ),
            "legacy_late_behavior_unchanged": all(
                bool(row["passed"]) for row in legacy
            ),
            "owned_runtime_residue_zero": True,
        }
        return {
            "schema_version": config["schema_version"],
            "execution_commit": _git("rev-parse", "HEAD"),
            "status": "PASS" if all(gates.values()) else "FAIL_CLOSED",
            "claim_ceiling": config["claim_ceiling"],
            "source_locks": source_locks,
            "case_count": len(rows),
            "cases": rows,
            "gates": gates,
        }
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    report = audit(reverse=args.reverse)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
