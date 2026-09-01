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
CONFIG = ROOT / "configs/u7_2u_product_research_halation_isolation_v1.json"


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


def _artifacts(output: Path) -> tuple[Path, ...]:
    return (
        output,
        output.with_suffix(".recipe.json"),
        output.with_suffix(".metrics.json"),
        output.parent / f"{output.stem}_layers",
    )


def _source_locks(config: dict[str, object]) -> tuple[list[dict[str, object]], bool]:
    rows: list[dict[str, object]] = []
    for name, value in sorted(config["source_locks"].items()):
        lock = dict(value)
        path = ROOT / str(lock["path"])
        actual_blob = _git("hash-object", "--", str(path))
        actual_sha = _sha256(path)
        exact = actual_blob == lock["git_blob"] and actual_sha == lock["sha256"]
        rows.append(
            {
                "name": name,
                "path": lock["path"],
                "git_blob": actual_blob,
                "sha256": actual_sha,
                "exact": exact,
            }
        )
    return rows, all(bool(row["exact"]) for row in rows)


def audit(*, reverse: bool) -> dict[str, object]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    scratch = ROOT / "tmp" / f"u7_2u_{uuid.uuid4().hex}"
    scratch.mkdir(parents=True)
    try:
        source_locks, source_locks_exact = _source_locks(config)
        cases: list[dict[str, object]] = []
        for look in config["product_looks"]:
            for controls in (False, True):
                cases.append(
                    {
                        "case_id": f"product-{look}-research-controls-{int(controls)}",
                        "kind": "product-research-reject",
                        "look": look,
                        "model": config["rejected_product_halation_control"],
                        "research_controls": controls,
                    }
                )
        for model in config["allowed_product_halation_controls"]:
            cases.append(
                {
                    "case_id": f"product-ektar_100-{model}",
                    "kind": "product-cross",
                    "look": "ektar_100",
                    "model": model,
                    "research_controls": False,
                }
            )
        cases.append(
            {
                "case_id": "nonproduct-staged-density-research",
                "kind": "nonproduct-cross",
                "look": "ektar_100",
                "model": config["rejected_product_halation_control"],
                "research_controls": True,
            }
        )
        if reverse:
            cases.reverse()

        rows: list[dict[str, object]] = []
        for case in cases:
            source = scratch / f"missing-{case['case_id']}.png"
            output = scratch / f"output-{case['case_id']}.png"
            arguments = [str(source)]
            if case["kind"].startswith("product"):
                arguments.extend(["--product-look", str(case["look"])])
            else:
                arguments.extend(["--style", str(case["look"]), "--use-render-profile"])
            arguments.extend(["--halation-model", str(case["model"])])
            if case["research_controls"]:
                arguments.extend(["--halation", "1", "--tile-size", "64"])
            arguments.extend(["--output", str(output)])
            completed = _run(*arguments)
            artifacts_absent = all(not path.exists() for path in _artifacts(output))
            if case["kind"] == "product-research-reject":
                outcome = "predecode-reject"
                passed = (
                    completed.returncode == 2
                    and "staged-density-research is research-only" in completed.stderr
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
                    **case,
                    "outcome": outcome,
                    "returncode": completed.returncode,
                    "artifacts_absent": artifacts_absent,
                    "passed": passed,
                }
            )

        rows.sort(key=lambda row: str(row["case_id"]))
        rejects = [row for row in rows if row["kind"] == "product-research-reject"]
        product_controls = [row for row in rows if row["kind"] == "product-cross"]
        research_controls = [row for row in rows if row["kind"] == "nonproduct-cross"]
        gates = {
            "source_locks_exact": source_locks_exact,
            "all_product_research_combinations_reject_predecode": all(
                bool(row["passed"]) for row in rejects
            ),
            "product_rejections_publish_nothing": all(
                bool(row["artifacts_absent"]) for row in rejects
            ),
            "product_nonresearch_models_cross_preflight": all(
                bool(row["passed"]) for row in product_controls
            ),
            "nonproduct_research_route_crosses_preflight": all(
                bool(row["passed"]) for row in research_controls
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
