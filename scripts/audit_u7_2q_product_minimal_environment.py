from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u7_2q_product_minimal_environment_v1.json"
CONTRACT = ROOT / "docs/planning/U7_2Q_PRODUCT_MINIMAL_ENVIRONMENT_CONTRACT.md"
REQUIREMENTS = ROOT / "requirements-product.txt"
PARENT_AUDIT = ROOT / "scripts/audit_u7_2p_product_cli_research_dependency_isolation.py"
PARENT_CONFIG = ROOT / "configs/u7_2p_product_cli_research_dependency_isolation_v1.json"
SCRATCH_PARENT = ROOT / "tmp"

CONTRACT_COMMIT = "087ab95d"
IMPLEMENTATION_COMMIT = "49e6fa09"
BOOTSTRAP_DISTRIBUTIONS = {"pip", "setuptools", "wheel"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_distribution(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def _run(
    arguments: list[str], *, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _python_in(venv: Path) -> Path:
    return venv / "Scripts/python.exe"


def _inventory(python: Path, env: dict[str, str]) -> dict[str, str]:
    code = (
        "import importlib.metadata as m,json;"
        "print(json.dumps(sorted((d.metadata['Name'],d.version) "
        "for d in m.distributions()),separators=(',',':')))"
    )
    result = _run([str(python), "-c", code], env=env)
    if result.returncode != 0:
        return {}
    rows = json.loads(result.stdout)
    return {
        _canonical_distribution(str(name)): str(version)
        for name, version in rows
        if _canonical_distribution(str(name)) not in BOOTSTRAP_DISTRIBUTIONS
    }


def _build_environment(
    *, root: Path, environment_id: str, nested_order: str, config: dict[str, Any]
) -> dict[str, Any]:
    venv = root / environment_id
    cache = root / f"{environment_id}-pip-cache"
    nested_report = root / f"{environment_id}-u7_2p.json"
    env = dict(os.environ)
    env["PIP_CACHE_DIR"] = str(cache)
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["PYTHONNOUSERSITE"] = "1"

    create = _run(
        ["py", config["python"]["windows_launcher_selector"], "-m", "venv", str(venv)],
        env=env,
    )
    python = _python_in(venv)
    install = (
        _run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-input",
                "--only-binary=:all:",
                "--requirement",
                str(REQUIREMENTS),
            ],
            env=env,
        )
        if create.returncode == 0 and python.is_file()
        else None
    )
    check = (
        _run([str(python), "-m", "pip", "check"], env=env)
        if install is not None and install.returncode == 0
        else None
    )
    inventory = (
        _inventory(python, env) if check is not None and check.returncode == 0 else {}
    )
    nested = (
        _run(
            [
                str(python),
                str(PARENT_AUDIT),
                "--config",
                str(PARENT_CONFIG),
                "--order",
                nested_order,
                "--output",
                str(nested_report),
            ],
            env=env,
        )
        if inventory == config["required_distributions"]
        else None
    )
    nested_payload = nested_report.read_bytes() if nested_report.is_file() else b""
    nested_json = json.loads(nested_payload) if nested_payload else {}
    return {
        "environment_id": environment_id,
        "python_version": (
            _run([str(python), "--version"], env=env).stdout.strip()
            if python.is_file()
            else None
        ),
        "create_returncode": create.returncode,
        "install_returncode": install.returncode if install is not None else None,
        "pip_check_returncode": check.returncode if check is not None else None,
        "inventory": inventory,
        "nested_order": nested_order,
        "nested_returncode": nested.returncode if nested is not None else None,
        "nested_report_bytes": len(nested_payload),
        "nested_report_sha256": hashlib.sha256(nested_payload).hexdigest()
        if nested_payload
        else None,
        "nested_status": nested_json.get("status"),
        "nested_gates": nested_json.get("gates", {}),
    }


def build_report(config_path: Path, order: str) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = config["required_distributions"]
    source_locks = {
        "config_sha256": _sha256(config_path),
        "contract_sha256": _sha256(CONTRACT),
        "requirements_sha256": _sha256(REQUIREMENTS),
        "render_film_sha256": _sha256(ROOT / "scripts/render_film.py"),
        "u7_2p_evidence_sha256": _sha256(
            ROOT
            / "docs/evidence/U7_2P_PRODUCT_CLI_RESEARCH_DEPENDENCY_ISOLATION_RESULT.json"
        ),
    }
    source_locks_exact = (
        source_locks["render_film_sha256"]
        == config["source_locks"]["render_film_sha256"]
        and source_locks["u7_2p_evidence_sha256"]
        == config["source_locks"]["u7_2p_evidence_sha256"]
    )
    labels = ["environment-a", "environment-b"]
    execution_labels = labels if order == "forward" else list(reversed(labels))
    nested_orders = {"environment-a": "forward", "environment-b": "reverse"}
    work_root: Path | None = None
    rows: list[dict[str, Any]] = []
    try:
        SCRATCH_PARENT.mkdir(parents=True, exist_ok=True)
        work_root = Path(tempfile.mkdtemp(prefix="u7_2q-", dir=SCRATCH_PARENT))
        for label in execution_labels:
            rows.append(
                _build_environment(
                    root=work_root,
                    environment_id=label,
                    nested_order=nested_orders[label],
                    config=config,
                )
            )
    finally:
        if work_root is not None:
            shutil.rmtree(work_root, ignore_errors=True)

    rows.sort(key=lambda row: row["environment_id"])
    inventories_exact = all(row["inventory"] == expected for row in rows)
    forbidden = {
        _canonical_distribution(name) for name in config["forbidden_distribution_roots"]
    }
    forbidden_absent = all(forbidden.isdisjoint(row["inventory"]) for row in rows)
    nested_exact = (
        len(rows) == config["environment_count"]
        and all(row["nested_status"] == "PASS" for row in rows)
        and all(all(row["nested_gates"].values()) for row in rows)
        and len({row["nested_report_sha256"] for row in rows}) == 1
        and rows[0]["nested_report_sha256"]
        == config["source_locks"]["u7_2p_report_sha256"]
        and rows[0]["nested_report_bytes"]
        == config["source_locks"]["u7_2p_report_bytes"]
    )
    gates = {
        "binary_only_fresh_installs_pass": len(rows) == config["environment_count"]
        and all(
            row["create_returncode"] == row["install_returncode"] == 0 for row in rows
        ),
        "forbidden_research_distributions_absent": forbidden_absent,
        "installed_inventories_exact_and_fully_pinned": inventories_exact,
        "nested_u7_2p_reports_byte_exact": nested_exact,
        "owned_runtime_residue_zero": work_root is not None and not work_root.exists(),
        "pip_check_passes": all(row["pip_check_returncode"] == 0 for row in rows),
        "python_3_12_exact": all(
            row["python_version"] == f"Python {config['python']['qualified_patch']}"
            for row in rows
        ),
        "source_locks_exact": source_locks_exact,
    }
    return {
        "schema_version": "neuro-film.u7-2q-product-minimal-environment-result.v1",
        "status": "PASS" if all(gates.values()) else "FAIL_CLOSED",
        "contract_commit": CONTRACT_COMMIT,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "source_locks": source_locks,
        "requirements": expected,
        "environments": rows,
        "gates": gates,
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.config, args.order)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
