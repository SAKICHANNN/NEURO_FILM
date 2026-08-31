from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import scripts.audit_u7_2o_product_look_cli_entry as parent

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/render_film.py"
CONFIG = ROOT / "configs/u7_2p_product_cli_research_dependency_isolation_v1.json"
CONTRACT = (
    ROOT / "docs/planning/U7_2P_PRODUCT_CLI_RESEARCH_DEPENDENCY_ISOLATION_CONTRACT.md"
)
PARENT_EVIDENCE = ROOT / "docs/evidence/U7_2O_PRODUCT_LOOK_CLI_ENTRY_RESULT.json"
PARENT_FORWARD = ROOT / "outputs/eval/u7_2o_product_look_cli_entry_v1/formal.json"
PARENT_REVERSE = (
    ROOT / "outputs/eval/u7_2o_product_look_cli_entry_v1/formal_replay.json"
)
SCRATCH_PARENT = ROOT / "tmp"

CONTRACT_COMMIT = "96d6d10a"
IMPLEMENTATION_COMMIT = "657b563"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sitecustomize(path: Path, blocked: tuple[str, ...]) -> None:
    path.write_text(
        "import atexit\n"
        "import importlib.abc\n"
        "import json\n"
        "import os\n"
        "import pathlib\n"
        "import sys\n"
        f"BLOCKED = {blocked!r}\n"
        "ATTEMPTS = []\n"
        "class Blocker(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if any(fullname == root or fullname.startswith(root + '.') "
        "for root in BLOCKED):\n"
        "            ATTEMPTS.append(fullname)\n"
        "            raise ImportError(f'U7.2P blocked import: {fullname}')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "def record():\n"
        "    loaded = sorted(name for name in sys.modules if any("
        "name == root or name.startswith(root + '.') for root in BLOCKED))\n"
        "    directory = pathlib.Path(os.environ['U7_2P_IMPORT_LOG_DIR'])\n"
        "    directory.mkdir(parents=True, exist_ok=True)\n"
        "    payload = {'attempts': ATTEMPTS, 'loaded': loaded}\n"
        "    (directory / f'{os.getpid()}.json').write_text("
        "json.dumps(payload, sort_keys=True), encoding='utf-8')\n"
        "atexit.register(record)\n",
        encoding="utf-8",
    )


def _blocked_environment(probe_root: Path, blocked: tuple[str, ...]) -> dict[str, str]:
    probe_root.mkdir()
    log_root = probe_root / "logs"
    log_root.mkdir()
    _sitecustomize(probe_root / "sitecustomize.py", blocked)
    env = dict(os.environ)
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(probe_root) if not current else f"{probe_root}{os.pathsep}{current}"
    )
    env["U7_2P_IMPORT_LOG_DIR"] = str(log_root)
    return env


def _run_with_environment(
    env: dict[str, str], *arguments: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _observations(log_root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(log_root.glob("*.json"))
    ]


def build_report(config_path: Path, order: str) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    blocked = tuple(config["blocked_import_roots"])
    parent_evidence = json.loads(PARENT_EVIDENCE.read_text(encoding="utf-8"))
    parent_report = json.loads(PARENT_FORWARD.read_text(encoding="utf-8"))
    parent_reverse_exact = (
        PARENT_FORWARD.read_bytes() == PARENT_REVERSE.read_bytes()
        and PARENT_FORWARD.stat().st_size == config["u7_2o_formal_report_bytes"]
        and _sha256(PARENT_FORWARD) == config["u7_2o_formal_report_sha256"]
    )
    source_locks = {
        "config_sha256": _sha256(CONFIG),
        "contract_sha256": _sha256(CONTRACT),
        "implementation_sha256": _sha256(SCRIPT),
        "parent_evidence_sha256": _sha256(PARENT_EVIDENCE),
        "parent_report_sha256": _sha256(PARENT_FORWARD),
        "parent_reverse_report_sha256": _sha256(PARENT_REVERSE),
    }
    expected_parent = {
        (row["look"], float(row["amount"])): row for row in parent_report["comparisons"]
    }

    SCRATCH_PARENT.mkdir(parents=True, exist_ok=True)
    scratch_path: Path | None = None
    with tempfile.TemporaryDirectory(prefix="u7_2p_", dir=SCRATCH_PARENT) as raw:
        scratch_path = Path(raw)
        probe_root = scratch_path / "import_probe"
        env = _blocked_environment(probe_root, blocked)
        source = scratch_path / "source.png"
        parent._source(source)
        source_sha = _sha256(source)

        original_run = parent._run

        def blocked_run(*arguments: str) -> subprocess.CompletedProcess[str]:
            return _run_with_environment(env, *arguments)

        parent._run = blocked_run
        try:
            ordered = [
                (look, amount)
                for look in config["product_looks"]
                for amount in config["look_amounts"]
            ]
            if order == "reverse":
                ordered.reverse()
            comparisons = [
                parent._comparison(scratch_path, source, look=look, amount=amount)
                for look, amount in ordered
            ]
            comparisons.sort(key=lambda row: (row["look"], row["amount"]))
            bundle = parent._bundle(scratch_path, source)
            conflicts = parent._conflicts(scratch_path)
            conflicts.sort(key=lambda row: row["name"])
            invalid = parent._non_product_values(scratch_path)
            invalid.sort(key=lambda row: row["value"])
            discovery = parent._run("--list-product-looks")
            legacy = scratch_path / "legacy.png"
            legacy_run = parent._run(
                str(source), "--write-recipe", "--output", str(legacy)
            )
            legacy_sha256 = _sha256(legacy) if legacy.is_file() else None
        finally:
            parent._run = original_run

        product_observations = _observations(probe_root / "logs")
        product_process_count = len(product_observations)
        product_imports_clean = all(
            row == {"attempts": [], "loaded": []} for row in product_observations
        )

        logs_before_analytic = {
            path.name for path in (probe_root / "logs").glob("*.json")
        }
        blocked_analytic_output = scratch_path / "blocked-analytic.png"
        blocked_analytic = _run_with_environment(
            env,
            str(source),
            "--color-engine",
            "analytic-y-chromaticity",
            "--output",
            str(blocked_analytic_output),
        )
        analytic_logs = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in (probe_root / "logs").glob("*.json")
            if path.name not in logs_before_analytic
        ]
        analytic_block_exact = (
            blocked_analytic.returncode != 0
            and "U7.2P blocked import: src.eval" in blocked_analytic.stderr
            and not blocked_analytic_output.exists()
            and len(analytic_logs) == 1
            and any(
                name == "src.eval" or name.startswith("src.eval.")
                for name in analytic_logs[0]["attempts"]
            )
        )

        clean_env = dict(os.environ)
        analytic_output = scratch_path / "analytic.png"
        analytic_run = _run_with_environment(
            clean_env,
            str(source),
            "--color-engine",
            "analytic-y-chromaticity",
            "--output",
            str(analytic_output),
        )
        analytic_unblocked_exact = (
            analytic_run.returncode == 0 and analytic_output.is_file()
        )
        analytic_output_sha256 = (
            _sha256(analytic_output) if analytic_output.is_file() else None
        )
        source_immutable = _sha256(source) == source_sha

    assert scratch_path is not None
    residue_zero = not scratch_path.exists()
    comparisons_exact = all(
        row == expected_parent[(row["look"], float(row["amount"]))]
        for row in comparisons
    )
    bundle_exact = bundle == parent_report["full_bundle"]
    conflicts_exact = conflicts == sorted(
        parent_report["conflicts"], key=lambda row: row["name"]
    )
    invalid_exact = invalid == sorted(
        parent_report["non_product_values"], key=lambda row: row["value"]
    )
    discovery_exact = (
        discovery.returncode == 0
        and hashlib.sha256(discovery.stdout.encode("utf-8")).hexdigest()
        == parent_report["discovery_stdout_sha256"]
    )
    legacy_expected = json.loads(parent.U7_2H_CONFIG.read_text(encoding="utf-8"))[
        "prechange_oracle"
    ]["legacy_default"]["output_sha256"]
    legacy_exact = legacy_run.returncode == 0 and legacy_sha256 == legacy_expected
    parent_evidence_exact = (
        parent_evidence["decision"] == "PASS_PRIVATE_U7_2O_PRODUCT_LOOK_CLI_ENTRY"
        and parent_evidence["formal_reports"]["sha256"]
        == config["u7_2o_formal_report_sha256"]
    )
    gates = {
        "analytic_branch_imports_only_when_selected": analytic_block_exact,
        "analytic_branch_still_executes_unblocked": analytic_unblocked_exact,
        "all_three_looks_all_amounts_exact_to_u7_2o": comparisons_exact,
        "conflict_controls_unchanged": conflicts_exact,
        "discovery_exact_to_u7_2o": discovery_exact,
        "full_bundle_exact_to_u7_2o": bundle_exact,
        "legacy_output_unchanged": legacy_exact,
        "non_product_values_unchanged": invalid_exact,
        "owned_runtime_residue_zero": residue_zero,
        "parent_evidence_and_reports_exact": parent_evidence_exact
        and parent_reverse_exact,
        "product_processes_import_no_blocked_research_roots": product_imports_clean,
        "source_immutable": source_immutable,
    }
    return {
        "schema_version": "neuro-film.u7-2p-product-cli-research-dependency-isolation-result.v1",
        "status": "PASS" if all(gates.values()) else "FAIL_CLOSED",
        "contract_commit": CONTRACT_COMMIT,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "source_locks": source_locks,
        "blocked_import_roots": list(blocked),
        "product_process_count": product_process_count,
        "comparisons": comparisons,
        "full_bundle": bundle,
        "conflicts": conflicts,
        "non_product_values": invalid,
        "discovery_stdout_sha256": hashlib.sha256(
            discovery.stdout.encode("utf-8")
        ).hexdigest(),
        "analytic_output_sha256": analytic_output_sha256,
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
