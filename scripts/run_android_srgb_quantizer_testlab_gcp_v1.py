"""Run one owned Firebase Test Lab matrix with strict cleanup and evidence."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any
from urllib import error, request


ROOT = Path(__file__).resolve().parents[1]
GCLOUD = Path(
    r"C:\Users\hhvrf\AppData\Local\Google\Cloud SDK"
    r"\google-cloud-sdk\bin\gcloud.cmd"
)
CONFIGURATION = "nf-019f9f37-agent"
RESOURCE_PREFIX = "nf-019f9f37-"
PACKAGE_REPORT = (
    ROOT / "outputs/eval/android_srgb_quantizer_testlab_package_v1.json"
)
LEDGER = ROOT / "outputs/cloud/nf-019f9f37-ownership.json"
LOCAL_RESULTS = ROOT / "outputs/cloud/nf-019f9f37-p90-results"
RUNTIME_REPORT = (
    ROOT / "outputs/eval/android_srgb_quantizer_device_runtime_v1.json"
)
BUDGET_USD = 2500.0
WARNING_USD = 2000.0
STOP_NEW_USD = 2250.0
WORST_CASE_USD = 1.0
PENDING_CANCEL_SECONDS = 120.0
OVERALL_SECONDS = 600.0
POLL_SECONDS = 10.0
OFFICIAL_DOCS = (
    "https://firebase.google.com/docs/test-lab/usage-quotas-pricing",
    "https://docs.cloud.google.com/sdk/gcloud/reference/firebase/test/android/run",
    "https://docs.cloud.google.com/storage/docs/creating-buckets",
    "https://docs.cloud.google.com/storage/docs/uniform-bucket-level-access",
)
REQUIRED_SERVICES = frozenset(
    {
        "storage.googleapis.com",
        "testing.googleapis.com",
        "toolresults.googleapis.com",
    }
)


class CloudRuntimeError(RuntimeError):
    """Raised when the owned cloud leaf cannot satisfy its strict contract."""


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _gcloud(
    arguments: list[str],
    *,
    timeout: float = 120.0,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    if not GCLOUD.is_file():
        raise CloudRuntimeError("gcloud executable is unavailable")
    completed = subprocess.run(
        [
            os.environ.get("COMSPEC", "cmd.exe"),
            "/d",
            "/c",
            str(GCLOUD),
            *arguments,
            f"--configuration={CONFIGURATION}",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env={**os.environ, "CLOUDSDK_CORE_DISABLE_PROMPTS": "1"},
    )
    if check and completed.returncode != 0:
        raise CloudRuntimeError(
            f"gcloud command failed with exit {completed.returncode}"
        )
    return completed


def _configuration_value(name: str) -> str:
    completed = _gcloud(
        ["config", "get-value", name],
        timeout=30.0,
    )
    value = completed.stdout.strip()
    if not value or value == "(unset)":
        raise CloudRuntimeError(f"gcloud configuration lacks {name}")
    return value


def _access_token() -> str:
    token = _gcloud(
        ["auth", "print-access-token"],
        timeout=30.0,
    ).stdout.strip()
    if not token:
        raise CloudRuntimeError("gcloud access token is unavailable")
    return token


def _api_json(
    method: str,
    url: str,
    *,
    body: bytes | None = None,
    allowed_status: tuple[int, ...] = (),
) -> tuple[int, dict[str, Any] | None]:
    headers = {"Authorization": f"Bearer {_access_token()}"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    operation = request.Request(
        url,
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with request.urlopen(operation, timeout=30.0) as response:
            payload = response.read()
            return response.status, (
                json.loads(payload) if payload else None
            )
    except error.HTTPError as exc:
        if exc.code in allowed_status:
            return exc.code, None
        raise CloudRuntimeError(
            f"Cloud Testing API returned HTTP {exc.code}"
        ) from None


def _find_matrix_id(value: Any) -> str | None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key.casefold() in {"matrixid", "testmatrixid"}:
                candidate = str(item)
                if re.fullmatch(r"[A-Za-z0-9_-]+", candidate):
                    return candidate
            found = _find_matrix_id(item)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_matrix_id(item)
            if found is not None:
                return found
    return None


def _submission_matrix_id(
    completed: subprocess.CompletedProcess[str],
) -> str | None:
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        payload = None
    matrix_id = _find_matrix_id(payload)
    if matrix_id is not None:
        return matrix_id
    match = re.search(
        r"\bmatrix-[A-Za-z0-9_-]+\b",
        completed.stdout + completed.stderr,
    )
    return match.group(0) if match is not None else None


def _matrix_failure_codes(matrix: dict[str, Any]) -> list[str]:
    codes: set[str] = set()
    detail = matrix.get("invalidMatrixDetails")
    if isinstance(detail, str) and re.fullmatch(r"[A-Z0-9_]+", detail):
        codes.add(detail)
    extended = matrix.get("extendedInvalidMatrixDetails")
    if isinstance(extended, list):
        for item in extended:
            if not isinstance(item, dict):
                continue
            reason = item.get("reason")
            if isinstance(reason, str) and re.fullmatch(
                r"[A-Z0-9_]+", reason
            ):
                codes.add(reason)
    return sorted(codes)


def _enabled_services(project: str) -> set[str]:
    completed = _gcloud(
        [
            "services",
            "list",
            "--enabled",
            f"--project={project}",
            "--format=value(config.name)",
        ],
        timeout=60.0,
    )
    return {
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip()
    }


def _new_ledger(project: str) -> dict[str, Any]:
    return {
        "schema": "neuro-film.gcp-ownership-ledger.v1",
        "configuration": CONFIGURATION,
        "project": project,
        "resource_prefix": RESOURCE_PREFIX,
        "budget": {
            "hard_limit_usd": BUDGET_USD,
            "warning_usd": WARNING_USD,
            "stop_new_usd": STOP_NEW_USD,
            "cumulative_estimated_usd": 0.0,
            "cumulative_actual_usd": 0.0,
        },
        "official_docs": list(OFFICIAL_DOCS),
        "resources": [],
        "updated_at": _now(),
    }


def _load_ledger(project: str) -> dict[str, Any]:
    if LEDGER.is_file():
        ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
        if (
            ledger.get("configuration") != CONFIGURATION
            or ledger.get("project") != project
            or ledger.get("resource_prefix") != RESOURCE_PREFIX
        ):
            raise CloudRuntimeError("ownership ledger identity mismatch")
        return ledger
    ledger = _new_ledger(project)
    _atomic_json(LEDGER, ledger)
    return ledger


def _save_ledger(ledger: dict[str, Any]) -> None:
    ledger["updated_at"] = _now()
    _atomic_json(LEDGER, ledger)


def _package() -> dict[str, Any]:
    if not PACKAGE_REPORT.is_file():
        raise CloudRuntimeError("package report is unavailable")
    package = json.loads(PACKAGE_REPORT.read_text(encoding="utf-8"))
    if (
        package.get("schema")
        != "neuro-film.android-srgb-quantizer-testlab-package.v1"
        or package.get("status") != "PASS"
        or package.get("package_prefix") != "nf-019f9f37-p90"
    ):
        raise CloudRuntimeError("package report contract mismatch")
    for role in ("app", "test"):
        artifact = package["artifacts"][role]
        path = Path(artifact["path"]).resolve()
        if _sha256(path) != artifact["sha256"]:
            raise CloudRuntimeError(f"{role} APK hash mismatch")
    return package


def preflight() -> dict[str, Any]:
    project = _configuration_value("project")
    account = _configuration_value("account")
    _gcloud(
        ["projects", "describe", project, "--format=none"],
        timeout=30.0,
    )
    billing = json.loads(
        _gcloud(
            [
                "billing",
                "projects",
                "describe",
                project,
                "--format=json",
            ],
            timeout=30.0,
        ).stdout
    )
    if billing.get("billingEnabled") is not True:
        raise CloudRuntimeError("configured project is not billing enabled")
    missing_services = REQUIRED_SERVICES - _enabled_services(project)
    if missing_services:
        raise CloudRuntimeError(
            "required project APIs are not enabled; shared project "
            "configuration remains unchanged"
        )
    package = _package()
    ledger = _load_ledger(project)
    cumulative = float(
        ledger["budget"].get("cumulative_estimated_usd", 0.0)
    )
    if cumulative + WORST_CASE_USD > STOP_NEW_USD:
        raise CloudRuntimeError("thread cloud start threshold reached")
    url = (
        f"https://testing.googleapis.com/v1/projects/{project}/"
        f"testMatrices/{RESOURCE_PREFIX}readonly-preflight"
    )
    status, _ = _api_json("GET", url, allowed_status=(404,))
    if status != 404:
        raise CloudRuntimeError("Cloud Testing API preflight is ambiguous")
    return {
        "project": project,
        "account_present": bool(account),
        "package": package,
        "ledger": ledger,
        "worst_case_usd": WORST_CASE_USD,
        "testing_api_reachable": True,
    }


def _matrix_state(
    project: str,
    matrix_id: str,
) -> dict[str, Any]:
    url = (
        f"https://testing.googleapis.com/v1/projects/{project}/"
        f"testMatrices/{matrix_id}"
    )
    status, payload = _api_json("GET", url)
    if status != 200 or not isinstance(payload, dict):
        raise CloudRuntimeError("matrix state response is invalid")
    return payload


def _cancel_matrix(project: str, matrix_id: str) -> None:
    url = (
        f"https://testing.googleapis.com/v1/projects/{project}/"
        f"testMatrices/{matrix_id}:cancel"
    )
    _api_json("POST", url, body=b"{}")


def _collect_results(bucket: str, results_dir: str) -> list[Path]:
    destination = LOCAL_RESULTS / results_dir
    destination.mkdir(parents=True, exist_ok=False)
    _gcloud(
        [
            "storage",
            "cp",
            "--recursive",
            f"gs://{bucket}/{results_dir}/**",
            str(destination),
        ],
        timeout=180.0,
    )
    return sorted(path for path in destination.rglob("*") if path.is_file())


def _runtime_tokens(files: list[Path]) -> dict[str, Any]:
    required = (
        '"schema":"neuro-film.android-srgb-quantizer-runtime.v1"',
        '"status":"PASS"',
        '"sample_count":4096',
        (
            '"threshold_identity":"'
            "fae645ef1aad04fcd1233631a32f820c"
            'f7696e3ca65d31939acf60d7f123674c"'
        ),
        (
            '"vector_sha256":"'
            "3d4205e51de80392ea7a4e5eccaf05ab"
            '6d322a48475603764c28e47d61aa7628"'
        ),
        '"q8_exact":true',
        '"q16_exact":true',
        '"inner_replay_exact":true',
        '"failure_atomic":true',
        '"icc_exact":true',
        '"eotf_q8_roundtrip_exact":true',
        '"eotf_q16_roundtrip_exact":true',
        '"eotf_failure_atomic":true',
    )
    matched: dict[str, str] = {}
    for path in files:
        if path.stat().st_size > 64 * 1024 * 1024:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        # Test Lab may serialize instrumentation status values as an escaped
        # JSON string inside another result format.  Search both the original
        # representation and one JSON-string unescape layer without accepting
        # partial or reordered evidence.
        representations = (text, text.replace(r"\"", '"'))
        for token in required:
            if any(token in value for value in representations):
                matched[token] = path.name
    missing = [token for token in required if token not in matched]
    if missing:
        raise CloudRuntimeError("device result lacks required runtime tokens")
    return {
        "required_token_count": len(required),
        "matched_file_names": sorted(set(matched.values())),
    }


def execute() -> dict[str, Any]:
    state = preflight()
    project = state["project"]
    package = state["package"]
    ledger = state["ledger"]
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    suffix = package["package_identity"][:8]
    bucket = f"{RESOURCE_PREFIX}p90-{stamp}-{suffix}".lower()
    results_dir = f"p90-{stamp}-{suffix}"
    bucket_uri = f"gs://{bucket}"
    resource = {
        "kind": "storage-bucket",
        "full_name": bucket_uri,
        "region": "US",
        "created_at": None,
        "purpose": "Firebase Test Lab result isolation for P90",
        "labels": {
            "owner": "codex",
            "thread": "019f9f37",
            "project": "neuro-film-color-match",
        },
        "estimated_cost_usd": WORST_CASE_USD,
        "actual_cost_usd": None,
        "max_runtime_seconds": 120,
        "status": "planned",
        "cleanup": {"status": "pending", "at": None},
        "children": [],
    }
    ledger["resources"].append(resource)
    ledger["budget"]["cumulative_estimated_usd"] = (
        float(ledger["budget"]["cumulative_estimated_usd"])
        + WORST_CASE_USD
    )
    _save_ledger(ledger)
    matrix_id: str | None = None
    timeline: list[dict[str, Any]] = []
    matrix: dict[str, Any] | None = None
    local_files: list[Path] = []
    error_name: str | None = None
    try:
        _gcloud(
            [
                "storage",
                "buckets",
                "create",
                bucket_uri,
                f"--project={project}",
                "--location=US",
                "--uniform-bucket-level-access",
                "--public-access-prevention",
                "--soft-delete-duration=0",
                "--quiet",
            ],
            timeout=120.0,
        )
        resource["created_at"] = _now()
        resource["status"] = "created"
        _save_ledger(ledger)
        labels = ",".join(
            f"{key}={value}" for key, value in resource["labels"].items()
        )
        _gcloud(
            [
                "storage",
                "buckets",
                "update",
                bucket_uri,
                f"--update-labels={labels}",
                f"--project={project}",
                "--quiet",
            ],
            timeout=60.0,
        )
        app = package["artifacts"]["app"]["path"]
        test = package["artifacts"]["test"]["path"]
        submission = _gcloud(
            [
                "firebase",
                "test",
                "android",
                "run",
                "--type=instrumentation",
                f"--app={app}",
                f"--test={test}",
                "--device=model=shiba,version=34,locale=en,orientation=portrait",
                "--timeout=2m",
                f"--results-bucket={bucket_uri}",
                f"--results-dir={results_dir}",
                "--results-history-name=nf-019f9f37-p90",
                (
                    "--client-details="
                    f"matrixLabel={RESOURCE_PREFIX}p90-{stamp}"
                ),
                "--num-flaky-test-attempts=0",
                "--no-auto-google-login",
                "--no-record-video",
                "--no-performance-metrics",
                "--async",
                "--format=json",
                f"--project={project}",
                "--quiet",
            ],
            timeout=180.0,
            check=False,
        )
        matrix_id = _submission_matrix_id(submission)
        if matrix_id is None:
            raise CloudRuntimeError(
                "matrix submission failed before a matrix identity was issued"
            )
        matrix_resource = {
            "kind": "firebase-test-matrix",
            "full_name": (
                f"projects/{project}/testMatrices/{matrix_id}"
            ),
            "created_at": _now(),
            "purpose": "P90 Pixel 8 API34 physical device runtime",
            "client_label": f"{RESOURCE_PREFIX}p90-{stamp}",
            "status": (
                "submitted"
                if submission.returncode == 0
                else "submitted-validation-nonzero"
            ),
            "submission_exit": submission.returncode,
            "cleanup": {
                "status": "server-terminates-after-test",
                "at": None,
            },
        }
        resource["children"].append(matrix_resource)
        _save_ledger(ledger)
        started = time.monotonic()
        pending_started = started
        while True:
            matrix = _matrix_state(project, matrix_id)
            current = str(matrix.get("state", "UNKNOWN"))
            timeline.append(
                {
                    "at": _now(),
                    "state": current,
                    "outcome": matrix.get("outcomeSummary"),
                }
            )
            matrix_resource["status"] = current
            _save_ledger(ledger)
            elapsed = time.monotonic() - started
            if current in {"FINISHED", "ERROR", "INVALID"}:
                break
            if current == "PENDING":
                if time.monotonic() - pending_started > PENDING_CANCEL_SECONDS:
                    _cancel_matrix(project, matrix_id)
                    matrix_resource["status"] = "CANCEL_REQUESTED_PENDING_TIMEOUT"
                    _save_ledger(ledger)
                    raise CloudRuntimeError("matrix pending timeout")
            else:
                pending_started = time.monotonic()
            if elapsed > OVERALL_SECONDS:
                _cancel_matrix(project, matrix_id)
                matrix_resource["status"] = "CANCEL_REQUESTED_OVERALL_TIMEOUT"
                _save_ledger(ledger)
                raise CloudRuntimeError("matrix overall timeout")
            time.sleep(POLL_SECONDS)
        if (
            matrix.get("state") != "FINISHED"
            or matrix.get("outcomeSummary") != "SUCCESS"
        ):
            failure_codes = _matrix_failure_codes(matrix)
            blocked = {
                "schema": (
                    "neuro-film.android-srgb-quantizer-device-runtime.v1"
                ),
                "status": "BLOCKED_PRE_DEVICE",
                "claim_ceiling": "no-device-runtime-evidence",
                "package": {
                    "package_identity": package["package_identity"],
                    "app_sha256": package["artifacts"]["app"]["sha256"],
                    "test_sha256": package["artifacts"]["test"]["sha256"],
                },
                "matrix": {
                    "id_sha256": hashlib.sha256(
                        matrix_id.encode("utf-8")
                    ).hexdigest(),
                    "state": matrix.get("state"),
                    "outcome_summary": matrix.get("outcomeSummary"),
                    "failure_codes": failure_codes,
                    "timeline": timeline,
                },
                "device_execution_count": len(
                    matrix.get("testExecutions", [])
                ),
                "cost": {
                    "device_charge_usd": 0.0,
                    "actual_total_cost_usd": None,
                    "worst_case_reserved_usd": WORST_CASE_USD,
                },
            }
            _atomic_json(RUNTIME_REPORT, blocked)
            matrix_resource["failure_codes"] = failure_codes
            raise CloudRuntimeError("physical-device matrix did not succeed")
        local_files = _collect_results(bucket, results_dir)
        token_evidence = _runtime_tokens(local_files)
        evidence = {
            "schema": (
                "neuro-film.android-srgb-quantizer-device-runtime.v1"
            ),
            "status": "PASS",
            "claim_ceiling": (
                "Pixel-8-Android-14-device-runtime-for-exact-consumer-"
                "quantizer-only"
            ),
            "package": {
                "package_identity": package["package_identity"],
                "app_sha256": package["artifacts"]["app"]["sha256"],
                "test_sha256": package["artifacts"]["test"]["sha256"],
                "certificate_sha256": package["artifacts"]["test"][
                    "certificate_sha256"
                ],
                "core_sha256": package["native"]["core_library_sha256"],
                "jni_sha256": package["native"]["jni_library_sha256"],
            },
            "target": package["target"],
            "matrix": {
                "id_sha256": hashlib.sha256(
                    matrix_id.encode("utf-8")
                ).hexdigest(),
                "state": matrix["state"],
                "outcome_summary": matrix["outcomeSummary"],
                "timeline": timeline,
            },
            "results": {
                "file_count": len(local_files),
                "aggregate_sha256": hashlib.sha256(
                    b"".join(
                        path.relative_to(LOCAL_RESULTS)
                        .as_posix()
                        .encode("utf-8")
                        + b"\0"
                        + bytes.fromhex(_sha256(path))
                        for path in local_files
                    )
                ).hexdigest(),
                **token_evidence,
            },
            "cost": {
                "worst_case_usd": WORST_CASE_USD,
                "execution_charge_upper_bound_usd": 1.0 / 6.0,
                "billing_rate_basis": (
                    "physical device $5/hour above daily no-cost quota; "
                    "2-minute execution timeout"
                ),
            },
        }
        _atomic_json(RUNTIME_REPORT, evidence)
        resource["status"] = "evidence-collected"
        _save_ledger(ledger)
        return evidence
    except Exception as exc:
        error_name = type(exc).__name__
        resource["status"] = f"failed:{error_name}"
        _save_ledger(ledger)
        raise
    finally:
        deletion = _gcloud(
            [
                "storage",
                "rm",
                "--recursive",
                f"{bucket_uri}/**",
                "--quiet",
            ],
            timeout=180.0,
            check=False,
        )
        bucket_delete = _gcloud(
            [
                "storage",
                "buckets",
                "delete",
                bucket_uri,
                "--quiet",
            ],
            timeout=120.0,
            check=False,
        )
        cleaned = bucket_delete.returncode == 0
        resource["cleanup"] = {
            "status": "deleted" if cleaned else "delete-failed",
            "at": _now(),
            "object_delete_exit": deletion.returncode,
            "bucket_delete_exit": bucket_delete.returncode,
        }
        if matrix_id is not None and resource["children"]:
            resource["children"][0]["cleanup"]["at"] = _now()
        if cleaned and error_name is None:
            resource["status"] = "completed-cleaned"
        _save_ledger(ledger)
        if not cleaned and error_name is None:
            raise CloudRuntimeError("owned result bucket cleanup failed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Create the owned bucket and submit one physical-device matrix.",
    )
    args = parser.parse_args()
    if args.execute:
        result = execute()
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "schema": result["schema"],
                    "runtime_report_sha256": _sha256(RUNTIME_REPORT),
                },
                sort_keys=True,
            )
        )
    else:
        state = preflight()
        print(
            json.dumps(
                {
                    "status": "DRY_RUN_PASS",
                    "configuration": CONFIGURATION,
                    "resource_prefix": RESOURCE_PREFIX,
                    "testing_api_reachable": state[
                        "testing_api_reachable"
                    ],
                    "worst_case_usd": WORST_CASE_USD,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
