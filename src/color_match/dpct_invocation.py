"""Exact-wheel local invocation adapter for the pinned D-PCT SDR package."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping
import zipfile

from .contracts import ReferenceMatchContractError
from .core_adapter import PreparedMatchViewV1, validate_prepared_match_view
from .dpct_adapter import (
    DPCT_CANONICAL_JSON,
    DPCT_MATCH_VIEW_SCHEMA,
    DPCT_PIXEL_LAYOUT,
    DPCT_PRODUCER_PROFILE_ID,
    AdaptedDpctCandidateV2,
    DpctProducerFailureV2,
    adapt_dpct_candidate_v2,
    dpct_producer_view_id_for_prepared_v2,
    verify_dpct_failed_diagnostics_v2,
)
from .dpct_invocation_profile import (
    DpctInvocationProfileV2,
    dpct_invocation_profile_payload_v2,
    load_dpct_invocation_profile_v2,
)


DPCT_INVOCATION_COMPATIBILITY_PROFILE_ID = (
    "neuro-film.dpct-invocation-consumer.v1"
)
DPCT_INVOCATION_STABLE_COMMIT = "e22725d"
DPCT_INVOCATION_SOURCE_COMMIT = (
    "01ef0616610b16246623236301234ff3b4c4a7f2"
)
DPCT_INVOCATION_WHEEL_SIZE = 95994
DPCT_INVOCATION_WHEEL_SHA256 = (
    "fd995ad88c9f30f2136508f7c3879ce6768fde7ff2e149537d2b58f78e36c292"
)
DPCT_INVOCATION_CAPABILITY_ID = "zhuise.dpct-chroma.cpu-reference.v1"
DPCT_INVOCATION_REQUEST_SCHEMA = "zhuise.invocation-request.v1"
DPCT_INVOCATION_RESPONSE_SCHEMA = "zhuise.invocation-response.v1"
DPCT_INVOCATION_CLAIM_CEILING = (
    "candidate-only-not-promoted-not-applied-not-delivered"
)
_EXPECTED_FILES_CANDIDATE = {
    "output.f32be",
    "transform.payload",
    "response.json",
}
_EXPECTED_FILES_FAILED = {"response.json"}


@dataclass(frozen=True)
class DpctInvocationOutcomeV1:
    compatibility_profile_id: str
    producer_stable_commit: str
    producer_source_commit: str
    wheel_sha256: str
    request_id: str
    response_id: str
    status: str
    claim_ceiling: str
    candidate: AdaptedDpctCandidateV2 | None
    failure: DpctProducerFailureV2 | None


@dataclass(frozen=True)
class DpctInvocationOutcomeV2:
    invocation_profile_id: str
    lower_compatibility_profile_id: str
    producer_stable_commit: str
    producer_source_commit: str
    wheel_sha256: str
    capability_id: str
    request_id: str
    response_id: str
    status: str
    claim_ceiling: str
    candidate: AdaptedDpctCandidateV2 | None
    failure: DpctProducerFailureV2 | None


@dataclass(frozen=True)
class _InvocationParameters:
    compatibility_profile_id: str
    producer_stable_commit: str
    producer_source_commit: str
    wheel_filename: str
    wheel_size_bytes: int
    wheel_sha256: str
    package_distribution: str
    package_version: str
    package_entrypoint: str
    python_major_minor: str
    numpy_version: str
    pillow_version: str
    request_schema: str
    response_schema: str
    capability_id: str
    claim_ceiling: str


_V1_PARAMETERS = _InvocationParameters(
    compatibility_profile_id=DPCT_INVOCATION_COMPATIBILITY_PROFILE_ID,
    producer_stable_commit=DPCT_INVOCATION_STABLE_COMMIT,
    producer_source_commit=DPCT_INVOCATION_SOURCE_COMMIT,
    wheel_filename="zhuise_research-0.2.0-py3-none-any.whl",
    wheel_size_bytes=DPCT_INVOCATION_WHEEL_SIZE,
    wheel_sha256=DPCT_INVOCATION_WHEEL_SHA256,
    package_distribution="zhuise-research",
    package_version="0.2.0",
    package_entrypoint="zhuise-producer-invoke",
    python_major_minor="3.12",
    numpy_version="2.4.4",
    pillow_version="12.1.1",
    request_schema=DPCT_INVOCATION_REQUEST_SCHEMA,
    response_schema=DPCT_INVOCATION_RESPONSE_SCHEMA,
    capability_id=DPCT_INVOCATION_CAPABILITY_ID,
    claim_ceiling=DPCT_INVOCATION_CLAIM_CEILING,
)


def _parameters_v2(
    profile: DpctInvocationProfileV2,
) -> tuple[DpctInvocationProfileV2, _InvocationParameters]:
    validated = load_dpct_invocation_profile_v2(
        dpct_invocation_profile_payload_v2(profile)
    )
    return validated, _InvocationParameters(
        compatibility_profile_id=(
            validated.lower_compatibility_profile_id
        ),
        producer_stable_commit=validated.producer_stable_commit,
        producer_source_commit=(
            validated.producer_package_source_commit
        ),
        wheel_filename=validated.wheel_filename,
        wheel_size_bytes=validated.wheel_size_bytes,
        wheel_sha256=validated.wheel_sha256,
        package_distribution=validated.package_distribution,
        package_version=validated.package_version,
        package_entrypoint=validated.package_entrypoint,
        python_major_minor=validated.python_major_minor,
        numpy_version=validated.numpy_version,
        pillow_version=validated.pillow_version,
        request_schema=validated.request_schema,
        response_schema=validated.response_schema,
        capability_id=validated.capability_id,
        claim_ceiling=validated.claim_ceiling,
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "D-PCT invocation JSON is not canonicalizable"
        ) from exc


def _domain_id(domain: bytes, payload: Mapping[str, Any]) -> str:
    return "sha256:" + _sha256(domain + _canonical_json(payload))


def _prepared_wire(
    prepared: PreparedMatchViewV1,
) -> tuple[dict[str, Any], bytes]:
    validate_prepared_match_view(prepared)
    height, width, channels = prepared.descriptor.shape
    if channels != 3:
        raise ReferenceMatchContractError(
            "D-PCT invocation requires three-channel pixels"
        )
    pixels = prepared.pixels.astype(">f4", copy=False).tobytes(order="C")
    view = {
        "schema": DPCT_MATCH_VIEW_SCHEMA,
        "profile_id": DPCT_PRODUCER_PROFILE_ID,
        "width": width,
        "height": height,
        "channels": 3,
        "pixel_layout": DPCT_PIXEL_LAYOUT,
        "pixel_byte_length": len(pixels),
        "pixel_sha256": "sha256:" + _sha256(pixels),
        "view_id": dpct_producer_view_id_for_prepared_v2(prepared),
    }
    return view, pixels


def _prepare_dpct_invocation_request(
    *,
    source: PreparedMatchViewV1,
    reference: PreparedMatchViewV1,
    parameters: _InvocationParameters,
) -> tuple[dict[str, Any], bytes, bytes]:
    source_view, source_pixels = _prepared_wire(source)
    reference_view, reference_pixels = _prepared_wire(reference)
    request: dict[str, Any] = {
        "schema": parameters.request_schema,
        "canonical_json": DPCT_CANONICAL_JSON,
        "operation": "fit-apply",
        "capability_id": parameters.capability_id,
        "source": {
            "view": source_view,
            "pixels_file": "source.f32be",
        },
        "reference": {
            "view": reference_view,
            "pixels_file": "reference.f32be",
        },
    }
    request["request_id"] = _domain_id(
        b"ZhuiseInvocationRequestV1\0", request
    )
    return request, source_pixels, reference_pixels


def prepare_dpct_invocation_request_v1(
    *,
    source: PreparedMatchViewV1,
    reference: PreparedMatchViewV1,
) -> tuple[dict[str, Any], bytes, bytes]:
    return _prepare_dpct_invocation_request(
        source=source,
        reference=reference,
        parameters=_V1_PARAMETERS,
    )


def prepare_dpct_invocation_request_v2(
    *,
    profile: DpctInvocationProfileV2,
    source: PreparedMatchViewV1,
    reference: PreparedMatchViewV1,
) -> tuple[dict[str, Any], bytes, bytes]:
    _, parameters = _parameters_v2(profile)
    return _prepare_dpct_invocation_request(
        source=source,
        reference=reference,
        parameters=parameters,
    )


def _strict_object(
    value: Any,
    keys: set[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ReferenceMatchContractError(
            f"{label} fields differ from the pinned invocation contract"
        )
    return value


def _load_response(path: Path) -> tuple[Mapping[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ReferenceMatchContractError(
            "D-PCT invocation response is unavailable"
        ) from exc
    if len(raw) > 1024 * 1024 or not raw.endswith(b"\n"):
        raise ReferenceMatchContractError(
            "D-PCT invocation response size/terminator is invalid"
        )
    try:
        response = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReferenceMatchContractError(
            "D-PCT invocation response is not strict UTF-8 JSON"
        ) from exc
    if raw != _canonical_json(response) + b"\n":
        raise ReferenceMatchContractError(
            "D-PCT invocation response is not canonical JSON"
        )
    return response, raw


def _validate_request(
    request: Mapping[str, Any],
    *,
    source: PreparedMatchViewV1,
    reference: PreparedMatchViewV1,
    parameters: _InvocationParameters,
) -> None:
    request = _strict_object(
        request,
        {
            "schema",
            "canonical_json",
            "operation",
            "capability_id",
            "source",
            "reference",
            "request_id",
        },
        "D-PCT invocation request",
    )
    if (
        request["schema"] != parameters.request_schema
        or request["canonical_json"] != DPCT_CANONICAL_JSON
        or request["operation"] != "fit-apply"
        or request["capability_id"] != parameters.capability_id
    ):
        raise ReferenceMatchContractError(
            "D-PCT invocation request identity mismatch"
        )
    expected_source, _ = _prepared_wire(source)
    expected_reference, _ = _prepared_wire(reference)
    for label, expected in (
        ("source", expected_source),
        ("reference", expected_reference),
    ):
        envelope = _strict_object(
            request[label],
            {"view", "pixels_file"},
            f"D-PCT invocation request {label}",
        )
        if (
            envelope["view"] != expected
            or envelope["pixels_file"] != f"{label}.f32be"
        ):
            raise ReferenceMatchContractError(
                f"D-PCT invocation request {label} binding mismatch"
            )
    identity = dict(request)
    request_id = identity.pop("request_id")
    if request_id != _domain_id(
        b"ZhuiseInvocationRequestV1\0", identity
    ):
        raise ReferenceMatchContractError(
            "D-PCT invocation request ID mismatch"
        )


def _verify_dpct_invocation_output(
    *,
    output_directory: Path,
    request: Mapping[str, Any],
    source: PreparedMatchViewV1,
    reference: PreparedMatchViewV1,
    intent_id: str,
    parameters: _InvocationParameters,
) -> tuple[str, str, AdaptedDpctCandidateV2 | None, DpctProducerFailureV2 | None]:
    _validate_request(
        request,
        source=source,
        reference=reference,
        parameters=parameters,
    )
    output_directory = output_directory.resolve()
    response, _ = _load_response(output_directory / "response.json")
    response = _strict_object(
        response,
        {
            "schema",
            "canonical_json",
            "producer_package",
            "request_id",
            "status",
            "source",
            "reference",
            "transform",
            "diagnostics",
            "apply_result",
            "artifacts",
            "failure",
            "response_id",
        },
        "D-PCT invocation response",
    )
    if (
        response["schema"] != parameters.response_schema
        or response["canonical_json"] != DPCT_CANONICAL_JSON
        or response["request_id"] != request["request_id"]
    ):
        raise ReferenceMatchContractError(
            "D-PCT invocation response/request binding mismatch"
        )
    package = _strict_object(
        response["producer_package"],
        {"distribution", "version", "entrypoint"},
        "D-PCT producer package",
    )
    if dict(package) != {
        "distribution": parameters.package_distribution,
        "version": parameters.package_version,
        "entrypoint": parameters.package_entrypoint,
    }:
        raise ReferenceMatchContractError(
            "D-PCT invocation package identity mismatch"
        )
    identity = dict(response)
    response_id = identity.pop("response_id")
    if response_id != _domain_id(
        b"ZhuiseInvocationResponseV1\0", identity
    ):
        raise ReferenceMatchContractError(
            "D-PCT invocation response ID mismatch"
        )
    if (
        response["source"] != request["source"]["view"]
        or response["reference"] != request["reference"]["view"]
    ):
        raise ReferenceMatchContractError(
            "D-PCT invocation returned foreign input views"
        )
    actual_files = {
        item.name for item in output_directory.iterdir() if item.is_file()
    }
    status = response["status"]
    candidate = None
    failure = None
    if status == "candidate":
        if actual_files != _EXPECTED_FILES_CANDIDATE:
            raise ReferenceMatchContractError(
                "D-PCT candidate artifact set mismatch"
            )
        artifacts = _strict_object(
            response["artifacts"],
            {
                "output_pixels_file",
                "output_pixel_sha256",
                "bundle_payload_file",
                "bundle_payload_sha256",
            },
            "D-PCT candidate artifacts",
        )
        if (
            artifacts["output_pixels_file"] != "output.f32be"
            or artifacts["bundle_payload_file"] != "transform.payload"
            or response["failure"] is not None
        ):
            raise ReferenceMatchContractError(
                "D-PCT candidate artifact names/failure mismatch"
            )
        output_path = output_directory / "output.f32be"
        payload_path = output_directory / "transform.payload"
        expected_output_size = (
            source.descriptor.shape[0]
            * source.descriptor.shape[1]
            * 12
        )
        if (
            output_path.stat().st_size != expected_output_size
            or payload_path.stat().st_size > 16 * 1024 * 1024
        ):
            raise ReferenceMatchContractError(
                "D-PCT invocation artifact size is invalid"
            )
        output = output_path.read_bytes()
        payload = payload_path.read_bytes()
        if (
            artifacts["output_pixel_sha256"]
            != "sha256:" + _sha256(output)
            or artifacts["bundle_payload_sha256"]
            != "sha256:" + _sha256(payload)
        ):
            raise ReferenceMatchContractError(
                "D-PCT invocation artifact hash mismatch"
            )
        candidate = adapt_dpct_candidate_v2(
            source=source,
            reference=reference,
            producer_source=response["source"],
            producer_reference=response["reference"],
            producer_transform=response["transform"],
            producer_transform_payload=payload,
            producer_diagnostics=response["diagnostics"],
            producer_apply_result=response["apply_result"],
            output_pixel_f32be=output,
            intent_id=intent_id,
        )
        if (
            candidate.aliases.capability_id
            != parameters.capability_id
        ):
            raise ReferenceMatchContractError(
                "D-PCT invocation capability is not pinned"
            )
    elif status == "failed":
        if actual_files != _EXPECTED_FILES_FAILED:
            raise ReferenceMatchContractError(
                "failed D-PCT invocation emitted candidate artifacts"
            )
        if any(
            response[key] is not None
            for key in ("transform", "apply_result", "artifacts")
        ):
            raise ReferenceMatchContractError(
                "failed D-PCT invocation carries candidate state"
            )
        failure = verify_dpct_failed_diagnostics_v2(
            response["diagnostics"]
        )
        failure_envelope = _strict_object(
            response["failure"],
            {"code", "message"},
            "D-PCT invocation failure",
        )
        if failure_envelope["code"] != failure.failure_code:
            raise ReferenceMatchContractError(
                "D-PCT failure envelope/diagnostics mismatch"
            )
    else:
        raise ReferenceMatchContractError(
            "D-PCT invocation status is unsupported"
        )
    return str(response_id), str(status), candidate, failure


def verify_dpct_invocation_output_v1(
    *,
    output_directory: Path,
    request: Mapping[str, Any],
    source: PreparedMatchViewV1,
    reference: PreparedMatchViewV1,
    intent_id: str,
) -> DpctInvocationOutcomeV1:
    response_id, status, candidate, failure = (
        _verify_dpct_invocation_output(
            output_directory=output_directory,
            request=request,
            source=source,
            reference=reference,
            intent_id=intent_id,
            parameters=_V1_PARAMETERS,
        )
    )
    return DpctInvocationOutcomeV1(
        compatibility_profile_id=(
            DPCT_INVOCATION_COMPATIBILITY_PROFILE_ID
        ),
        producer_stable_commit=DPCT_INVOCATION_STABLE_COMMIT,
        producer_source_commit=DPCT_INVOCATION_SOURCE_COMMIT,
        wheel_sha256=DPCT_INVOCATION_WHEEL_SHA256,
        request_id=str(request["request_id"]),
        response_id=response_id,
        status=status,
        claim_ceiling=DPCT_INVOCATION_CLAIM_CEILING,
        candidate=candidate,
        failure=failure,
    )


def verify_dpct_invocation_output_v2(
    *,
    profile: DpctInvocationProfileV2,
    output_directory: Path,
    request: Mapping[str, Any],
    source: PreparedMatchViewV1,
    reference: PreparedMatchViewV1,
    intent_id: str,
) -> DpctInvocationOutcomeV2:
    validated, parameters = _parameters_v2(profile)
    response_id, status, candidate, failure = (
        _verify_dpct_invocation_output(
            output_directory=output_directory,
            request=request,
            source=source,
            reference=reference,
            intent_id=intent_id,
            parameters=parameters,
        )
    )
    return DpctInvocationOutcomeV2(
        invocation_profile_id=validated.profile_id,
        lower_compatibility_profile_id=(
            validated.lower_compatibility_profile_id
        ),
        producer_stable_commit=validated.producer_stable_commit,
        producer_source_commit=(
            validated.producer_package_source_commit
        ),
        wheel_sha256=validated.wheel_sha256,
        capability_id=validated.capability_id,
        request_id=str(request["request_id"]),
        response_id=response_id,
        status=status,
        claim_ceiling=validated.claim_ceiling,
        candidate=candidate,
        failure=failure,
    )


def _verify_runtime(
    python_executable: Path,
    parameters: _InvocationParameters,
) -> None:
    code = (
        "import json,sys,numpy,PIL;"
        "print(json.dumps({'python':f'{sys.version_info.major}."
        "{sys.version_info.minor}','numpy':numpy.__version__,"
        "'pillow':PIL.__version__},sort_keys=True))"
    )
    try:
        completed = subprocess.run(
            [python_executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ReferenceMatchContractError(
            "D-PCT invocation runtime preflight failed closed"
        ) from exc
    try:
        identity = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ReferenceMatchContractError(
            "D-PCT invocation runtime identity is invalid"
        ) from exc
    if identity != {
        "python": parameters.python_major_minor,
        "numpy": parameters.numpy_version,
        "pillow": parameters.pillow_version,
    }:
        raise ReferenceMatchContractError(
            "D-PCT invocation runtime identity mismatch"
        )


def _invoke_dpct_package(
    *,
    source: PreparedMatchViewV1,
    reference: PreparedMatchViewV1,
    intent_id: str,
    wheel_path: Path,
    python_executable: Path,
    scratch_directory: Path,
    timeout_seconds: float,
    parameters: _InvocationParameters,
    profile: DpctInvocationProfileV2 | None,
) -> DpctInvocationOutcomeV1 | DpctInvocationOutcomeV2:
    wheel_path = wheel_path.resolve()
    python_executable = python_executable.resolve()
    scratch_directory = scratch_directory.resolve()
    if (
        not wheel_path.is_file()
        or wheel_path.name
        != parameters.wheel_filename
        or wheel_path.stat().st_size != parameters.wheel_size_bytes
        or _sha256(wheel_path.read_bytes())
        != parameters.wheel_sha256
    ):
        raise ReferenceMatchContractError(
            "D-PCT invocation wheel identity mismatch"
        )
    if not python_executable.is_file():
        raise ReferenceMatchContractError(
            "D-PCT invocation Python executable is unavailable"
        )
    if not scratch_directory.is_dir():
        raise ReferenceMatchContractError(
            "D-PCT invocation scratch directory is unavailable"
        )
    _verify_runtime(python_executable, parameters)
    request, source_pixels, reference_pixels = (
        _prepare_dpct_invocation_request(
            source=source,
            reference=reference,
            parameters=parameters,
        )
    )
    with tempfile.TemporaryDirectory(
        prefix="nfcm-dpct-invocation-",
        dir=scratch_directory,
    ) as temporary:
        root = Path(temporary)
        package_directory = root / "package"
        request_directory = root / "request"
        output_directory = root / "output"
        package_directory.mkdir()
        with zipfile.ZipFile(wheel_path) as archive:
            members = archive.infolist()
            for member in members:
                member_path = Path(member.filename)
                if (
                    member_path.is_absolute()
                    or ".." in member_path.parts
                    or member.filename.endswith("\\")
                ):
                    raise ReferenceMatchContractError(
                        "D-PCT wheel contains an unsafe archive path"
                    )
            archive.extractall(package_directory)
        request_directory.mkdir()
        (request_directory / "source.f32be").write_bytes(source_pixels)
        (request_directory / "reference.f32be").write_bytes(
            reference_pixels
        )
        (request_directory / "request.json").write_bytes(
            _canonical_json(request) + b"\n"
        )
        code = (
            "import importlib,os,sys;"
            "package=os.path.normcase(os.path.abspath(sys.argv.pop(1)));"
            "sys.path.insert(0,package);"
            "module=importlib.import_module('zhuise.producer_invocation');"
            "origin=os.path.normcase(os.path.abspath(module.__file__));"
            "assert origin.startswith(package + os.sep),origin;"
            "module.main()"
        )
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        try:
            completed = subprocess.run(
                [
                    python_executable,
                    "-c",
                    code,
                    package_directory,
                    "--request",
                    request_directory / "request.json",
                    "--output-directory",
                    output_directory,
                ],
                cwd=root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ReferenceMatchContractError(
                "D-PCT exact-wheel invocation process failed closed"
            ) from exc
        if completed.returncode != 0:
            raise ReferenceMatchContractError(
                "D-PCT exact-wheel invocation failed closed: "
                + completed.stderr.strip()[:512]
            )
        if profile is None:
            return verify_dpct_invocation_output_v1(
                output_directory=output_directory,
                request=request,
                source=source,
                reference=reference,
                intent_id=intent_id,
            )
        return verify_dpct_invocation_output_v2(
            profile=profile,
            output_directory=output_directory,
            request=request,
            source=source,
            reference=reference,
            intent_id=intent_id,
        )


def invoke_dpct_package_v1(
    *,
    source: PreparedMatchViewV1,
    reference: PreparedMatchViewV1,
    intent_id: str,
    wheel_path: Path,
    python_executable: Path,
    scratch_directory: Path,
    timeout_seconds: float = 300.0,
) -> DpctInvocationOutcomeV1:
    outcome = _invoke_dpct_package(
        source=source,
        reference=reference,
        intent_id=intent_id,
        wheel_path=wheel_path,
        python_executable=python_executable,
        scratch_directory=scratch_directory,
        timeout_seconds=timeout_seconds,
        parameters=_V1_PARAMETERS,
        profile=None,
    )
    if not isinstance(outcome, DpctInvocationOutcomeV1):
        raise AssertionError("v1 invocation returned a v2 outcome")
    return outcome


def invoke_dpct_package_v2(
    *,
    profile: DpctInvocationProfileV2,
    source: PreparedMatchViewV1,
    reference: PreparedMatchViewV1,
    intent_id: str,
    wheel_path: Path,
    python_executable: Path,
    scratch_directory: Path,
    timeout_seconds: float = 300.0,
) -> DpctInvocationOutcomeV2:
    validated, parameters = _parameters_v2(profile)
    outcome = _invoke_dpct_package(
        source=source,
        reference=reference,
        intent_id=intent_id,
        wheel_path=wheel_path,
        python_executable=python_executable,
        scratch_directory=scratch_directory,
        timeout_seconds=timeout_seconds,
        parameters=parameters,
        profile=validated,
    )
    if not isinstance(outcome, DpctInvocationOutcomeV2):
        raise AssertionError("v2 invocation returned a v1 outcome")
    return outcome


__all__ = [
    "DPCT_INVOCATION_CLAIM_CEILING",
    "DPCT_INVOCATION_COMPATIBILITY_PROFILE_ID",
    "DPCT_INVOCATION_SOURCE_COMMIT",
    "DPCT_INVOCATION_STABLE_COMMIT",
    "DPCT_INVOCATION_WHEEL_SHA256",
    "DpctInvocationOutcomeV1",
    "DpctInvocationOutcomeV2",
    "invoke_dpct_package_v1",
    "invoke_dpct_package_v2",
    "prepare_dpct_invocation_request_v1",
    "prepare_dpct_invocation_request_v2",
    "verify_dpct_invocation_output_v1",
    "verify_dpct_invocation_output_v2",
]
