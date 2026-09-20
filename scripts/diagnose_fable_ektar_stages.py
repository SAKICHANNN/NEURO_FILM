import ast
import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/fable_ektar_stage_diagnostic_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(name, body):
    (OUT / name).write_text(json.dumps(body, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "claim.json").open("x") as handle:
        json.dump(dict(pid=os.getpid(), script_sha256=sha(__file__)), handle)
    started = time.perf_counter()
    report = dict(status="RUNNING", scope="one synthetic neutral stage attribution; no candidate quality claim", photo_reads=0, photo_renders=0, parameter_searches=0)
    try:
        config_path = ROOT / "configs/fable_ektar_external_control_v1.json"
        config = json.loads(config_path.read_text())
        previous = ROOT / config["output"]
        adoption = json.loads((previous / "adoption.json").read_text())
        prior_report = json.loads((previous / "report.json").read_text())
        assert sha(config_path) == prior_report["config_sha256"]
        assert Path(sys.executable).resolve() == (ROOT / config["runtime_python"]).resolve()
        assert sha(ROOT / config["configure_source"]["path"]) == config["configure_source"]["sha256"]
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
            os.environ[key] = str(config["threads"])
        os.environ["NUMBA_CACHE_DIR"] = str(OUT / "numba_cache")
        os.environ["MPLCONFIGDIR"] = str(OUT / "matplotlib")
        sys.dont_write_bytecode = True
        external = (ROOT / config["external_root"]).resolve()
        sys.path.insert(0, str(external / "src"))
        import numpy as np
        import colour
        import spektrafilm
        from spektrafilm import init_params, digest_params
        from spektrafilm.runtime.pipeline import SimulationPipeline
        import spektrafilm.runtime.stages.scanning as scanning
        import spektrafilm.model.develop as development
        import spektrafilm.model.couplers as couplers

        assert Path(spektrafilm.__file__).resolve().is_relative_to(external)
        tree = ast.parse((ROOT / config["configure_source"]["path"]).read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "configure")
        namespace = dict(init_params=init_params, digest_params=digest_params)
        exec(compile(ast.Module(body=[function], type_ignores=[]), "pinned_configure", "exec"), namespace)
        params = namespace["configure"](config["profile"], config["policy"])
        encoded_params = json.dumps(dataclasses.asdict(params), sort_keys=True, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value.item() if isinstance(value, np.generic) else str(value), allow_nan=True)
        parameter_sha = hashlib.sha256(encoded_params.encode()).hexdigest()
        assert parameter_sha == adoption["parameter_identity"]["complete_parameter_sha256"]
        report["runtime"] = dict(pid=os.getpid(), executable=sys.executable, external_module=spektrafilm.__file__, threads=config["threads"], gpu_used=False)
        report["bindings"] = dict(config_sha256=sha(config_path), run3_report_sha256=sha(previous / "report.json"), adoption_sha256=sha(previous / "adoption.json"), parameter_sha256=parameter_sha, script_sha256=sha(__file__))
        report["protocol"] = dict(gray_codes=256, neighbour_delta_encoded=1e-6, bins=[0, .05, .1, .2, .4, 1.000001], reference_interval=[.2, .6], parameter_changes="only input_cctf_decoding=False for exact linear-representation equivalence", endpoint_rule="actual interpolation input <= first or >= last curve abscissa, per channel", derivative_rule="central fixed-neighbour finite differences; each stage slope normalized only to its own middle-range median absolute slope; no cross-unit magnitude comparison")
        write_json("adoption.json", report)
        print(json.dumps(dict(stage="starting_synthetic_only", pid=os.getpid(), parameter_sha256=parameter_sha)), flush=True)

        codes = np.arange(256, dtype=float) / 255
        samples = np.stack([codes, np.maximum(codes - 1e-6, 0), np.minimum(codes + 1e-6, 1)])
        toy = np.repeat(samples[..., None], 3, axis=-1)
        captures = {}
        interpolation = []
        context = {"run": "encoded", "stage": "initialization"}
        old_compress = scanning.compress_rgb
        old_interp = development.interpolate_exposure_to_density

        def capture_compress(rgb, *args, **kwargs):
            captures[context["run"] + "_scan_linear_pre_gamut"] = rgb.copy()
            result = old_compress(rgb, *args, **kwargs)
            captures[context["run"] + "_scan_linear_post_gamut"] = result.copy()
            return result

        def capture_interp(log_raw, curves, exposure, gamma_factor):
            result = old_interp(log_raw, curves, exposure, gamma_factor)
            gamma = np.broadcast_to(np.asarray(gamma_factor), (3,))
            low, high = np.asarray(exposure)[0] / gamma, np.asarray(exposure)[-1] / gamma
            name = f"interp_{len(interpolation):02d}"
            captures[name + "_log_exposure"] = np.asarray(log_raw).copy()
            captures[name + "_density"] = result.copy()
            interpolation.append(dict(name=name, run=context["run"], stage=context["stage"], shape=list(result.shape), low=low.tolist(), high=high.tolist(), below_or_equal_count=np.sum(log_raw <= low, axis=(0, 1)).tolist(), above_or_equal_count=np.sum(log_raw >= high, axis=(0, 1)).tolist(), base_row_below_codes=[np.flatnonzero(log_raw[0, :, channel] <= low[channel]).tolist() for channel in range(3)], base_row_above_codes=[np.flatnonzero(log_raw[0, :, channel] >= high[channel]).tolist() for channel in range(3)]))
            return result

        scanning.compress_rgb = capture_compress
        development.interpolate_exposure_to_density = capture_interp
        couplers.interpolate_exposure_to_density = capture_interp
        states = {}
        for name, values in (("encoded", toy), ("linear", colour.models.eotf_sRGB(toy))):
            context.update(run=name, stage="initialization")
            current = copy.deepcopy(params)
            if name == "linear":
                current.io.input_cctf_decoding = False
            pipeline = SimulationPipeline(current)
            state = {"rgb_in": values}
            for node in pipeline._topology:
                context["stage"] = node.label
                node.fire(state)
                print(json.dumps(dict(run=name, stage=node.label)), flush=True)
            states[name] = state
            for key, value in state.items():
                captures[name + "_" + key] = np.asarray(value)
        for name in ("encoded", "linear"):
            captures[name + "_scan_encoded_pre_gamut"] = colour.models.eotf_inverse_sRGB(captures[name + "_scan_linear_pre_gamut"])
        report["equivalence"] = {key: float(np.max(np.abs(states["encoded"][key] - states["linear"][key]))) for key in states["encoded"] if key not in ("rgb_in", "rgb_pre")}
        report["equivalence"]["all_max_abs_below_1e-10"] = all(value < 1e-10 for value in report["equivalence"].values())
        report["interpolation"] = interpolation
        keys = ["log_e_film", "cmy_film", "log_e_print", "cmy_print", "scan_linear_pre_gamut", "scan_linear_post_gamut", "scan_encoded_pre_gamut", "rgb_out"]
        report["stages"] = {}
        bins = report["protocol"]["bins"]
        for key in keys:
            value = captures["encoded_" + key]
            derivative = (value[2] - value[1]) / (samples[2] - samples[1])[:, None]
            captures[key + "_derivative_wrt_encoded_input"] = derivative
            reference = np.median(np.abs(derivative[(codes >= .2) & (codes < .6)]), axis=0)
            ratios = np.abs(derivative) / np.maximum(reference, 1e-30)
            rows = []
            for lower, upper in zip(bins[:-1], bins[1:]):
                selection = (codes >= lower) & (codes < upper)
                rows.append(dict(input_encoded_interval=[lower, upper], count=int(selection.sum()), mean_abs_slope=np.mean(np.abs(derivative[selection]), axis=0).tolist(), mean_relative_to_own_mid_slope=np.mean(ratios[selection], axis=0).tolist(), exactly_zero_derivative_count=np.sum(derivative[selection] == 0, axis=0).tolist()))
            report["stages"][key] = dict(minimum=value.min(axis=(0, 1)).tolist(), maximum=value.max(axis=(0, 1)).tolist(), own_mid_slope=reference.tolist(), bins=rows)
        pre = captures["encoded_scan_encoded_pre_gamut"]
        post = states["encoded"]["rgb_out"]
        dpre = (pre[2] - pre[1]) / (samples[2] - samples[1])[:, None]
        dpost = (post[2] - post[1]) / (samples[2] - samples[1])[:, None]
        report["gamut_same_domain_comparison"] = [dict(input_encoded_interval=[lo, hi], post_to_pre_mean_abs_slope=(np.mean(np.abs(dpost[(codes >= lo) & (codes < hi)]), axis=0) / np.maximum(np.mean(np.abs(dpre[(codes >= lo) & (codes < hi)]), axis=0), 1e-30)).tolist(), max_abs_encoded_change=np.max(np.abs(post[:, (codes >= lo) & (codes < hi)] - pre[:, (codes >= lo) & (codes < hi)]), axis=(0, 1)).tolist()) for lo, hi in zip(bins[:-1], bins[1:])]
        captures["input_encoded_samples"] = samples
        np.savez(OUT / "synthetic_stages.npz", **captures)
        report["arrays"] = dict(path=(OUT / "synthetic_stages.npz").relative_to(ROOT).as_posix(), sha256=sha(OUT / "synthetic_stages.npz"), count=len(captures), all_finite=all(bool(np.isfinite(value).all()) for value in captures.values()))
        assert report["arrays"]["all_finite"]
        assert report["equivalence"]["all_max_abs_below_1e-10"]
        report["status"] = "COMPLETE_SYNTHETIC_ATTRIBUTION_ONLY"
        report["limitations"] = ["neutral rays do not establish colored night-pixel attribution", "sRGB decoding does not invert unknown photographic tone processing", "no treatment change or candidate acceptance follows from this diagnostic"]
    except Exception:
        report["status"] = "FAILED"
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        write_json("report.json", report)
        print(json.dumps(dict(status=report["status"], seconds=report["seconds"], report=str(OUT / "report.json"))), flush=True)


if __name__ == "__main__":
    main()
