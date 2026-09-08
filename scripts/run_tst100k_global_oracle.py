import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import tifffile
from scipy import sparse
from scipy.optimize import lsq_linear
from skimage.color import deltaE_ciede2000, rgb2lab
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/tst100k_global_oracle_v1.json"
G1 = ROOT / "scripts/run_salut_g1_projection.py"
G1_SHA = "b4bfc66741fcba40701cd79e99997e0bf21ee3c344e46fe5895c1811dd233e94"
SUPERVISOR_SHA = "9446a923cb149d299a650d900c987a3ab788419c8f89938ef66be4ced5240dff"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def g1_functions():
    assert digest(G1) == G1_SHA
    wanted = {"design", "curvature", "fit_lut"}
    nodes = [n for n in ast.parse(G1.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in nodes} == wanted
    namespace = {"np": np, "sparse": sparse, "lsq_linear": lsq_linear}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(G1), "exec"), namespace)  # noqa: S102 - three reviewed functions from SHA-bound local source
    return namespace


G1_FUNCTIONS = g1_functions()
design = G1_FUNCTIONS["design"]
curvature = G1_FUNCTIONS["curvature"]


def fit_lut(rgb, target, dimension, weight):
    return G1_FUNCTIONS["fit_lut"](rgb, target, {
        "lut_dimension": dimension,
        "smoothness_weight": weight * (dimension - 1) ** 4,
        "bounds": [0.0, 1.0],
    })


def block_split(height, width, config, index, fold):
    y, x = np.indices((height, width))
    b, margin = config["block_size"], config["block_border_exclusion"]
    valid = ((x % b >= margin) & (x % b < b - margin)
             & (y % b >= margin) & (y % b < b - margin))
    parity = (x // b + y // b) % 2
    available = np.flatnonzero(valid & (parity == fold))
    check = np.flatnonzero(valid & (parity != fold))
    assert len(available) > 0 and len(check) > 0
    rng = np.random.default_rng(config["seed"] + 100 * index + fold)
    fit = np.sort(rng.choice(available, min(len(available), config["max_fit_pixels"]), replace=False))
    assert not np.intersect1d(fit, check).size
    return fit, check


def schedule(config):
    return [(i, d, fold, weight)
            for weight in config["smoothness_weights"]
            for fold in config["folds"]
            for i in config["source_indices"]
            for d in config["lut_dimensions"]]


def preflight():
    config = json.loads(CONFIG.read_text())
    assert config["source_indices"] == [2, 4, 6, 7, 8]
    assert config["lut_dimensions"] == [17, 33]
    assert config["folds"] == [0, 1] and config["threads"] == 2
    assert config["solver"] == {"tol": 1e-6, "lsmr_tol": 1e-7, "max_iter": 50, "lsmr_maxiter": 300}
    parent_path = ROOT / config["parent_report"]
    assert digest(parent_path) == config["parent_sha256"]
    assert digest(ROOT / config["science_result"]) == config["science_sha256"]
    parent = json.loads(parent_path.read_text())
    queue = []
    for index in config["source_indices"]:
        row = next(r for r in parent["triplets"] if r["index"] == index)
        item = {"index": index, "arms": {}, "color_assumption": index in (6, 8)}
        for role in ("content", "gt"):
            arm = row["arms"][role]
            path = parent_path.parent / arm["srgb16_path"]
            assert digest(path) == arm["srgb16_sha256"]
            with tifffile.TiffFile(path) as handle:
                page = handle.pages[0]
                assert page.dtype == np.uint16 and page.shape[-1] == 3
                shape = list(page.shape)
            item["arms"][role] = {"path": str(path), "sha256": digest(path), "shape": shape}
        assert item["arms"]["content"]["shape"] == item["arms"]["gt"]["shape"]
        queue.append(item)
    checks = {"config_sha256": digest(CONFIG), "script_sha256": digest(Path(__file__)),
              "g1_sha256": G1_SHA, "queue": queue, "planned_fits": len(schedule(config)),
              "status": "WAIT_ALIGNMENT_PIN_NO_REAL_FITS"}
    if config["alignment_sha256"]:
        alignment_path = ROOT / config["alignment_report"]
        assert digest(alignment_path) == config["alignment_sha256"]
        alignment = json.loads(alignment_path.read_text())
        for item in queue:
            entry = next(r for r in alignment["rows"] if r["index"] == item["index"])
            assert entry["manual_admissibility"] == "ADMITTED_CONDITIONAL_NATIVE_CORRESPONDENCE"
            assert entry["source_sha256"] == item["arms"]["content"]["sha256"]
            assert entry["target_sha256"] == item["arms"]["gt"]["sha256"]
        checks["alignment_sha256"] = config["alignment_sha256"]
        checks["status"] = "READY_FOR_ROOT_REVIEW"
    return config, checks


def statistics(prediction, target, source, indices):
    errors, identity_errors = [], []
    absolute, square, newly_boundary, target_new_boundary = 0.0, 0.0, 0, 0
    for start in range(0, len(indices), 32768):
        idx = indices[start:start + 32768]
        p, y, x = prediction[idx], target[idx], source[idx]
        residual = p - y
        absolute += np.abs(residual).sum()
        square += np.square(residual).sum()
        ylab = rgb2lab(y)
        errors.append(deltaE_ciede2000(rgb2lab(p), ylab))
        identity_errors.append(deltaE_ciede2000(rgb2lab(x), ylab))
        newly_boundary += np.any(((p <= 0) | (p >= 1)) & (x > 0) & (x < 1), axis=1).sum()
        target_new_boundary += np.any(((p <= 0) | (p >= 1)) & (y > 0) & (y < 1), axis=1).sum()
    de, identity_de = np.concatenate(errors), np.concatenate(identity_errors)
    return {"pixels": len(indices), "rgb_mae_255": float(absolute / (3 * len(indices)) * 255),
            "rgb_rmse_255": float(np.sqrt(square / (3 * len(indices))) * 255),
            "delta_e00_mean": float(de.mean()), "delta_e00_p95": float(np.quantile(de, .95)),
            "identity_delta_e00_mean": float(identity_de.mean()),
            "q_mean_de_ratio_floor1": float(de.mean() / max(identity_de.mean(), 1.0)),
            "new_boundary_relative_source_fraction": float(newly_boundary / len(indices)),
            "new_boundary_relative_target_fraction": float(target_new_boundary / len(indices))}


def support(rgb, fit, check, dimension):
    cells = np.minimum((rgb * (dimension - 1)).astype(np.int32), dimension - 2)
    key = cells[:, 0] * dimension**2 + cells[:, 1] * dimension + cells[:, 2]
    return float((~np.isin(key[check], np.unique(key[fit]))).mean())


def worker(attempt):
    config, checks = preflight()
    assert checks == json.loads((attempt / "lock.json").read_text())
    assert checks["status"] == "READY_FOR_ROOT_REVIEW"
    report = {"status": "RUNNING", "rows": [], "new_forwards": 0, "checks": checks,
              "scope": config["scope"], "complete_fit_count": 0, "affine_rows": [],
              "claim": "All fixed settings retained; no test-based winner or training admission."}
    save(attempt / "report.json", report)
    threadpool_limits(limits=2)
    for index, dimension, fold, weight in schedule(config):
        item = next(r for r in checks["queue"] if r["index"] == index)
        source = tifffile.imread(item["arms"]["content"]["path"]).astype(np.float64) / 65535
        target = tifffile.imread(item["arms"]["gt"]["path"]).astype(np.float64) / 65535
        h, w = source.shape[:2]
        x, y = source.reshape(-1, 3), target.reshape(-1, 3)
        fit, check = block_split(h, w, config, index, fold)
        if not any(r["index"] == index and r["fold"] == fold for r in report["affine_rows"]):
            matrix = np.column_stack([x[fit], np.ones(len(fit))])
            coefficients, _, rank, singular = np.linalg.lstsq(matrix, y[fit], rcond=None)
            affine = x @ coefficients[:3] + coefficients[3]
            affine_row = {"index": index, "fold": fold, "rank": int(rank),
                          "singular_values": singular.tolist(), "coefficients": coefficients.tolist(),
                          "raw_check_rgb_mae_255": float(np.abs(affine[check] - y[check]).mean() * 255),
                          "raw_out_of_range_fraction": float(np.any((affine < 0) | (affine > 1), axis=1).mean())}
            affine = np.clip(affine, 0, 1)
            affine_row["clipped_fit"] = statistics(affine, y, x, fit)
            affine_row["clipped_check"] = statistics(affine, y, x, check)
            affine_path = attempt / f"{index:02d}_fold{fold}_affine_srgb16.tif"
            with tifffile.TiffFile(item["arms"]["content"]["path"]) as original:
                profile = original.pages[0].tags[34675].value
            tifffile.imwrite(affine_path, np.rint(affine.reshape(h, w, 3) * 65535).astype(np.uint16),
                             photometric="rgb", extratags=[(34675, "B", len(profile), profile, False)])
            affine_row["artifact"] = {"path": affine_path.name, "sha256": digest(affine_path)}
            report["affine_rows"].append(affine_row)
            save(attempt / "report.json", report)
            del affine
        row = {"index": index, "dimension": dimension, "fold": fold, "lambda": weight,
               "status": "FITTING", "color_assumption": item["color_assumption"]}
        report["rows"].append(row)
        save(attempt / "report.json", report)
        prefix = f"{index:02d}_d{dimension}_fold{fold}_lambda{weight:g}"
        coordinates = attempt / f"{index:02d}_fold{fold}_coordinates.npz"
        if not coordinates.exists():
            np.savez(coordinates, fit=fit, check=check)
        row["coordinates"] = {"path": coordinates.name, "sha256": digest(coordinates)}
        row["uncovered_check_cell_fraction"] = support(x, fit, check, dimension)
        save(attempt / "report.json", report)
        lut, states = fit_lut(x[fit], y[fit], dimension, weight)
        row["solver_states"] = states
        assert np.isfinite(lut).all()
        row["status"] = "SOLVED" if all(s["success"] for s in states) else "NONCONVERGED_INCONCLUSIVE"
        save(attempt / "report.json", report)
        prediction = np.empty_like(x)
        for start in range(0, len(x), 32768):
            prediction[start:start + 32768] = design(x[start:start + 32768], dimension) @ lut
        row["fit"] = statistics(prediction, y, x, fit)
        row["check"] = statistics(prediction, y, x, check)
        lut_path, image_path = attempt / (prefix + "_lut.npy"), attempt / (prefix + "_srgb16.tif")
        np.save(lut_path, lut)
        with tifffile.TiffFile(item["arms"]["content"]["path"]) as original:
            icc = original.pages[0].tags[34675].value
        tifffile.imwrite(image_path, np.rint(np.clip(prediction.reshape(h, w, 3), 0, 1) * 65535).astype(np.uint16),
                         photometric="rgb", extratags=[(34675, "B", len(icc), icc, False)])
        row["artifacts"] = [{"path": p.name, "sha256": digest(p)} for p in (lut_path, image_path)]
        report["complete_fit_count"] += 1
        save(attempt / "report.json", report)
    report["status"] = ("COMPLETE_NOT_PROMOTED" if all(r["status"] == "SOLVED" for r in report["rows"])
                        else "NONCONVERGED_INCONCLUSIVE")
    save(attempt / "report.json", report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return 0
    config, checks = preflight()
    if not args.run:
        print(json.dumps(checks, indent=2))
        return 0
    assert checks["status"] == "READY_FOR_ROOT_REVIEW", "Alignment evidence must be pinned first"
    source = ROOT / "scripts/run_salut_photo_development_v1.py"
    assert digest(source) == SUPERVISOR_SHA
    spec = importlib.util.spec_from_file_location("salut_supervisor", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.__file__, module.CONFIG = __file__, CONFIG
    return module.launch(config, checks)


if __name__ == "__main__":
    raise SystemExit(main())
