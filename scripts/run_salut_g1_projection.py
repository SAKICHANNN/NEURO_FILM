import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "2"

import numpy as np
from PIL import Image
from scipy import sparse
from scipy.optimize import lsq_linear

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/salut_g1_projection_v1.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def design(rgb, dimension):
    position = np.clip(rgb, 0, 1) * (dimension - 1)
    lower = np.minimum(position.astype(np.int64), dimension - 2)
    fraction = position - lower
    columns, weights = [], []
    for r in (0, 1):
        for g in (0, 1):
            for b in (0, 1):
                columns.append(
                    (lower[:, 0] + r) * dimension**2
                    + (lower[:, 1] + g) * dimension
                    + lower[:, 2]
                    + b
                )
                weights.append(
                    (fraction[:, 0] if r else 1 - fraction[:, 0])
                    * (fraction[:, 1] if g else 1 - fraction[:, 1])
                    * (fraction[:, 2] if b else 1 - fraction[:, 2])
                )
    return sparse.csr_matrix(
        (
            np.stack(weights, axis=1).ravel(),
            (np.repeat(np.arange(len(rgb)), 8), np.stack(columns, axis=1).ravel()),
        ),
        shape=(len(rgb), dimension**3),
    )


def curvature(dimension):
    one = sparse.diags(
        [np.ones(dimension - 2), -2 * np.ones(dimension - 2), np.ones(dimension - 2)],
        [0, 1, 2],
        shape=(dimension - 2, dimension),
        format="csr",
    )
    identity = sparse.eye(dimension, format="csr")
    return sparse.vstack(
        [
            sparse.kron(sparse.kron(one, identity), identity),
            sparse.kron(sparse.kron(identity, one), identity),
            sparse.kron(sparse.kron(identity, identity), one),
        ],
        format="csr",
    )


def split(height, width, config):
    y, x = np.indices((height, width))
    block, border = config["block_size"], config["block_border_exclusion"]
    valid = (
        (x % block >= border)
        & (x % block < block - border)
        & (y % block >= border)
        & (y % block < block - border)
    )
    parity = (x // block + y // block) % 2
    fit_all = np.flatnonzero(valid & (parity == config["fit_parity"]))
    fit = fit_all[
        np.linspace(
            0,
            len(fit_all) - 1,
            min(len(fit_all), config["max_fit_pixels"]),
            dtype=np.int64,
        )
    ]
    test = np.flatnonzero(valid & (parity != config["fit_parity"]))
    assert len(np.intersect1d(fit, test)) == 0
    return fit, test


def fit_lut(rgb, teacher, config):
    d = config["lut_dimension"]
    smooth = curvature(d)
    matrix = sparse.vstack(
        [
            design(rgb, d) / np.sqrt(len(rgb)),
            smooth * np.sqrt(config["smoothness_weight"] / smooth.shape[0]),
        ],
        format="csr",
    )
    values, states = [], []
    for channel in range(3):
        target = np.concatenate(
            [teacher[:, channel] / np.sqrt(len(rgb)), np.zeros(smooth.shape[0])]
        )
        solution = lsq_linear(
            matrix,
            target,
            bounds=tuple(config["bounds"]),
            tol=1e-6,
            lsmr_tol=1e-7,
            max_iter=50,
            lsmr_maxiter=300,
        )
        values.append(solution.x)
        states.append(
            {
                "status": int(solution.status),
                "success": bool(solution.success)
                and (
                    solution.status != 3
                    or solution.unbounded_sol[1] in (0, 1, 2, 4, 5)
                    or solution.optimality <= 1e-6
                ),
                "unbounded_lsmr": [float(v) for v in solution.unbounded_sol[1:]],
                "message": solution.message,
                "optimality": float(solution.optimality),
                "iterations": solution.nit,
                "cost": float(solution.cost),
            }
        )
    return np.stack(values, axis=1), states


def preflight():
    config = json.loads(CONFIG.read_text())
    queue = []
    for parent in config["parents"]:
        path = ROOT / parent["report"]
        assert digest(path) == parent["sha256"]
        report = json.loads(path.read_text())
        for row in report["rows"]:
            entries = {}
            for arm in ("original", "salut", "g0"):
                entry = row["arms"][arm]
                file = path.parent / entry["path"]
                assert digest(file) == entry["sha256"]
                entries[arm] = {"path": str(file), "sha256": entry["sha256"]}
            queue.append({"source_index": row["source_index"], "inputs": entries})
    assert [r["source_index"] for r in queue] == config["source_indices"]
    return config, {
        "config_sha256": digest(CONFIG),
        "script_sha256": digest(Path(__file__)),
        "queue": queue,
    }


def worker(attempt):
    config, checks = preflight()
    assert checks == json.loads((attempt / "lock.json").read_text())
    report = {
        "status": "running",
        "new_forwards": 0,
        "rows": [],
        "checks": checks,
        "solver": {"tol": 1e-6, "lsmr_tol": 1e-7, "max_iter": 50, "lsmr_maxiter": 300},
    }
    for item in checks["queue"]:
        row = {"source_index": item["source_index"], "status": "fitting"}
        report["rows"].append(row)
        save(attempt / "report.json", report)
        original = np.asarray(
            Image.open(item["inputs"]["original"]["path"]).convert("RGB")
        )
        teacher = np.asarray(Image.open(item["inputs"]["salut"]["path"]).convert("RGB"))
        height, width = original.shape[:2]
        fit, test = split(height, width, config)
        x, target = original.reshape(-1, 3) / 255.0, teacher.reshape(-1, 3) / 255.0
        lut, states = fit_lut(x[fit], target[fit], config)
        row["solver_states"] = states
        row["fit_pixels"], row["test_pixels"] = len(fit), len(test)
        row["coordinate_intersection"] = 0
        prefix = str(item["source_index"]).zfill(2)
        np.savez(attempt / (prefix + "_coordinates.npz"), fit=fit, test=test)
        np.save(attempt / (prefix + "_lut.npy"), lut)
        prediction = np.empty_like(x)
        for start in range(0, len(x), 65536):
            prediction[start : start + 65536] = (
                design(x[start : start + 65536], config["lut_dimension"]) @ lut
            )
        for label, indices in (("fit", fit), ("test", test)):
            residual = prediction[indices] - target[indices]
            row[label + "_rgb_mae_255"] = float(np.abs(residual).mean() * 255)
            row[label + "_rgb_rmse_255"] = float(
                np.sqrt(np.square(residual).mean()) * 255
            )
        cells = np.minimum(
            (x * (config["lut_dimension"] - 1)).astype(int), config["lut_dimension"] - 2
        )
        keys = (
            cells[:, 0] * config["lut_dimension"] ** 2
            + cells[:, 1] * config["lut_dimension"]
            + cells[:, 2]
        )
        row["test_uncovered_cell_fraction"] = float(
            (~np.isin(keys[test], np.unique(keys[fit]))).mean()
        )
        result = np.rint(
            np.clip(prediction.reshape(height, width, 3), 0, 1) * 255
        ).astype(np.uint8)
        Image.fromarray(result).save(attempt / (prefix + "_g1.png"))
        s = np.float32(config["display_strength"])
        for label, image in (
            ("g1", result),
            ("salut", teacher),
            ("g0", np.asarray(Image.open(item["inputs"]["g0"]["path"]).convert("RGB"))),
        ):
            blended = np.rint(
                (1 - s) * original.astype(np.float32) + s * image.astype(np.float32)
            ).astype(np.uint8)
            Image.fromarray(blended).save(
                attempt / (prefix + "_" + label + "_strength065.png")
            )
        row["artifacts"] = [
            {"path": f.name, "sha256": digest(f)}
            for f in sorted(attempt.glob(prefix + "_*"))
        ]
        row["status"] = (
            "complete" if all(s["success"] for s in states) else "nonconverged"
        )
        save(attempt / "report.json", report)
    report["status"] = (
        "COMPLETE_NOT_PROMOTED"
        if all(r["status"] == "complete" for r in report["rows"])
        else "NONCONVERGED_NOT_PROMOTED"
    )
    save(attempt / "report.json", report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    config, checks = preflight()
    if not args.run:
        print(json.dumps(checks, indent=2))
        return
    source = ROOT / "scripts/run_salut_photo_development_v1.py"
    assert (
        digest(source)
        == "9446a923cb149d299a650d900c987a3ab788419c8f89938ef66be4ced5240dff"
    )
    spec = importlib.util.spec_from_file_location("salut_supervisor", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.__file__, module.CONFIG = __file__, CONFIG
    config["output"] = "outputs/salut_g1_projection_v1"
    config["budget"] = {
        "maximum_rss_bytes": config["tree_rss_limit_bytes"],
        "maximum_worker_seconds": config["wall_seconds_total"],
        "poll_seconds": 0.1,
    }
    raise SystemExit(module.launch(config, checks))


if __name__ == "__main__":
    main()
