import hashlib
import importlib.util
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "scripts/run_salut_photo_development_v1.py"
    expected = "9446a923cb149d299a650d900c987a3ab788419c8f89938ef66be4ced5240dff"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
    spec = importlib.util.spec_from_file_location("salut_frozen_v1", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CONFIG = root / "configs/salut_reference02_response_v1.json"
    module.__file__ = str(Path(__file__).resolve())
    original_preflight = module.preflight

    def reference_preflight():
        config, checks = original_preflight()
        assert config["parent_runner_sha256"] == expected
        assert config["parent_runner"] == "scripts/run_salut_photo_development_v1.py"
        old_reference = checks["reference"].copy()
        replacement = config["reference_override"]
        manifest = module.read_json(root / old_reference["manifest"])
        assert replacement["manifest_index"] == 0
        entry = manifest["rows"][0]
        assert entry["path"] == "02.jpg"
        path = (root / old_reference["manifest"]).parent / entry["path"]
        assert path == root / replacement["path"]
        assert path.stat().st_size == replacement["bytes"] == entry["bytes"]
        assert module.digest(path) == entry["sha256"] == replacement["sha256"]
        report_path = root / config["reference18_report"]
        assert module.digest(report_path) == config["reference18_report_sha256"]
        prior = module.read_json(report_path)
        assert prior["reference_sha256"] == old_reference["sha256"]
        assert prior["status"] == "COMPLETE_NOT_PROMOTED"
        loading = prior["model_loading"]
        assert (
            loading["checkpoint_sha256"]
            == "5ad3685cf6bbf14230d71580b9b93f03e76bf6d7dfaceecd0a3527341cc92f2b"
        )
        assert loading["source_sha256"] == module.PINS
        assert loading["input_shape"] == [1, 3, 512, 512]
        assert loading["context_target"] == [512, 512]
        assert (loading["seed"], loading["query_chunk"], loading["threads"]) == (
            48,
            512,
            2,
        )
        for row, queued in zip(prior["rows"], checks["queue"], strict=True):
            assert row["source_index"] == queued["source_index"]
            assert row["source_sha256"] == queued["source_sha256"]
            for arm in ("salut", "g0"):
                assert (
                    module.digest(report_path.parent / row["arms"][arm]["path"])
                    == row["arms"][arm]["sha256"]
                )
        checks["reference18_report_sha256"] = config["reference18_report_sha256"]
        checks["baseline_reference18"] = old_reference
        checks["parent_runner_sha256"] = expected
        checks["reference"] = {
            **old_reference,
            "index": 0,
            "path": replacement["path"],
            "sha256": replacement["sha256"],
            "appearance_intent": "Manifest-first reference02: sunny cyan sky, brighter yellow-green vegetation, warm stone; compare directional response to18 without tuning",
        }
        checks["reference_change"] = (
            "18 to02 only; old_ncc remains legacy reference18 control"
        )
        return config, checks

    module.preflight = reference_preflight
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
