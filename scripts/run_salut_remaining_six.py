import ast
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
    module.CONFIG = root / "configs/salut_photo_development_remaining_six_v1.json"
    module.__file__ = str(Path(__file__).resolve())
    config = module.read_json(module.CONFIG)
    assert config["parent_runner_sha256"] == expected
    assert config["parent_runner"] == str(source.relative_to(root)).replace("\\", "/")
    node = next(
        n
        for n in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(n, ast.FunctionDef) and n.name == "preflight"
    )
    old_queue = ast.parse('assert config["source_indices"] == [0, 1, 2]').body[0]
    new_queue = ast.parse('assert config["source_indices"] == [3, 4, 5, 6, 7, 8]').body[
        0
    ]
    old_budget = ast.parse(
        'assert config["budget"] == {"maximum_forwards": 3, "maximum_worker_seconds": 600, "maximum_rss_bytes": 4294967296, "poll_seconds": 0.1}'
    ).body[0]
    new_budget = ast.parse(
        'assert config["budget"] == {"maximum_forwards": 6, "maximum_worker_seconds": 600, "maximum_rss_bytes": 4294967296, "poll_seconds": 0.1}'
    ).body[0]
    replacements = {ast.dump(old_queue): new_queue, ast.dump(old_budget): new_budget}
    replaced = []
    for index, statement in enumerate(node.body):
        identity = ast.dump(statement)
        if identity in replacements:
            node.body[index] = ast.copy_location(replacements[identity], statement)
            replaced.append(identity)
    assert len(set(replaced)) == len(replaced) == 2
    transformed = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    ast_sha = hashlib.sha256(ast.dump(transformed).encode()).hexdigest()
    exec(compile(transformed, str(source), "exec"), module.__dict__)  # noqa: S102 - two exact assertion changes in hash-pinned source.
    adapted_preflight = module.preflight

    def checked_preflight():
        config, checks = adapted_preflight()
        checks["parent_runner_sha256"] = expected
        checks["adapted_preflight_ast_sha256"] = ast_sha
        checks["adaptations"] = ["queue 0..2 to 3..8", "maximum forwards 3 to 6"]
        assert len(checks["queue"]) == 6
        return config, checks

    module.preflight = checked_preflight
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
