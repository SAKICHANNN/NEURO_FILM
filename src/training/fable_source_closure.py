import ast
import hashlib
from pathlib import Path


def local_import_closure(root: Path, entries: list[str]) -> dict[str, str]:
    pending = [Path(p) for p in entries]
    found = {}
    while pending:
        relative = pending.pop()
        key = relative.as_posix()
        if key in found:
            continue
        blob = (root / relative).read_bytes()
        found[key] = hashlib.sha256(blob).hexdigest()
        for parent in relative.parents:
            initializer = parent / '__init__.py'
            if (root / initializer).is_file() and initializer != relative:
                pending.append(initializer)
        tree = ast.parse(blob, filename=key)
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    package = list(relative.parent.parts)
                    package = package[:len(package)-node.level+1]
                    module = '.'.join(package + ([node.module] if node.module else []))
                else:
                    module = node.module or ''
                names = [module] + [module+'.'+alias.name for alias in node.names]
            for name in names:
                if name.split('.')[0] not in ['src', 'scripts']:
                    continue
                path = Path(*name.split('.'))
                if (root / path.with_suffix('.py')).is_file():
                    pending.append(path.with_suffix('.py'))
                elif (root / path / '__init__.py').is_file():
                    pending.append(path / '__init__.py')
    return dict(sorted(found.items()))
