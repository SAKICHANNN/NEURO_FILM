import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import rawpy
import tifffile

from src.preprocess.fable_canonical_raw import render_canonical_raw


def verify_render_lock(root: Path, lock: dict) -> dict:
    actual = {'python': sys.version, 'rawpy': rawpy.__version__,
              'libraw': list(rawpy.libraw_version), 'flags': rawpy.flags,
              'numpy': np.__version__, 'tifffile': tifffile.__version__}
    if actual != lock['runtime']:
        raise ValueError('render runtime version or flags drift')
    package = Path(rawpy.__file__).parent
    for row in lock['runtime_files']:
        path = package / row['name']
        if path.name != row['name'] or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('render runtime binary drift')
    for relative, expected in lock['project_files'].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f'render code or config drift: {relative}')
    return json.loads((root / lock['config']).read_text(encoding='utf8'))


def render_locked_canonical_raw(path: Path, *, root: Path, lock: dict) -> tuple[np.ndarray, dict]:
    config = verify_render_lock(root, lock)
    return render_canonical_raw(path, config)
