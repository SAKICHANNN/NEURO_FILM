import hashlib
import json
from pathlib import Path

import numpy as np
import rawpy

from src.preprocess.fable_raw_eligibility import inspect_numeric_metadata, inspect_decoder_numeric
from src.preprocess.fable_render_lock import verify_render_lock, render_locked_canonical_raw


def file_sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def render_screened_raw(path: Path, *, root: Path, seal: dict, expected_raw_sha256: str) -> tuple:
    for relative, expected in seal['project_files'].items():
        if file_sha256(root / relative) != expected:
            raise ValueError(f'screen code or render lock drift: {relative}')
    lock = json.loads((root / seal['render_lock']).read_text(encoding='utf-8'))
    verify_render_lock(root, lock)
    if file_sha256(path) != expected_raw_sha256:
        raise ValueError('RAW identity differs from expected digest')
    numeric = inspect_numeric_metadata(path)
    with rawpy.imread(str(path)) as raw:
        decoder = inspect_decoder_numeric(raw, numeric)
    linear, rendered = render_locked_canonical_raw(path, root=root, lock=lock)
    if not np.array_equal(rendered['as_shot_neutral'], numeric['as_shot_neutral']):
        raise ValueError('screened and rendered neutral differ')
    if file_sha256(path) != expected_raw_sha256:
        raise ValueError('RAW changed during screened rendering')
    return linear, {'status': 'SCREENED_RENDER_NOT_SCIENTIFIC_SOURCE_ADMISSION',
                    'raw_sha256': expected_raw_sha256, 'numeric': numeric,
                    'decoder': decoder, 'rendered': rendered,
                    'limits': 'Requires trusted caller-bound screen seal and separate exposure/role admission. Not physical calibration.'}
