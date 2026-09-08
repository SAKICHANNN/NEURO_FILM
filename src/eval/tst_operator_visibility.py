from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab

from src.eval.tst_correspondence_operator import render_absolute

CHUNK_PIXELS = 32768
VISIBILITY_THRESHOLD = 4.0


@dataclass(frozen=True)
class VisibilityProbe:
    probe_id: str
    split: str
    source: np.ndarray


def _validate_source(source):
    if not isinstance(source, np.ndarray) or source.dtype != np.uint16:
        raise ValueError('Source must be canonical encoded-sRGB uint16')
    if source.ndim != 3 or source.shape[-1] != 3 or min(source.shape[:2]) < 1:
        raise ValueError('Source must have nonempty H,W,3 shape')
    if not source.flags.c_contiguous:
        raise ValueError('Source must be C-contiguous for bounded native-order chunks')


def _validate_grid(grid):
    if np.iscomplexobj(grid):
        raise ValueError('Grid must be real')
    values = np.asarray(grid, dtype=np.float64)
    if values.shape != (343, 3) or not np.isfinite(values).all():
        raise ValueError('Grid must be finite absolute7cube values with shape343,3')
    return values


def classify_mean(mean_delta_e2000: float) -> bool:
    value = float(mean_delta_e2000)
    if not np.isfinite(value) or value < 0:
        raise ValueError('Mean DeltaE2000 must be finite and nonnegative')
    return value >= VISIBILITY_THRESHOLD


def mean_native_delta_e(source: np.ndarray, absolute_grid: np.ndarray) -> dict:
    _validate_source(source)
    grid = _validate_grid(absolute_grid)
    flat = source.reshape(-1, 3)
    total, chunks = 0.0, 0
    for start in range(0, len(flat), CHUNK_PIXELS):
        rgb = flat[start:start+CHUNK_PIXELS].astype(np.float64)/65535.0
        prediction = render_absolute(rgb, grid, 7)
        if not np.isfinite(prediction).all():
            raise ValueError('Nonfinite rendered prediction')
        prediction = np.clip(prediction, 0.0, 1.0)
        delta = deltaE_ciede2000(rgb2lab(rgb), rgb2lab(prediction))
        if not np.isfinite(delta).all():
            raise ValueError('Nonfinite DeltaE2000')
        total += float(np.sum(delta, dtype=np.float64))
        chunks += 1
    mean = total/len(flat)
    classify_mean(mean)
    return {'mean_delta_e2000': mean, 'sum_delta_e2000': total,
            'pixel_count': len(flat), 'chunks': chunks, 'chunk_pixels': CHUNK_PIXELS,
            'arithmetic': 'float64 encoded-sRGB; clip B(X)V before Lab; native C order; ordered sums of32768pixel chunks; no display quantization'}


def screen_visibility(probes: Sequence[VisibilityProbe], absolute_grid: np.ndarray,
                      *, candidate_split: str) -> dict:
    if candidate_split not in ('fit', 'monitor', 'constructed_test'):
        raise ValueError('Unsupported candidate partition')
    if len(probes) != 4 or len({p.probe_id for p in probes}) != 4:
        raise ValueError('Exactly four distinct frozen probe ids required')
    if any(p.split != candidate_split for p in probes):
        raise ValueError('All four probes must belong to candidate partition')
    grid = _validate_grid(absolute_grid)
    for probe in probes:
        _validate_source(probe.source)
    rows, below, visible = [], 0, 0
    for probe in probes:
        if below >= 2:
            rows.append({'probe_id': probe.probe_id, 'split': probe.split,
                         'status': 'NOT_COMPUTED_AFTER_NECESSARY_FAILURE',
                         'mean_delta_e2000': None, 'visible': None})
            continue
        metric = mean_native_delta_e(probe.source, grid)
        passed = classify_mean(metric['mean_delta_e2000'])
        visible += int(passed)
        below += int(not passed)
        rows.append({'probe_id': probe.probe_id, 'split': probe.split,
                     'status': 'COMPUTED', **metric, 'visible': passed})
    possible = below < 2
    return {'status': 'VISIBILITY_POSSIBLE_AWAITING_AGENT_REVIEW' if possible else 'PROVISIONAL_OPERATOR_NOT_VISIBLE_PAIR_UNREVIEWED',
            'candidate_split': candidate_split, 'strength': 1.0,
            'required_visible_probes': 3, 'total_probes': 4,
            'threshold_mean_delta_e2000': VISIBILITY_THRESHOLD,
            'computed_visible_count': visible, 'computed_below_threshold_count': below,
            'uncomputed_count': sum(r['status'] != 'COMPUTED' for r in rows),
            'visibility_possible': possible, 'probes': rows,
            'claim': 'Necessary numeric screen only; pair correctness, comfort and content preservation remain unreviewed.'}
