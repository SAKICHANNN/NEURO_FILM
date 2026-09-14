import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from src.eval.fable_protected_regions import protected_region_counts
from src.preprocess.fable_canonical_raw import linear16_to_q8


def load_query_counts(rows: list[dict], masks: list[dict], *, locked_ids: list[str]) -> dict:
    identities = [r['identity'] for r in rows]
    if (len(identities) != 4 or len(set(identities)) != 4 or identities != locked_ids
            or set(r['identity'] for r in masks) != set(identities)):
        raise ValueError('exact four locked queries and their masks required')
    counts, regions = [], []
    for row in rows:
        native = Path(row['native_path'])
        with native.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != row['native_sha256']:
                raise ValueError('query native changed')
        codes = linear16_to_q8(np.load(native, mmap_mode='r', allow_pickle=False))
        counts.append(np.stack([np.bincount(codes[..., c].ravel(), minlength=256) for c in range(3)]))
        named = {}
        for index, mask_row in enumerate(masks):
            if mask_row['identity'] != row['identity']:
                continue
            if mask_row['reviewed'] is not True or mask_row['native_sha256'] != row['native_sha256']:
                raise ValueError('mask review or native binding differs')
            for key in ['mask', 'crop']:
                if hashlib.sha256(Path(mask_row[key+'_path']).read_bytes()).hexdigest() != mask_row[key+'_sha256']:
                    raise ValueError('query mask or crop changed')
            x0, y0, x1, y1 = mask_row['box_xyxy']
            if not (0 <= x0 < x1 <= codes.shape[1] and 0 <= y0 < y1 <= codes.shape[0]):
                raise ValueError('mask crop outside native query')
            crop = codes[y0:y1, x0:x1]
            with Image.open(mask_row['crop_path']) as image:
                if not np.array_equal(np.asarray(image), crop):
                    raise ValueError('reviewed crop differs from native query')
            mask = np.load(mask_row['mask_path'], allow_pickle=False)
            if (mask.shape != crop.shape[:2] or not np.isin(mask, [0, 1]).all()
                    or int(mask.sum()) != mask_row['mask_pixels']):
                raise ValueError('mask shape, binary content or pixel count differs')
            name = f'{index}:{mask_row["category"]}'
            named.update(protected_region_counts(crop, {name: mask.astype(bool)}))
        regions.append(named)
        del codes
    return {'identities': identities, 'counts': counts, 'regions': regions}
