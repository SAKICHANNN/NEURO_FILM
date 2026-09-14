import numpy as np
import tifffile


def reconcile_neutrals(records) -> list:
    if not records:
        raise ValueError('explicit AsShotNeutral required')
    values = [np.asarray(record['values'], dtype=np.float64) for record in records]
    if any(v.shape != (3,) or not np.isfinite(v).all() or np.any(v <= 0) for v in values):
        raise ValueError('positive finite RGB neutral required')
    if any(not np.array_equal(v, values[0]) for v in values[1:]):
        raise ValueError('conflicting AsShotNeutral values across IFDs')
    return values[0].tolist()


def matrix_singular_values(matrix, shape) -> list:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != shape or not np.isfinite(matrix).all():
        raise ValueError('invalid color matrix shape or values')
    singular = np.linalg.svd(matrix, compute_uv=False)
    tolerance = max(shape) * np.finfo(np.float64).eps * singular[0]
    if np.count_nonzero(singular > tolerance) != 3:
        raise ValueError('color matrix numerical rank must be three')
    return singular.tolist()


def inspect_decoder_numeric(raw, metadata) -> dict:
    if raw.color_desc != b'RGBG':
        raise ValueError('RGBG decoder channel order required')
    xyz = np.asarray(raw.rgb_xyz_matrix, dtype=np.float64)
    matrix_singular_values(xyz, (4, 3))
    singular = matrix_singular_values(xyz[:3], (3, 3))
    black = np.asarray(raw.black_level_per_channel, dtype=np.float64)
    if black.shape != (4,) or not np.isfinite(black).all() or np.any(black < 0) or np.any(black >= metadata['white']):
        raise ValueError('decoder black outside supported range')
    return {'rgb_xyz_matrix': xyz.tolist(), 'rgb_singular_values': singular,
            'decoder_black': black.tolist(), 'rank_rule': 'float64 eps * max(shape) * largest singular value',
            'limits': 'Numerical validity only; does not establish matrix source or physical accuracy.'}


def tag_numbers(tag, tiff=None) -> np.ndarray:
    if int(tag.dtype) in (5, 10) and tiff is not None:
        handle = tiff.filehandle
        position = handle.tell()
        try:
            handle.seek(tag.valueoffset)
            encoded = handle.read(tag.count * 8)
        finally:
            handle.seek(position)
        if len(encoded) != tag.count * 8:
            raise ValueError(f'truncated rational payload: {tag.name}')
        dtype = tiff.byteorder + ('u4' if int(tag.dtype) == 5 else 'i4')
        values = np.frombuffer(encoded, dtype=dtype).astype(np.float64)
    else:
        values = np.asarray(tag.value, dtype=np.float64).reshape(-1)
    if int(tag.dtype) in (5, 10):
        if len(values) != 2 * tag.count or np.any(values[1::2] == 0):
            raise ValueError(f'invalid rational encoding: {tag.name}')
        values = values[::2] / values[1::2]
    if len(values) != tag.count or not np.isfinite(values).all():
        raise ValueError(f'invalid numeric values: {tag.name}')
    return values


def inspect_numeric_metadata(path) -> dict:
    with tifffile.TiffFile(path) as tiff:
        queue = list(tiff.pages)
        planes, matrices, neutrals = [], [], []
        while queue:
            page = queue.pop(0)
            if 'AsShotNeutral' in page.tags:
                neutrals.append({'ifd_offset': page.offset, 'values': tag_numbers(page.tags['AsShotNeutral'], tiff).tolist()})
            if int(page.photometric) == 32803:
                planes.append(page)
            for name in ('ColorMatrix1', 'ColorMatrix2'):
                if name in page.tags:
                    values = tag_numbers(page.tags[name], tiff)
                    if values.size != 9:
                        raise ValueError('finite full-rank 3x3 DNG color matrix required')
                    matrix_singular_values(values.reshape(3, 3), (3, 3))
                    matrices.append({'tag': name, 'values': values.tolist()})
            if page.pages is not None:
                queue.extend(page.pages)
        if len(planes) != 1 or not matrices:
            raise ValueError('one CFA plane and explicit color matrix required')
        neutral = reconcile_neutrals(neutrals)
        tags = planes[0].tags
        required = ('BlackLevel', 'WhiteLevel', 'ActiveArea', 'BlackLevelRepeatDim', 'SamplesPerPixel')
        if any(name not in tags for name in required):
            raise ValueError('explicit black/white/area/repeat/sample metadata required')
        black, white, area, repeat, samples = [tag_numbers(tags[name], tiff) for name in required]
        if white.size != 1 or white[0] != int(white[0]) or not 0 < white[0] <= 65535:
            raise ValueError('integer white level in uint16 range required')
        if area.size != 4 or np.any(area != np.floor(area)) or np.any(area < 0):
            raise ValueError('integer nonnegative active area required')
        height, width = area[2:] - area[:2]
        if min(height, width) <= 0 or samples.tolist() != [1.]:
            raise ValueError('positive active area and single CFA sample required')
        if repeat.size != 2 or np.any(repeat < 1) or np.any(repeat != np.floor(repeat)) or black.size != int(np.prod(repeat)):
            raise ValueError('black repeat dimensions disagree with black values')
        deltas, arrays = {}, {}
        for name, count in [('BlackLevelDeltaH', width), ('BlackLevelDeltaV', height)]:
            values = tag_numbers(tags[name], tiff) if name in tags else np.zeros(1)
            if name in tags and values.size != count:
                raise ValueError('black delta dimension disagrees with active area')
            deltas[name] = {'present': name in tags, 'count': values.size, 'minimum': float(values.min()), 'maximum': float(values.max())}
            arrays[name] = values if name in tags else np.zeros(int(count))
        bounds = []
        rh, rw = map(int, repeat)
        for row in range(min(rh, int(height))):
            vertical = arrays['BlackLevelDeltaV'][row::rh]
            for col in range(min(rw, int(width))):
                horizontal = arrays['BlackLevelDeltaH'][col::rw]
                value = black[row * rw + col]
                bounds.append((value + vertical.min() + horizontal.min(), value + vertical.max() + horizontal.max()))
        lower = float(min(b[0] for b in bounds))
        upper = float(max(b[1] for b in bounds))
        if lower < 0 or upper >= white[0]:
            raise ValueError('black plus delta bounds outside supported [0, white)')
        return {'status': 'NUMERIC_METADATA_SCREEN_ONLY', 'black_values': black.tolist(),
                'black_repeat': repeat.tolist(), 'black_plus_delta_bounds': [lower, upper],
                'deltas': deltas, 'white': float(white[0]), 'color_matrices': matrices,
                'as_shot_neutral': neutral, 'neutral_sources': neutrals,
                'limits': ['Numeric screening does not prove physical calibration or decoder matrix selection.',
                           'Black range is a protocol support restriction; no source is admitted by this check alone.',
                           'Decoder numeric screening is separate; source independence and rendering contract remain required.']}
