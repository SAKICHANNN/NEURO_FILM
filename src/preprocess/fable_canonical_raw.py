from pathlib import Path

import numpy as np
import rawpy
import tifffile


def dng_render_metadata(path: Path) -> dict:
    with tifffile.TiffFile(path) as tiff:
        queue = list(tiff.pages)
        raw_pages, neutral = [], None
        while queue:
            page = queue.pop(0)
            if 'AsShotNeutral' in page.tags:
                tag = page.tags['AsShotNeutral']
                values = np.asarray(tag.value, dtype=np.float64)
                neutral = values.reshape(-1, 2)[:, 0] / values.reshape(-1, 2)[:, 1] if tag.dtype == 5 else values
            if int(page.photometric) == 32803:
                raw_pages.append(page)
            if page.pages is not None:
                queue.extend(page.pages)
        if len(raw_pages) != 1 or neutral is None or neutral.shape != (3,) or not np.isfinite(neutral).all() or np.any(neutral <= 0):
            raise ValueError('one CFA DNG plane and explicit positive RGB AsShotNeutral required')
        page = raw_pages[0]
        if 'WhiteLevel' not in page.tags or 'ActiveArea' not in page.tags or 'BlackLevel' not in page.tags:
            raise ValueError('explicit DNG white/black levels and active area required')
        white = np.asarray(page.tags['WhiteLevel'].value).reshape(-1)
        if len(white) != 1 or not 0 < white[0] <= 65535:
            raise ValueError('single positive integer DNG white level required')
        gains = 1 / neutral
        gains /= gains.min()
        return {'as_shot_neutral': neutral.tolist(), 'user_wb': [*gains, gains[1]],
                'white_level': int(white[0]), 'active_area': list(page.tags['ActiveArea'].value),
                'black_level_tag': np.asarray(page.tags['BlackLevel'].value).tolist(),
                'black_delta_h_present': 'BlackLevelDeltaH' in page.tags,
                'black_delta_v_present': 'BlackLevelDeltaV' in page.tags}


def linear16_to_q8(linear: np.ndarray) -> np.ndarray:
    if linear.dtype != np.uint16 or linear.ndim != 3 or linear.shape[-1] != 3:
        raise ValueError('uint16 HWC linear RGB required')
    x = linear.astype(np.float64) / 65535
    encoded = np.where(x <= .0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - .055)
    return np.floor(np.clip(encoded, 0, 1) * 255 + .5).astype(np.uint8)


def render_canonical_raw(path: Path, config: dict, *, diagnostic_raw_scale: float = 1.) -> tuple[np.ndarray, dict]:
    if diagnostic_raw_scale not in (1., .5):
        raise ValueError('only identity or declared consumed zero-black half-signal diagnostic supported')
    metadata = dng_render_metadata(path)
    with rawpy.imread(str(path)) as raw:
        pattern = raw.raw_pattern
        if (pattern is None or pattern.shape != (2, 2) or raw.color_desc != b'RGBG'
                or sorted(raw.color_desc[i] for i in pattern.ravel()) != sorted(b'RGGB')
                or raw.sizes.pixel_aspect != 1 or raw.sizes.flip not in (0, 3, 5, 6)):
            raise ValueError('ordinary RGB Bayer, square pixels and supported orientation required')
        s = raw.sizes
        area = [s.top_margin, s.left_margin, s.top_margin + s.height, s.left_margin + s.width]
        if area != metadata['active_area'] or raw.white_level != metadata['white_level']:
            raise ValueError('decoder active area or white level differs from DNG metadata')
        wb = np.asarray(raw.camera_whitebalance[:3])
        if not np.isfinite(wb).all() or np.any(wb <= 0) or not np.allclose(wb / wb.min(), metadata['user_wb'][:3], rtol=1e-5):
            raise ValueError('decoder WB does not agree with explicit AsShotNeutral')
        metadata.update(black_level_per_channel=raw.black_level_per_channel, raw_pattern=pattern.tolist(),
                        flip=s.flip, rgb_xyz_matrix=raw.rgb_xyz_matrix.tolist(),
                        rawpy_version=rawpy.__version__, libraw_version=list(rawpy.libraw_version),
                        diagnostic_raw_scale=diagnostic_raw_scale)
        if diagnostic_raw_scale != 1:
            if any(raw.black_level_per_channel) or metadata['black_delta_h_present'] or metadata['black_delta_v_present']:
                raise ValueError('half-signal diagnostic requires zero black and no delta fields')
            raw.raw_image[:] = np.floor(raw.raw_image.astype(np.float64) * diagnostic_raw_scale).astype(np.uint16)
        params = dict(config['params'])
        params['demosaic_algorithm'] = rawpy.DemosaicAlgorithm[params['demosaic_algorithm']]
        params['output_color'] = rawpy.ColorSpace[params['output_color']]
        params['highlight_mode'] = rawpy.HighlightMode[params['highlight_mode']]
        params['fbdd_noise_reduction'] = rawpy.FBDDNoiseReductionMode[params['fbdd_noise_reduction']]
        params['gamma'] = tuple(params['gamma'])
        params['user_wb'] = metadata['user_wb']
        params['user_flip'] = s.flip
        linear = raw.postprocess(**params)
        expected = (s.width, s.height, 3) if s.flip in (5, 6) else (s.height, s.width, 3)
        if linear.shape != expected or linear.dtype != np.uint16:
            raise ValueError('renderer changed active-area dimensions or output type')
    metadata['shape'] = list(linear.shape)
    return linear, metadata
