from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from src.preprocess.fable_raw_eligibility import tag_numbers, reconcile_neutrals, matrix_singular_values, inspect_decoder_numeric


def test_conflicting_neutral_is_not_last_ifd_wins():
    with pytest.raises(ValueError, match='conflicting'):
        reconcile_neutrals([{'values': [1, 1, 1]}, {'values': [.5, 1, 1]}])
    assert reconcile_neutrals([{'values': [1, 1, 1]}] * 2) == [1, 1, 1]


def test_decoder_negative_coefficients_and_zero_fourth_row_allowed():
    raw = SimpleNamespace(color_desc=b'RGBG', rgb_xyz_matrix=[[1, -.2, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], black_level_per_channel=[0, 1, 2, 3])
    assert len(inspect_decoder_numeric(raw, {'white': 100})['rgb_singular_values']) == 3
    raw.black_level_per_channel = [0, 1, 2, 100]
    with pytest.raises(ValueError, match='black'):
        inspect_decoder_numeric(raw, {'white': 100})


@pytest.mark.parametrize('matrix', [np.zeros((3, 3)), np.full((3, 3), np.nan), np.ones((4, 3))])
def test_degenerate_color_matrix_rejected(matrix):
    with pytest.raises(ValueError):
        matrix_singular_values(matrix, (3, 3))


def test_signed_and_unsigned_rational_decoding():
    for dtype in (5, 10):
        tag = SimpleNamespace(dtype=dtype, count=2, value=(0, 256, -1, 4), name='test')
        np.testing.assert_array_equal(tag_numbers(tag), [0., -.25])


@pytest.mark.parametrize('values,count', [((1, 0), 1), ((1, 2, 3), 2), ((float('nan'), 1), 1)])
def test_invalid_rational_rejected(values, count):
    with pytest.raises(ValueError):
        tag_numbers(SimpleNamespace(dtype=10, count=count, value=values, name='test'))


@pytest.mark.parametrize('byteorder', ['<', '>'])
@pytest.mark.parametrize('count', [1, 1024, 1025, 2348])
def test_rational_file_payload_crosses_numpy_reader_threshold(tmp_path, byteorder, count):
    path = tmp_path / 'rational.tif'
    numerators = np.arange(count, dtype=np.int32) - count // 2
    values = np.column_stack([numerators, np.full(count, 256)]).ravel().tolist()
    tifffile.imwrite(path, np.zeros((2, 2), dtype=np.uint16), byteorder=byteorder,
                     extratags=[(50716, '2i', count, values, False)])
    with tifffile.TiffFile(path) as tiff:
        position = tiff.filehandle.tell()
        decoded = tag_numbers(tiff.pages[0].tags[50716], tiff)
        assert tiff.filehandle.tell() == position
    np.testing.assert_array_equal(decoded, numerators / 256)
