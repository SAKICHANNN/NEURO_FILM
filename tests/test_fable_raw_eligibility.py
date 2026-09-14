from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from src.preprocess.fable_raw_eligibility import tag_numbers


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
