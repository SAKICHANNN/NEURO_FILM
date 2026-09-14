import numpy as np

from src.eval.fable_cdfe_case_metrics import paired_case_metrics
from src.eval.fable_photometry_metrics import image_histograms
from src.eval.fable_reference_photometry import transform, quantize8


def test_histogram_cases_match_direct_native_pixel_errors():
    rng = np.random.default_rng(19)
    codes = rng.integers(0, 256, (13, 17, 3), dtype=np.uint8)
    source = codes.astype(np.float64)/255
    ideal = np.array([[.7, .4, -.3, .8], [-.5, -.7, .6, -.4]])
    learned = ideal*.65
    bounds = dict(slope_limit=1.25, offset_limit=.35)
    result = paired_case_metrics({'learned': learned, 'oracle': ideal, 'identity': ideal*0},
                                ideal, [image_histograms(codes)['counts']], **bounds)
    for j in range(2):
        target = np.floor(255*quantize8(transform(source, ideal[j], **bounds))+.5)
        output = np.floor(255*quantize8(transform(source, learned[j], **bounds))+.5)
        np.testing.assert_allclose(result['D'][j, 0], np.mean((target-codes)**2), rtol=0, atol=1e-12)
        np.testing.assert_allclose(result['methods']['learned']['E'][j, 0],
                                   np.mean((output-target)**2), rtol=0, atol=1e-12)
        assert result['methods']['oracle']['E'][j, 0] == 0
        assert result['methods']['identity']['P'][j, 0] == 0
