import numpy as np

from src.eval.fable_cdfe_assessment_gates import assessment_gates


def test_good_photometry_cannot_hide_failed_controls():
    identity = np.repeat((np.arange(256)/255)[:, None], 3, axis=1)
    tables = np.broadcast_to(identity, (1024, 256, 3))
    methods = {name: {'E': np.ones((1024, 4)), 'P': np.full((1024, 4), 25.), 'tables': tables}
               for name in ['learned', 'constant', 'shuffled']}
    counts = np.ones((3, 256), dtype=np.int64)
    result = assessment_gates({'D': np.full((1024, 4), 100.), 'methods': methods,
        'oracle_tables': tables}, donor_ids=[str(j) for j in range(32)],
        query_ids=['q0', 'q1', 'q2', 'q3'], donor_cameras=['camera']*32,
        treatment_regions=[str(j//8) for j in range(32)], query_counts=[counts]*4,
        query_regions=[{'region': counts}]*4)
    assert result['photometry']['photometry_passed']
    assert not result['controls']['passed']
    assert not result['numeric_passed']
    assert result['visual_review_required']
