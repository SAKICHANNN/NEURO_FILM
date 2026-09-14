import numpy as np

from src.eval.fable_cdfe_assessment import evaluate_assessment


def test_assessment_composition_preserves_donor_treatment_query_axes(monkeypatch):
    import src.eval.fable_cdfe_assessment as module

    ids = [str(j) for j in range(32)]
    assignment = {'represented_assessment': ids[:16], 'held_cameras': ['held'],
                  'held_assessment': {'held': ids[16:]}, 'queries': ['q0', 'q1', 'q2', 'q3']}
    rng = np.random.default_rng(7)
    ideal = rng.uniform(.25, .9, (32, 4))
    for j in range(32):
        ideal[j, 1] *= -1 if j//8 in [1, 3] else 1
        ideal[j, 2] *= -1 if j//8 in [2, 3] else 1
    canonical = rng.uniform(-.1, .1, (32, 4))
    a = np.exp(np.log(1.25)*ideal[:, 0])
    mu = a[None, :, None]*canonical[:, None, :3]+.35*ideal[None, :, 1:]
    scale = a[None, :]*np.exp(canonical[:, None, 3])
    prediction = np.repeat(canonical[:, None, :], 32, axis=1)

    def fake_predictions(model, rows, treatments, **kwargs):
        assert kwargs['locked_ids'] == ids
        return {'predicted_canonical': prediction, 'canonical_targets': prediction.copy(),
                'after_mu': mu, 'after_scale': scale}

    counts = []
    for j in range(4):
        c = np.zeros((3, 256), dtype=np.int64)
        c[:, 30+j*40:60+j*40] = 1
        counts.append(c)

    monkeypatch.setattr(module, 'assessment_predictions', fake_predictions)
    monkeypatch.setattr(module, 'load_query_counts', lambda *a, **k:
                        {'counts': counts, 'regions': [{'r': c} for c in counts]})
    plan = {'rows': [{'camera': 'camera'} for _ in ids],
        'treatments': [{'id': str(j), 'u': u} for j, u in enumerate(ideal)],
        'treatment_ids': ids, 'treatment_regions': [str(j//8) for j in range(32)],
        'shuffle_mapping': {str(j): str((j+1)%32) for j in range(32)}, 'queries': [], 'masks': []}
    result = evaluate_assessment(None, plan, assignment=assignment,
        normalizer={'mean': np.zeros(4)}, native_config={}, progress=lambda _: None)
    np.testing.assert_allclose(result['arrays']['learned_u'], np.tile(ideal, (32, 1, 1)), atol=1e-14)
    assert result['arrays']['learned_E'].shape == (1024, 4)
    np.testing.assert_array_equal(result['arrays']['learned_E'], np.zeros((1024, 4)))
    np.testing.assert_array_equal(result['arrays']['paired_oracle_E'], np.zeros((1024, 4)))
    d = result['arrays']['D'].reshape(32, 32, 4)
    np.testing.assert_array_equal(d[0], d[31])
    assert not np.array_equal(d[:, :, 0], d[:, :, 3])
