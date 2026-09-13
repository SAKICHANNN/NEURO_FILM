import numpy as np
import pytest

from scripts.assess_fable_photometry import (
    improvement_summary, mean_render_tables, primary_decision, wrong_reference_indices,
)
from src.eval.fable_reference_photometry import quantize8


def test_wrong_reference_uses_cell_donor_repeat_and_antithetic_draw_after_shuffle():
    rows = [{'cell': cell, 'donor_id': donor, 'repeat': repeat, 'draw_id': draw,
             'pair_id': draw // 2, 'target': [(-1.) ** draw * (draw // 2 + 1), 0, 0, 0]}
            for cell, repeat in [('both_held', 0), ('both_held_repeat', 1)]
            for donor in (6, 7) for draw in range(4)]
    np.random.default_rng(9).shuffle(rows)
    indices = wrong_reference_indices(rows)
    for i, other in enumerate(indices):
        for field in ('cell', 'donor_id', 'repeat', 'pair_id'):
            assert rows[i][field] == rows[other][field]
        assert rows[i]['draw_id'] ^ 1 == rows[other]['draw_id']
        assert indices[other] == i
    with pytest.raises(ValueError):
        wrong_reference_indices(rows[:-1])


def test_mean_render_quantizes_each_training_target_before_averaging_codes():
    tables = np.array([.49, .51])[:, None, None] / 255
    floating, delivered = mean_render_tables(tables)
    np.testing.assert_array_equal(floating, tables.mean(0))
    np.testing.assert_array_equal(delivered, quantize8(quantize8(tables).mean(0)))
    tables = np.array([.49, .49, 1.49])[:, None, None] / 255
    _, delivered = mean_render_tables(tables)
    assert delivered.item() == 0
    assert quantize8(tables.mean(0)).item() == 1 / 255


def test_pair_aggregation_uses_16_blocks_and_is_order_invariant():
    records = [{'pair_id': pair, 'donor_id': donor, 'repeat': repeat, 'query_id': query}
               for pair in range(16) for donor in (6, 7) for repeat in (0, 1) for query in range(4)]
    values = np.asarray([1. if r['pair_id'] < 12 else -1. for r in records])
    result = improvement_summary(records, values, .05)
    assert result['nontied'] == 16 and result['pvalue'] == 2517 / 65536
    permutation = np.random.default_rng(98).permutation(len(records))
    shuffled = improvement_summary([records[i] for i in permutation], values[permutation], .05)
    np.testing.assert_array_equal(result['block_improvements'], shuffled['block_improvements'])
    assert shuffled['passed']
    donor_failure = np.asarray([3. if r['donor_id'] == 6 else -1. for r in records])
    assert not improvement_summary(records, donor_failure, .05)['passed']


def test_primary_gate_conjunction_and_missing_case_rejection():
    cases, parameters = [], []
    for draw in range(32):
        for donor in (6, 7):
            for repeat in (0, 1):
                row = {'draw_id': draw, 'pair_id': draw // 2, 'donor_id': donor, 'repeat': repeat}
                parameters.append({**row, 'errors': {'combined': 0., 'identity': 1., 'analytic': .5, 'stats': .5}})
                for query in (8, 9, 10, 11):
                    cases.append({**row, 'query_id': query, 'arms': {
                        arm: {'float_mse': 0. if arm == 'combined' else 1., 'code_mse': 0. if arm == 'combined' else 1.}
                        for arm in ('combined', 'stats', 'analytic', 'identity', 'mean_render', 'fixed_contrast', 'wrong_primary')}})
    result = primary_decision(cases, parameters, 'combined', .05)
    assert result['reference_contribution'] == 'PASS'
    assert len(result['required']) == 9
    for case in cases:
        case['arms']['mean_render']['code_mse'] = 0.
    assert primary_decision(cases, parameters, 'combined', .05)['reference_contribution'] == 'NOT_ESTABLISHED_STOP'
    with pytest.raises(ValueError):
        primary_decision(cases[:-1], parameters, 'combined', .05)
