import copy

import numpy as np
import pytest

from scripts.assess_fable_paired_contrast import assess_cell, branch_decision, validate_prediction_layout


def cases(donors):
    result = []
    for donor in donors:
        for draw in range(32):
            for query in (8, 9, 10, 11):
                arms = {}
                for arm in ('after_only', 'paired_control', 'frozen_combined', 'identity', 'mean_render', 'wrong_after_only', 'oracle'):
                    error = 4. if arm in ('after_only', 'paired_control') else 64.
                    if arm == 'oracle':
                        error = 0.
                    arms[arm] = {'code_E': error, 'float_E': error / 65025,
                                 'code_P': 64., 'float_P': 64. / 65025}
                result.append({'donor_id': donor, 'draw_id': draw, 'pair_id': draw // 2, 'query_id': query, 'arms': arms})
    return result


def test_complete_gate_and_order_invariant_equal_case_aggregation():
    rows = cases([12, 13, 14, 15])
    result = assess_cell(rows, [12, 13, 14, 15], [8, 9, 10, 11])
    assert result['after_only_complete_gate']
    assert all(t['wins'] == 16 and t['cases_per_block'] == 32 for t in result['after_only_sign_tests'].values())
    np.random.default_rng(8).shuffle(rows)
    shuffled = assess_cell(rows, [12, 13, 14, 15], [8, 9, 10, 11])
    for key, test in result['after_only_sign_tests'].items():
        np.testing.assert_array_equal(test['block_margins'], shuffled['after_only_sign_tests'][key]['block_margins'])
    with pytest.raises(ValueError):
        assess_cell(rows[:-1], [12, 13, 14, 15], [8, 9, 10, 11])


def test_tie_margins_remain_nonwins_and_paired_success_cannot_rescue():
    rows = cases([12, 13, 14, 15])
    for row in rows:
        if row['pair_id'] >= 11:
            row['arms']['frozen_combined']['code_E'] = row['arms']['after_only']['code_E']
    result = assess_cell(rows, [12, 13, 14, 15], [8, 9, 10, 11])
    test = result['after_only_sign_tests']['frozen_combined_error_minus_after_error']
    assert test['wins'] == 11 and test['ties'] == 5 and test['blocks'] == 16
    assert not result['after_only_complete_gate']
    assert result['engineering_gates']['paired_control']['passed']
    secondary = assess_cell(cases(list(range(6))), list(range(6)), [8, 9, 10, 11])
    assert branch_decision(result, secondary)['action'] == 'CLOSE_FIXED_DESCRIPTOR_LINEAR_DECODING_ROUTE'
    passed = copy.deepcopy(result)
    passed['after_only_complete_gate'] = True
    assert branch_decision(passed, secondary)['action'] == 'ADMIT_REAL_TWO_GRADES_FOUR_FRESH_SCENES_CALIBRATION'


def test_donor_failure_and_zero_strength_cannot_hide_in_pooled_metrics():
    rows = cases([12, 13, 14, 15])
    for row in rows:
        if row['donor_id'] == 12:
            row['arms']['after_only']['code_E'] = 20.
    result = assess_cell(rows, [12, 13, 14, 15], [8, 9, 10, 11])
    assert result['engineering_gates']['after_only']['accuracy']['pooled']['passed']
    assert not result['after_only_complete_gate']
    for row in rows:
        row['arms']['oracle']['code_P'] = 0.
    assert not assess_cell(rows, [12, 13, 14, 15], [8, 9, 10, 11])['after_only_complete_gate']


def test_wrong_reference_mapping_validates_full_fresh_predictions_after_shuffle():
    draws = np.random.default_rng(3).uniform(-1, 1, (16, 4))
    draws = np.stack([draws, -draws], axis=1).reshape(32, 4)
    donors = np.repeat([0, 1, 2, 3, 4, 5, 12, 13, 14, 15], 32)
    edits = np.tile(np.arange(32), 10)
    data = {'donor_ids': donors, 'draw_ids': edits, 'pair_ids': edits // 2, 'targets': draws[edits]}
    for arm in ('after_only', 'paired_control', 'frozen_combined'):
        data[arm] = draws[edits].copy()
        data[arm + '_raw'] = draws[edits].copy()
        data[arm + '_clipped'] = np.zeros((320, 4), dtype=bool)
    permutation = np.random.default_rng(11).permutation(320)
    data = {key: value[permutation] for key, value in data.items()}
    wrong = validate_prediction_layout(data, draws)
    np.testing.assert_array_equal(data['donor_ids'], data['donor_ids'][wrong])
    np.testing.assert_array_equal(data['draw_ids'] ^ 1, data['draw_ids'][wrong])
    np.testing.assert_array_equal(wrong[wrong], np.arange(320))
    data['after_only'][0, 0] += .01
    with pytest.raises(ValueError):
        validate_prediction_layout(data, draws)
