import numpy as np

from src.eval.fable_cdfe_gates import donor_control_gates, validation_gate
from src.eval.fable_cdfe_gates import stratified_photometry_gates


def test_sign_boundary_ties_and_strict_margin():
    learned = np.ones((32,32,4))
    control = learned.copy()
    control[:22] = 2
    d = np.ones_like(learned)
    r = donor_control_gates(learned, control, control, d)
    assert not r['passed'] and not r['controls']['constant']['sign_passed']
    control[22] = 2
    assert donor_control_gates(learned, control, control, d)['passed']
    zero = np.zeros_like(learned)
    r = donor_control_gates(zero, learned, learned, d*10)
    assert r['controls']['constant']['sign_passed']
    assert not r['controls']['constant']['margin_passed']


def test_validation_zero_baseline_cannot_pass():
    target = np.zeros((32,8,4))
    assert not validation_gate(target, target, np.zeros(4))['passed']
    target[:] = 1
    assert validation_gate(target, target, np.zeros(4))['passed']
    assert not validation_gate(np.zeros_like(target), target, np.zeros(4))['passed']


def test_local_donor_failure_cannot_hide_in_pooled_mean():
    e = np.ones((32,32,4))
    e[0] = 30
    labels = dict(donor_ids=[str(i) for i in range(32)], query_ids=list('abcd'),
                  donor_cameras=['camera']*32, treatment_regions=[str(i//8) for i in range(32)])
    result = stratified_photometry_gates(e, np.full_like(e, 100), np.full_like(e, 16), **labels)
    assert result['pooled']['passed']
    assert not result['donors']['0']['passed']
    assert not result['photometry_passed']
    e[:] = 25
    assert stratified_photometry_gates(e, np.full_like(e, 100), np.full_like(e, 16), **labels)['photometry_passed']
