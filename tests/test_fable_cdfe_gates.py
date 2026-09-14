import numpy as np

from src.eval.fable_cdfe_gates import donor_control_gates, validation_gate


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
