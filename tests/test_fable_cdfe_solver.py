import runpy
from pathlib import Path

from src.eval.fable_cdfe_solver import solve_cdfe_assignment


def cohort():
    fixture = runpy.run_path(str(Path(__file__).with_name('test_fable_cdfe_assignment.py')))['fixture']
    rows, _ = fixture()
    for n, row in enumerate(rows):
        row['candidate_rank'] = f'{n:04d}'
    return rows, {camera: camera for camera in {r['camera'] for r in rows}}


def test_joint_solver_accepts_shared_consumed_component():
    rows, ranks = cohort()
    result = solve_cdfe_assignment(rows, ranks, [])
    assert result is not None
    assert len(result['fit']) == 256
    assert result['held_cameras'] == ['H0', 'H1', 'H2', 'H3']


def test_fresh_component_collision_proves_infeasible():
    rows, ranks = cohort()
    rows[-1]['component'] = rows[-2]['component']
    assert solve_cdfe_assignment(rows, ranks, []) is None


def test_smoke_camera_cannot_be_held():
    rows, ranks = cohort()
    rows.append(dict(rows[0], identity='smoke', camera='H0', candidate_rank='9999', technical_eligible=False))
    assert solve_cdfe_assignment(rows, ranks, ['smoke']) is None


def test_lexicographic_capture_choice_independent_of_input_order():
    rows, ranks = cohort()
    rows.append(dict(rows[0], identity='extra_fit', candidate_rank='9999'))
    result = solve_cdfe_assignment(rows, ranks, [])
    assert 'extra_fit' not in result['fit']
    assert solve_cdfe_assignment(list(reversed(rows)), ranks, []) == result
