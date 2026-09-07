import pytest
import torch

from scripts.prepare_ai_nlut_pretrained import inspect_state


def test_only_nonempty_finite_tensor_state():
    assert inspect_state({"state_dict": {"a": torch.zeros(2)}})[0]["elements"] == 2
    for value in (
        {},
        {"state_dict": {}},
        {"state_dict": {"a": "no"}},
        {"state_dict": {"a": torch.tensor(float("nan"))}},
    ):
        with pytest.raises((ValueError, TypeError)):
            inspect_state(value)
