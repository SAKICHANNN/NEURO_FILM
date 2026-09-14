import pytest
import torch

from src.training.fable_cdfe_checkpoint import load_final_checkpoint, save_final_checkpoint


def test_only_final_checkpoint_opens_and_cannot_be_replaced(tmp_path):
    result = {'optimizer_updates': 2, 'training_mse': [1., .8],
              'state_dict': {'weight': torch.ones(4)}}
    normalizer = {'mean': [0.]*4, 'scale': [1.]*4}
    kwargs = dict(contract_sha256='a'*64, expected_steps=2)
    with pytest.raises(ValueError, match='final-step'):
        save_final_checkpoint(tmp_path, result | {'optimizer_updates': 1},
                              normalizer=normalizer, **kwargs)
    assert not (tmp_path / 'final.claim').exists()
    save_final_checkpoint(tmp_path, result, normalizer=normalizer, **kwargs)
    loaded = load_final_checkpoint(tmp_path, **kwargs)
    assert torch.equal(loaded['state_dict']['weight'], result['state_dict']['weight'])
    with pytest.raises(FileExistsError):
        save_final_checkpoint(tmp_path, result, normalizer=normalizer, **kwargs)
    with pytest.raises(ValueError, match='contract'):
        load_final_checkpoint(tmp_path, **(kwargs | {'contract_sha256': 'b'*64}))
    with (tmp_path / 'final.pt').open('ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(ValueError, match='changed'):
        load_final_checkpoint(tmp_path, **kwargs)
