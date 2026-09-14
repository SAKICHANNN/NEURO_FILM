import numpy as np
import torch

from src.eval.fable_canonical_prior import canonical_measure, decode_canonical_prior


class AfterOnlyCanonicalPredictor:
    def __init__(self, network: torch.nn.Module, *, target_mean: np.ndarray, target_scale: np.ndarray,
                 epsilon: float, scale_floor: float, tensor_limit: float,
                 slope_limit: float, offset_limit: float):
        self.network = network
        self.target_mean = np.array(target_mean, dtype=np.float64, copy=True)
        self.target_scale = np.array(target_scale, dtype=np.float64, copy=True)
        if (self.target_mean.shape != (4,) or self.target_scale.shape != (4,)
                or not np.isfinite(self.target_mean).all() or not np.isfinite(self.target_scale).all()
                or np.any(self.target_scale <= 0)):
            raise ValueError('four finite fitting-only target means and positive scales required')
        self.measure_bounds = dict(epsilon=epsilon, scale_floor=scale_floor, tensor_limit=tensor_limit)
        self.decode_bounds = dict(slope_limit=slope_limit, offset_limit=offset_limit)

    @torch.inference_mode()
    def predict(self, after: np.ndarray) -> list[dict]:
        images = np.asarray(after)
        if images.ndim != 4 or images.shape[0] == 0 or images.shape[-1] != 3:
            raise ValueError('nonempty BHWC AFTER images required')
        if self.network.training:
            raise ValueError('network must be in evaluation mode')
        measures = [canonical_measure(image, **self.measure_bounds) for image in images]
        parameter = next(self.network.parameters())
        tensors = torch.as_tensor(np.stack([m['tensor'] for m in measures]).transpose(0, 3, 1, 2),
                                  device=parameter.device, dtype=parameter.dtype)
        standardized = self.network(tensors).cpu().numpy().astype(np.float64)
        if standardized.shape != (len(images), 4) or not np.isfinite(standardized).all():
            raise ValueError('network must return four finite standardized statistics per AFTER')
        predicted = standardized * self.target_scale + self.target_mean
        return [dict(decode_canonical_prior(m, p, **self.decode_bounds), predicted_canonical=p)
                for m, p in zip(measures, predicted)]
