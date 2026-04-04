import os
import sys
import numpy as np

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from config import T_START, R_PEAK, FS, DATA_LENGTH


class ReprMixin:
    _repr_types = (int, float, bool, str, type(None),tuple,list,dict)

    def __repr__(self):
        attrs = ", ".join(
            f"{k}={v!r}"
            for k, v in self.__dict__.items()
            if isinstance(v, self._repr_types)
        )
        return f"{self.__class__.__name__}({attrs})"

class RandomAmplitudeScale(ReprMixin):
    def __init__(self, scale_range=(0.9, 1.1)):
        self.scale_range = scale_range

    def __call__(self, signal, **kwargs):
        scale = np.ones_like(signal, dtype=np.float32)
        scale_factor = np.random.uniform(*self.scale_range)
        scale[:] = scale_factor
        return signal * scale

class AddGaussianNoise(ReprMixin):
    def __init__(self, noise_std_range=(0.01, 0.3)):
        self.noise_std_range = noise_std_range

    def __call__(self, signal, **kwargs):
        std = np.random.uniform(*self.noise_std_range)
        noise = np.random.normal(0, std, size=signal.shape)
        return signal + noise


class ToTensor1D(ReprMixin):
    def __call__(self, signal):
        # 1. 如果是 list，先转 numpy
        if isinstance(signal, list):
            signal = np.array(signal)
        if isinstance(signal, np.ndarray):
            signal = torch.from_numpy(signal)
        if isinstance(signal, torch.Tensor):
            signal = signal.float()
        return signal
