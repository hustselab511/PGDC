import os
import sys

import torch

from config import (
    S_END,
    T_START,
)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

def _apply_mask(x, seg, dim):
    seg = seg.lower()
    if seg not in ['pqrs', 'st_t']:
        return x
    t = torch.arange(x.shape[dim], device=x.device)
    mask = (t < S_END) if seg == 'pqrs' else (t >= T_START)
    view_shape = [1] * x.ndim
    view_shape[dim] = -1
    return x * mask.view(view_shape)

def init_x(x, seg='seg'):
    return _apply_mask(x, seg, dim=2)


def reshape_tensor(x):
    shape = x.shape
    if len(shape) == 3 and shape[1] == 1:
        return x.squeeze(1).unsqueeze(-1)

    elif len(shape) == 2:
        return x.unsqueeze(-1)
    return x


if __name__ == '__main__':
    x = torch.randn(2, 1, 100)
    x = init_x(x, seg='qrs')
    print(x.shape)
    x = reshape_tensor(x)
    print(x.shape)
