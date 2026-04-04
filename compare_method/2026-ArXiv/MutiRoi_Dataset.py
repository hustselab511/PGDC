import numpy as np
import torch
from torch.utils.data import Dataset, default_collate


class MutiRoiDataset(Dataset):
    def __init__(
        self,
        segments: np.ndarray,
        labels: np.ndarray,
    ):
        super().__init__()

        self.segments = np.asarray(segments, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)

        assert self.segments.ndim == 2, "segments 应为 (N, 240)"
        assert self.segments.shape[1] == 240, "每个样本长度应为 240 = 60+40+60+80"

        self.num_classes = len(np.unique(self.labels)) if labels is not None else 0

        self.roi_slices = {
            "p": slice(0, 60),
            "qrs": slice(60, 100),
            "st": slice(100, 160),
            "tu": slice(160, 240),
        }

    def __len__(self):
        return len(self.labels)

    def _split_roi(self, seg: np.ndarray):
        return {
            "p": seg[self.roi_slices["p"]],
            "qrs": seg[self.roi_slices["qrs"]],
            "st": seg[self.roi_slices["st"]],
            "tu": seg[self.roi_slices["tu"]],
        }

    def _to_tensor_dict(self, roi_dict):
        out = {}
        for k, v in roi_dict.items():
            # 注意：这里不 unsqueeze，保持 (L,)
            out[k] = torch.as_tensor(v.copy(), dtype=torch.float32)
        return out

    def __getitem__(self, idx):
        if isinstance(idx, (list, tuple, np.ndarray, torch.Tensor, slice)):
            if isinstance(idx, slice):
                indices = range(*idx.indices(len(self)))
            else:
                indices = idx
            items = [self[i] for i in indices]
            return default_collate(items)

        seg = self.segments[idx].copy()
        roi_dict = self._split_roi(seg)
        roi_tensor_dict = self._to_tensor_dict(roi_dict)

        sample = {
            **roi_tensor_dict,
            "label": torch.as_tensor(self.labels[idx], dtype=torch.long),
        }
        return sample