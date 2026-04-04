import torch
from torch.utils.data import Dataset
import numpy as np

class ECGDataset(Dataset):
    def __init__(self, data, labels, n_concat=1):
        """
        data: shape (num_total_segments, 256)
        labels: shape (num_total_segments,)
        n_concat: 拼接的心跳数量 N [cite: 150]
        """
        self.n_concat = n_concat
        self.segments = []
        self.targets = []
        
        # 按类别（受试者）进行拼接处理
        unique_labels = np.unique(labels)
        self.class_num = unique_labels.size
        data = np.array(data)
        labels = np.array(labels)
        for label in unique_labels:
            idx = np.where(labels == label)[0]
            subj_data = data[idx]
            
            for i in range(0, len(subj_data) - n_concat + 1, n_concat):
                concat_seg = subj_data[i : i + n_concat].flatten() # 拼接为 N*L 长度
                self.segments.append(concat_seg)
                self.targets.append(label)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        x = torch.from_numpy(self.segments[index]).float().unsqueeze(0) # (1, N*L)
        y = torch.tensor(self.targets[index]).long()
        return x, y
    