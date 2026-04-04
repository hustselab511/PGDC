import numpy as np
import random
from collections import defaultdict
from torch.utils.data import Sampler

class WeightedSessionBalancedSampler(Sampler):
    def __init__(self, dataset, n_classes, n_samples, class_weights=None, n_batches=None):
        """
        参数:
            n_classes (P): 每个 Batch 选多少个类别
            n_samples (K): 每个类别选多少个样本 (P x K Sampling)
            class_weights: dict {label: weight} 或 list (对应 sorted(labels))
        """
        self.labels = dataset.labels
        self.sessions = dataset.session
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.batch_size = n_classes * n_samples
        
        self.index_map = defaultdict(lambda: defaultdict(list))
        for idx, (lbl, ses) in enumerate(zip(self.labels, self.sessions)):
            self.index_map[lbl][ses].append(idx)
            
        # 确保标签有序，以便与列表形式的权重对齐
        self.unique_labels = sorted(list(self.index_map.keys()))
        
        # 2. 计算 Epoch 长度
        if n_batches is None:
            self.n_batches = len(self.labels) // self.batch_size
        else:
            self.n_batches = n_batches

        # 3. 处理类别权重 (归一化为概率)
        self.class_probs = None
        if class_weights is not None:
            if isinstance(class_weights, dict):
                weights = np.array([class_weights.get(l, 1.0) for l in self.unique_labels])
            else:
                weights = np.array(class_weights) # 假设顺序与 sorted(unique_labels) 一致
            
            # 防止除以0
            if weights.sum() > 0:
                self.class_probs = weights / weights.sum()

    def __iter__(self):
        for _ in range(self.n_batches):
            batch_indices = []
            selected_classes = np.random.choice(
                self.unique_labels, 
                size=self.n_classes, 
                replace=False, 
                p=self.class_probs
            )
            
            for lbl in selected_classes:
                sessions = list(self.index_map[lbl].keys())
                if len(sessions) >= self.n_samples:
                    chosen_sessions = np.random.choice(sessions, size=self.n_samples, replace=False)
                else:
                    chosen_sessions = sessions + list(np.random.choice(sessions, size=self.n_samples - len(sessions), replace=True))

                for ses in chosen_sessions:
                    idx = random.choice(self.index_map[lbl][ses])
                    batch_indices.append(idx)
            
            yield batch_indices

    def __len__(self):
        return self.n_batches