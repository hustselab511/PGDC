# -*- coding: UTF-8 -*-
'''
@Project ：ECG identify
@File    ：feature_embedding.py
@Author  ：yankangli (Optimized)
@Date    ：2025/10/23
'''
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Callable

class PrototypesTracker:
    """
    优化版特征模板跟踪器：支持增量更新、数值安全检查及大规模 ID 管理。
    """

    def __init__(self, eps: float = 1e-8):
        self.templates: Dict[int, torch.Tensor] = {}
        self.counts: Dict[int, int] = {}
        self.eps = eps  # 用于防止除零的极小值

        # 缓存变量
        self._template_matrix: Optional[torch.Tensor] = None
        self._template_labels: Optional[torch.Tensor] = None
        self._dirty = False 

    def _invalidate_cache(self):
        self._dirty = True
        self._template_matrix = None
        self._template_labels = None

    def update(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        更新模板特征（带数值安全检查）
        """
        # 1. 基本清洗：确保在 CPU 且无梯度
        embeddings = embeddings.detach().cpu()
        labels = labels.cpu()

        # 2. 全量数值检查：跳过包含 NaN/Inf 的样本
        valid_mask = torch.isfinite(embeddings).all(dim=1)
        if not valid_mask.all():
            invalid_count = (~valid_mask).sum().item()
            print(f"⚠️ 拦截到 {invalid_count} 条包含 NaN/Inf 的输入样本，已跳过。")
            embeddings = embeddings[valid_mask]
            labels = labels[valid_mask]

        if len(labels) == 0:
            return

        unique_labels = labels.unique()
        for label in unique_labels:
            label_item = label.item()
            idxs = (labels == label).nonzero(as_tuple=True)[0]
            emb_subset = embeddings[idxs]

            # 计算当前 batch 的类均值
            batch_mean_emb = emb_subset.mean(dim=0)
            batch_count = len(emb_subset)

            if label_item not in self.templates:
                self.templates[label_item] = batch_mean_emb
                self.counts[label_item] = batch_count
            else:
                # 3. 增量更新公式优化
                old_count = self.counts[label_item]
                new_count = old_count + batch_count

                # 采用加权平均，并防止中间过程溢出
                new_val = (self.templates[label_item] * (old_count / new_count)) + \
                          (batch_mean_emb * (batch_count / new_count))
                
                # 最终确认更新值是否有效
                if torch.isfinite(new_val).all():
                    self.templates[label_item] = new_val
                    self.counts[label_item] = new_count

        self._invalidate_cache()

    def standardize(self):
        """
        对模板进行安全归一化
        """
        for label_item in self.templates:
            temp = self.templates[label_item]
            norm = temp.norm(p=2, dim=0, keepdim=True)
            # 增加 eps 防止模长为 0 导致归一化后产生 NaN
            self.templates[label_item] = temp / (norm + self.eps)
        self._invalidate_cache()

    def _sync_cache(self, device: torch.device):
        """
        将字典同步为 Tensor 矩阵（向量化预测加速）
        """
        if self._dirty or self._template_matrix is None or self._template_matrix.device != device:
            if not self.templates:
                return

            sorted_keys = sorted(self.templates.keys())
            self._template_labels = torch.tensor(sorted_keys, device=device, dtype=torch.long)
            
            # 使用列表推导式高效构建矩阵
            tensors = [self.templates[k].to(device) for k in sorted_keys]
            self._template_matrix = torch.stack(tensors)
            self._dirty = False

    def predict(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        向量化余弦相似度预测
        """
        if not self.templates:
            raise ValueError("模板库为空，无法预测")

        device = embeddings.device
        self._sync_cache(device)

        # 安全归一化输入
        norm_emb = F.normalize(embeddings, p=2, dim=1, eps=self.eps)
        norm_temp = F.normalize(self._template_matrix, p=2, dim=1, eps=self.eps)

        # 计算相似度矩阵 [Batch, Classes]
        similarities = torch.mm(norm_emb, norm_temp.T)
        indices = similarities.argmax(dim=1)
        predictions = self._template_labels[indices]

        return predictions, similarities

    def clear_nan(self):
        """
        手动清理模板库中已经存在的 NaN 项
        """
        bad_keys = [k for k, v in self.templates.items() if not torch.isfinite(v).all()]
        for k in bad_keys:
            del self.templates[k]
            del self.counts[k]
        if bad_keys:
            print(f"🧹 已从模板库中清理掉 {len(bad_keys)} 个坏损 ID")
            self._invalidate_cache()
    def copy(self):
        """
        创建当前追踪器的深拷贝，确保所有 Tensor 都在新内存中。
        """
        import copy
        
        # 创建一个新实例
        new_instance = self.__class__(eps=self.eps)
        
        # 深拷贝模板和计数字典 (确保 Tensor 被 clone)
        new_instance.templates = {k: v.clone() for k, v in self.templates.items()}
        new_instance.counts = self.counts.copy()
        
        # 复制状态标志
        new_instance._dirty = self._dirty
        
        # 缓存处理：如果原对象有缓存且不是 dirty，则克隆缓存；否则保持 None
        if not self._dirty and self._template_matrix is not None:
            new_instance._template_matrix = self._template_matrix.clone()
            new_instance._template_labels = self._template_labels.clone()
            
        return new_instance
    def distill_update(self, other: 'PrototypesTracker', alpha: float = 0.2):
        """
        蒸馏更新：当前原型向目标原型靠近。
        alpha: 融合系数。0.2 表示保留 80% 的旧知识，吸收 20% 的新知识。
        """
        common_ids = set(self.templates.keys()) & set(other.templates.keys())
        new_ids = set(other.templates.keys()) - set(self.templates.keys())

        # 1. 共有 ID：进行平滑演进
        for label in common_ids:
            target_val = other.templates[label].to(self.templates[label].device)
            self.templates[label] = (1.0 - alpha) * self.templates[label] + alpha * target_val
            self.counts[label] += int(other.counts[label] * alpha)

        # 2. 新增 ID：直接吸纳
        for label in new_ids:
            self.templates[label] = other.templates[label].clone()
            self.counts[label] = other.counts[label]

        self._invalidate_cache()
        print(f"✅ 蒸馏完成。共有 ID 更新: {len(common_ids)} 个，新增 ID: {len(new_ids)} 个。")