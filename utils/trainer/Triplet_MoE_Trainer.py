# -*- coding: UTF-8 -*-
"""
@Project ：ECG identify 
@File    ：TripletTrainer.py
@Author  ：yankangli
@Date    ：2025/10/29 14:45 
"""
from pytorch_metric_learning import losses, miners
from utils.trainer import TripletTrainer

class Triplet_MoE_Trainer(TripletTrainer):
    def __init__(self, is_test=False, **kwargs):
        super().__init__(**kwargs)
        self.is_test = is_test
        # proxy_loss = losses.ProxyAnchorLoss(num_classes=self.num_classes, embedding_size=256, 
        #                                     margin=0.1, alpha=32 )
        # self.optimizer.add_param_group({'params': proxy_loss.parameters(),'lr': 1e-2})
        self.metrics_loss = [
            {"loss": losses.CircleLoss(m=0.25,gamma=256,distance = self.distance),"theta":1},
            # {"loss": proxy_loss,"theta":1},
        ]
        
        self.miner = miners.MultiSimilarityMiner(epsilon=0.1, distance=self.distance)
    
    def get_per_loss(self,outputs,targets):
        result = {}
        labels = targets['label']
        embedding = outputs.get('embedding')
        miner_outputs = self.miner(embedding, labels) if self.miner is not None else None
        # 6. 计算 Metric Loss
        if self.metrics_loss and embedding is not None:
            for item in self.metrics_loss:
                loss_fn = item['loss']
                loss_val = loss_fn(embedding, labels, miner_outputs)
                key = item.get('name', loss_fn.__class__.__name__) 
                result[key] = loss_val * item.get("theta", 1.0)
        return result
    def log_start(self):
        log_msg = ""
        if hasattr(self.model, "experts"):
            log_msg = f"{len(self.model.experts)} experts"
        return log_msg
