from abc import abstractmethod
import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from pytorch_metric_learning import miners

from config import PROJECT_PATH, TRAIN_FOLDER
from model import AdamW
from functools import partial
from utils.util.loader import save2pth
from utils import setup_logger, topk_acc

class BaseHook:
    @abstractmethod
    def before_batch(self, batch):
        pass

    def after_batch(self, batch):
        pass

    def before_epoch(self, epoch):
        pass

    def after_epoch(self, epoch):
        pass

    @abstractmethod
    def before_process(self):
        pass

def _to_float(v):
    return v.item() if isinstance(v, torch.Tensor) else v

class BaseMetrics:
    def __init__(self):
        self.total_batches = 0
        self.running_loss = 0.0
        self.running_metrics = {}
    
    def _update_running_stats(self, loss: Any, metrics: Dict[str, float]) -> None:
        self.total_batches += 1
        if isinstance(loss, torch.Tensor):
            self.running_loss += loss.item()
        if not isinstance(loss, dict):
            loss = {"loss": loss}
        main_key = "loss" if "loss" in loss else next(iter(loss))
            
        for k, v in loss.items():
            val = _to_float(v)
            if k == main_key:
                self.running_loss += val
            else:
                self.running_metrics[k] = self.running_metrics.get(k, 0.0) + val
        
        for name, value in metrics.items():
            self.running_metrics[name] = self.running_metrics.get(name, 0.0) + value

    def _reset_running_stats(self) -> None:
        self.running_loss = 0.0
        self.running_metrics = {}
        self.total_batches = 0

    def _get_average_stats(self) -> Tuple[float, Dict[str, float]]:
        divisor = max(self.total_batches, 1)
        avg_loss = self.running_loss / divisor
        avg_metrics = {k: v / divisor for k, v in self.running_metrics.items()}
        
        return avg_loss, avg_metrics


class BaseProcessor(BaseHook, BaseMetrics):
    def __init__(
        self,
        model: nn.Module,
        loss=None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        metrics=None,
        checkpoint_dir: str = "checkpoints",
    ):
        if metrics is None:
            metrics = {"top1": partial(topk_acc, k=1)}
        BaseMetrics.__init__(self)
        self.metrics = metrics
        self.model = model.to(device)
        self.device = device
        self.loss = nn.CrossEntropyLoss() if loss is None else loss
        checkpoint_dir = os.path.join(PROJECT_PATH, TRAIN_FOLDER, checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir
        self.miner = miners.HDCMiner()

        # 初始化指标存储

    def _log_epoch_stats(
        self, epoch=0, phase="train", avg_loss=0.0, avg_metrics=None
    ) -> None:
        log_msg = f"epoch{epoch} {phase}: Loss: {avg_loss:.4f}"
        if isinstance(avg_metrics, dict):
            for name, value in avg_metrics.items():
                log_msg += f" - {name}: {value:.4f}"
        self.train_logger.info(log_msg)

    def before_process(self):
        self.train_logger.info("before process")
        self.train_logger.info(self.model)

    def before(self):
        self.scheduler = (
            ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=5)
        )
        total_params = sum(p.numel() for p in self.model.parameters())
        self.train_logger.info(f"Total parameters: {total_params}")
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self.train_logger.info(f"Trainable parameters: {trainable_params}")
        self.before_process()

    @abstractmethod
    def _process_one_batch(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        raise NotImplementedError("子类必须实现_process_one_batch方法")

    def _compute_metrics(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        targets = targets['label']
        metrics_results = {}
        logits, preds = self.get_logits(outputs)
        for name, metric_func in self.metrics.items():
            metrics_results[name] = metric_func(
                logits=logits, preds=preds, targets=targets
            ).item()
        return metrics_results

    def get_loss(self, outputs, targets):
        loss_dict = self.get_per_loss(outputs, targets)
        loss_dict = self.sum_per_loss(loss_dict)
        return loss_dict

    def sum_per_loss(self, loss_dict):
        total_loss = 0.0
        for name, loss in loss_dict.items():
            total_loss += loss
        loss_dict["loss"] = total_loss
        return loss_dict

    def get_per_loss(self, outputs, targets):
        loss = (
            self.loss(outputs, targets['label'])
            if self.loss is not None
            else torch.tensor(0.0)
        )
        return {"loss": loss}

    def get_logits(self, outputs):
        logits = F.softmax(outputs, dim=1)
        _, preds = logits.max(dim=1)
        return logits, preds

    def before_batch(self, batch):
        if isinstance(batch["seg"], list):
            batch["seg"] = torch.stack(batch["seg"], dim=0)  # [B, V, L]
            V, B, L = batch["seg"].shape
            batch["seg"] = (
                batch["seg"].reshape(B * V, L).unsqueeze(1).to(self.device)
            )
            keys = list(batch.keys())
            for k in keys:
                if k == "seg":
                    continue
                batch[k] = batch[k].repeat_interleave(V, dim=0).to(self.device)
        else:
            if batch["seg"].dim() == 2:
                batch["seg"] = batch["seg"].unsqueeze(1).to(self.device)
            elif batch['seg'].shape[2]!=1 and batch['seg'].dim()==3:
                batch = {k: v.squeeze() for k, v in batch.items()}
                batch['seg'] = batch['seg'].unsqueeze(1).to(self.device)
                if 'origin' in batch.keys():
                    batch['origin'] = batch['origin'].unsqueeze(1).to(self.device)
            batch = {k: v.to(self.device) for k, v in batch.items()}
        return batch

    def get_outputs(self, batch):
        return self.model(batch)
    
    def save_checkpoint(
        self, epoch: int, filename: Optional[str] = None, **kwargs: Any
    ) -> None:

        if filename is None:
            filename = f"checkpoint_best.pth"

        checkpoint_path = os.path.join(self.checkpoint_dir, filename)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            **kwargs,
        }

        # torch.save(checkpoint, checkpoint_path)
        save2pth(checkpoint,checkpoint_path)
        self.train_logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.train_logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint


class BaseTrainer(BaseProcessor):
    """训练器类，用于模型训练"""

    def __init__(
        self,
        model: nn.Module,
        lr=0.001,
        train_loader=None,
        train_val_loader=None,
        test_loader=None,
        val_loader=None,
        close_val_loader=None,
        open_val_loader=None,
        open_loader=None,
        optimizer=None,
        loss=None,
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
        metrics: Optional[Dict[str, callable]] = None,
        checkpoint_dir: str = "checkpoints",
        model_name="best_model.pth",
        val_frequency: int = 1,
        **kwargs,
    ):
        super().__init__(model, loss, device, metrics, checkpoint_dir)

        self.optimizer = (
            AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=1e-5,
                no_deprecation_warning=True,
            )
            if optimizer is None
            else optimizer(
                model.parameters(), model.parameters(), lr=lr, weight_decay=1e-5
            )
        )
        # 创建数据加载器
        self.train_loader = train_loader
        self.train_val_loader = train_val_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.close_val_loader = close_val_loader
        self.open_val_loader = open_val_loader

        self.open_loader = open_loader
        self.val_frequency = val_frequency
        self.best_val_metric = -np.inf  # 用于跟踪最佳验证指标
        self.best_val_epoch = 0
        self.model_name = model_name
    def get_tqdm_info(self, outputs, loss):
        # 情况 1：loss 是 Tensor
        if isinstance(loss, torch.Tensor):
            return {"loss": loss.item()}

        # 情况 2：loss 是 dict
        if isinstance(loss, dict):
            info = {}
            for k, v in loss.items():
                if isinstance(v, torch.Tensor):
                    info[k] = v.item()
                else:
                    info[k] = v
            return info

        # 情况 3：兜底
        else:
            return {"loss": loss}
    def get_batch_targets(self, batch):
        result = {}
        result['label'] = batch.get("label", batch.get('anchor',{}).get('label')).squeeze().to(self.device)
        return result
    def _process_one_batch(self, batch):
        # 清零梯度
        self.optimizer.zero_grad()
        targets = self.get_batch_targets(batch)
        # 前向传播
        with torch.set_grad_enabled(True):
            outputs = self.get_outputs(batch)
            loss = self.get_loss(outputs, targets)
            loss["loss"].backward()
            self.optimizer.step()

        return outputs, targets, loss

    def train_one_epoch(self, epoch: int) -> Tuple[float, Dict[str, float]]:

        self.model.train()
        self._reset_running_stats()

        start_time = time.time()

        for batch in tqdm(self.train_loader, desc=f"Training Epoch {epoch}"):
            batch = self.before_batch(batch)
            outputs, targets, loss = self._process_one_batch(batch)
            metrics = self._compute_metrics(outputs, targets)
            self._update_running_stats(loss, metrics)

        epoch_time = time.time() - start_time
        avg_loss, avg_metrics = self._get_average_stats()
        # self._log_epoch_stats('train', avg_loss, avg_metrics)

        return avg_loss, avg_metrics

    def validate(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        if self.val_loader is None:
            self.train_logger.warning("没有提供验证数据集，跳过验证")
            return 0.0, {}

        self.model.eval()
        self._reset_running_stats()

        start_time = time.time()

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Validating Epoch {epoch}"):
                batch = self.before_batch(batch)
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.get_outputs(inputs)
                loss = self.get_loss(outputs, targets)
                metrics = self._compute_metrics(outputs, targets)

                self._update_running_stats(loss, metrics)

        epoch_time = time.time() - start_time
        avg_loss, avg_metrics = self._get_average_stats()
        # self._log_epoch_stats('val', avg_loss, avg_metrics)

        return avg_loss, avg_metrics

    def train(
        self,
        num_epochs=10,
        start_epoch: int = 0,
        save_best: bool = True,
        k: int = 0,
    ) -> None:
        self.before()
        filepath = self.model_name+f"_{k}.pth"
        for epoch in range(start_epoch, num_epochs):
            # 训练一个 epoch
            train_loss, train_metrics = self.train_one_epoch(epoch)

            # 验证
            val_loss, val_metrics = 0.0, {}
            if self.val_loader is not None and (epoch + 1) % self.val_frequency == 0:
                val_loss, val_metrics = self.validate(epoch)
                self._log_epoch_stats(epoch, "train", train_loss, train_metrics)
                log_msg = f"val: Loss: {val_loss:.4f}"
                for name, value in val_metrics.items():
                    log_msg += f" - {name}: {value:.4f} - best val acc:  {self.best_val_metric:.4f}"
                self.train_logger.info(log_msg)
                # 保存最佳模型
                if save_best:
                        self.best_val_epoch = epoch
                        self.save_checkpoint(
                            epoch,
                            self.model_name,
                            val_loss=val_loss,
                            val_metrics=val_metrics,
                            optimizer_state_dict=self.optimizer.state_dict(),
                        )
            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
                self.train_logger.info(
                    f"Current learning rate: {self.scheduler.get_last_lr()[0]:.6f}"
                )
        # 训练结束时保存最后一个 epoch 的模型
        # self.save_checkpoint(num_epochs - 1, self.model_name,
        #                      optimizer_state_dict=self.optimizer.state_dict())


    def test(self):
        if self.test_loader is None:
            print("没有提供验证测试集，跳过验证")
            return 0.0, {}

        self.model.eval()
        self._reset_running_stats()

        # 短时间内
        logits_list = []
        with torch.no_grad():
            for batch in tqdm(self.train_val_loader, desc=f"Test"):
                batch = self.before_batch(batch)
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}

                outputs = self.get_outputs(inputs)
                loss = self.get_loss(outputs, targets)
                metrics = self._compute_metrics(outputs, targets)
                if hasattr(self.model, "loss") and self.model.loss:
                    logits, preds = self.get_logits(outputs)
                    logits_list.append(logits)
                self._update_running_stats(loss, metrics)
        _, close_metrics = self._get_average_stats()
        # 短时间内识别未知个体
        if hasattr(self.model, "loss") and self.model.loss:
            open_logits_list = []
            with torch.no_grad():
                for batch in tqdm(self.close_val_loader, desc=f"Test"):
                    batch = self.before_batch(batch)
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    outputs = self.get_outputs(inputs)
                    loss = self.get_loss(outputs, targets)
                    metrics = self._compute_metrics(outputs, targets)
                    logits, preds = self.get_logits(outputs)
                    open_logits_list.append(logits)
                    self._update_running_stats(loss, metrics)
            eer, threshold, auc = eer(logits_list, open_logits_list)
            close_metrics["eer"] = eer
            close_metrics["threshold"] = threshold
            close_metrics["auc"] = auc
        # 开集
        logits_list = []
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc=f"Test"):
                batch = self.before_batch(batch)
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.get_outputs(inputs)
                loss = self.get_loss(outputs, targets)
                metrics = self._compute_metrics(outputs, targets)
                if hasattr(self.model, "loss") and self.model.loss:
                    logits, preds = self.get_logits(outputs)
                    logits_list.append(logits)
                self._update_running_stats(loss, metrics)

        _, open_metrics = self._get_average_stats()
        if hasattr(self.model, "loss") and self.model.loss:
            open_logits_list = []
            with torch.no_grad():
                for batch in tqdm(self.open_val_loader, desc=f"Test"):
                    batch = self.before_batch(batch)
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    outputs = self.get_outputs(inputs)
                    loss = self.get_loss(outputs, targets)
                    metrics = self._compute_metrics(outputs, targets)
                    logits, preds = self.get_logits(outputs)
                    open_logits_list.append(logits)
                    self._update_running_stats(loss, metrics)
            eer, threshold, auc = eer(logits_list, open_logits_list)
            open_metrics["eer"] = eer
            open_metrics["threshold"] = threshold
            open_metrics["auc"] = auc
        log_msg = self.log_start()
        log_msg += f"\nclose metrics:"
        for name, value in close_metrics.items():
            log_msg += f" - {name}: {value:.4f}"
        log_msg += f"\nopen  metrics:"
        for name, value in open_metrics.items():
            log_msg += f" - {name}: {value:.4f}"
        test_logger = setup_logger(log_dir=self.checkpoint_dir, sub_path="test_info", log_filename="test_logs.log")
        test_logger.info(log_msg)
        return close_metrics, open_metrics

    def log_start(self):
        log_msg = ""
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "seg"):
            log_msg = f"{self.model.encoder.seg}"
        return log_msg
