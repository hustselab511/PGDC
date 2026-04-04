# -*- coding: UTF-8 -*-
"""
@Project ：ECG identify
@File    ：TripletTrainer.py
@Author  ：yankangli
@Date    ：2025/10/29 14:45
"""
from functools import partial
from math import inf
from typing import Optional, Dict, Tuple
import os
import csv
import pandas as pd
import torch
import torch.nn.functional as F
from pytorch_metric_learning import losses, distances
from tqdm import tqdm

from model import ExpertEncoder,PrototypesTracker
from utils.trainer import BaseTrainer
from utils.util.logger import setup_logger
from utils import (
    topk_acc,
    subject_accuracy,
    eval_allpairs_eer,
    save2npy,
)

class TripletTrainer(BaseTrainer):
    def __init__(
        self,
        model=ExpertEncoder,
        train_loader=None,
        val_loader=None,
        model_name="model",
        checkpoint_dir="checkpoint",
        **kwargs,
    ):
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            model_name=model_name,
            checkpoint_dir=checkpoint_dir,
            **kwargs,
        )

        self.unregistered_loader = kwargs.get("unregistered_loader", None)
        self.num_beats = kwargs.get("num_beats", 5)
        self.num_classes = kwargs.get("num_classes", 5)
        
        
        self.metrics = {
            "top1": partial(topk_acc, k=1),
            "top3": partial(topk_acc, k=3),
            "top5": partial(topk_acc, k=5),
        }
        self.tracker = PrototypesTracker()
        self.best_val = inf
        self.enroll_data = {}
        self.registeded_data = {}
        self.unregistered_data = {}
        self.low_acc_ids = []
        self.high_acc_ids = []
        self.distance = distances.CosineSimilarity()
        self.miner = None
        self.metrics_loss = [
            {"loss": losses.SupConLoss(temperature=0.1),"theta":1}
        ]

    def get_outputs(self, batch):
        return self.model(batch['seg'])


    def before(self):
        super().before()
        for metrics_dict in self.metrics_loss:
            metrics_loss = metrics_dict['loss']
            loss_name = metrics_loss.__class__.__name__

            main_params = {
                k: v for k, v in metrics_loss.__dict__.items() 
                if isinstance(v, (int, float, str, bool)) and not k.startswith('_')
            }

            if hasattr(metrics_loss, 'distance'):
                main_params['distance'] = metrics_loss.distance.__class__.__name__
            if hasattr(metrics_loss, 'reducer'):
                main_params['reducer'] = metrics_loss.reducer.__class__.__name__

            # 4. 打印到日志
            self.train_logger.info(f"metrics loss: {loss_name} | params: {main_params}")
    
    def get_metric_loss(self,outputs,targets,seg = ''):
        # 计算度量损失
        result = {}
        labels = targets['label']
        embedding = outputs.get('embedding')
        miner_outputs = self.miner(embedding, labels) if self.miner is not None else None
        # 6. 计算 Metric Loss
        if self.metrics_loss and embedding is not None:
            for item in self.metrics_loss:
                loss_fn = item['loss']
                loss_val = loss_fn(embedding, labels, miner_outputs)
                key = seg+'_'+item.get('name', loss_fn.__class__.__name__) 
                result[key] = loss_val * item.get("theta", 1.0)
        return result
    
    def get_ce_loss(self,outputs,targets,seg = ''):
        # 计算分类损失
        result = {}
        labels = targets['label']
        logit = outputs.get('logit', outputs.get('anchor', {}).get('logit'))
        if logit is not None:  
            result[f"{seg}_ce_loss"] = F.cross_entropy(logit, labels)
        return result
    
    def get_per_loss(self, outputs, targets):
        if 'seg' not in outputs:
            outputs = {'':outputs}
        result = {}
        
        for s,output in outputs.items():
            result.update(self.get_ce_loss(seg=s,outputs=output,targets=targets))
            # result.update(self.get_metric_loss(seg=s,outputs=output,targets=targets))
        result.update(self.get_multiview_loss(outputs=outputs,targets = targets))
        return result
    
    def get_multiview_loss(self, outputs, targets):
        # 计算多视图损失
        embeddings = [out['embedding'] for out in outputs.values() if 'embedding' in out]
        if len(embeddings) < 2:
            return {}
        stacked_embeddings = torch.cat(embeddings, dim=0)
        labels = targets['label']
        stacked_labels = torch.cat([labels] * len(embeddings), dim=0)
        miner_outputs = self.miner(stacked_embeddings, stacked_labels) if self.miner is not None else None
        result = {}
        if self.metrics_loss:
            for item in self.metrics_loss:
                loss_fn = item['loss']
                loss_val = loss_fn(stacked_embeddings, stacked_labels, miner_outputs)
                key = 'multiview_' + item.get('name', loss_fn.__class__.__name__)
                result[key] = loss_val * item.get("theta", 1.0) 
        
        return result
    def before_batch(self, batch):
        def _process_item(d):
            seg = d['seg'].squeeze()
            if seg.dim() == 1:
                seg = seg.unsqueeze(0)
            d['seg'] = seg.unsqueeze(1)
            if 'origin' in d:
                d['origin'] = d['origin'].squeeze().unsqueeze(1)
            return {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in d.items()}

        if 'anchor' in batch:
            return {k: _process_item(v) for k, v in batch.items()}

        if isinstance(batch.get("seg"), list):
            seg_stack = torch.stack(batch["seg"], dim=0)
            V, B, L = seg_stack.shape
            batch["seg"] = seg_stack.reshape(B * V, L).unsqueeze(1).to(self.device)
            for k in list(batch.keys()):
                if k != "seg":
                    v = batch[k]
                    if torch.is_tensor(v):
                        batch[k] = v.repeat_interleave(V, dim=0).to(self.device)
        else:
            batch = _process_item(batch)
        return batch
    
    def train_one_epoch(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        self.model.train()
        self._reset_running_stats()
        train_loop = tqdm(self.train_loader, desc=f"Training Epoch {epoch}")
        for batch in train_loop:
            batch = self.before_batch(batch)
            outputs, _, loss = self._process_one_batch(batch)
            self._update_running_stats(loss, {})
            train_loop.set_postfix(self.get_tqdm_info(outputs, loss))
        avg_loss, avg_metrics = self._get_average_stats()
        return avg_loss, avg_metrics

    def train(
        self,
        num_epochs=10,
        start_epoch: int = 0,
        k: int = 0,
    ) -> None:
        self.train_logger = setup_logger(log_dir=self.checkpoint_dir,sub_path=f"train_info/{k} fold/")
        self.before()
        best_val_loss = float('inf')
        filename = self.model_name+f"_{k}.pth"
        for epoch in range(start_epoch, num_epochs):
            train_loss, train_metrics = self.train_one_epoch(epoch)
            if self.val_loader is None:
                self._log_epoch_stats(epoch, "train", train_loss, train_metrics)
            # 验证
            val_loss, val_metrics = 0.0, {}
            if self.val_loader is not None and (epoch + 1) % self.val_frequency == 0:
                val_loss, val_metrics = self.validate(epoch)
                self._log_epoch_stats(epoch, "train", train_loss, train_metrics)
                log_msg = f"val: Loss: {val_loss:.4f}"
                for name, value in val_metrics.items():
                    log_msg += f" - {name}: {value:.4f}"
                self.train_logger.info(log_msg)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(
                            epoch,
                            filename = filename,
                            val_loss=val_loss,
                            val_metrics=val_metrics,
                            optimizer_state_dict=self.optimizer.state_dict(),
                        )
            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
                self.train_logger.info(
                    f"Current learning rate: {self.scheduler.get_last_lr()[0]})"
                )

    def set_tracker(self, loader,time = 0):
        tracker = PrototypesTracker()
        embeddings_list, labels_list = [], []

        with torch.no_grad():
            for batch in tqdm(loader, desc="set tracker"):
                if batch["seg"].ndim == 3:
                    beats = batch["seg"].shape[-2]
                    batch["seg"] = batch["seg"].flatten(0, 1) 
                    batch["label"] = batch["label"].repeat_interleave(beats)

                outputs, targets = self.model_out(batch)
                embeddings_list.append(outputs["embedding"].cpu())
                labels_list.append(targets["label"].cpu())
        
        all_embeddings = torch.cat(embeddings_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0)

        tracker.update(all_embeddings, all_labels)
        if time != 0:
            self.tracker = self.tracker.distill_update(tracker)
        else:
            self.tracker = tracker
        return {"embeddings": all_embeddings, "labels": all_labels}

    def predict(self, embeddings):
        # 准备模板矩阵
        template_labels = sorted(list(self.tracker.templates.keys()))
        template_matrix = torch.stack(
            [self.tracker.templates[label] for label in template_labels]
        )
        template_matrix = template_matrix.to(embeddings.device)

        similarities = self.distance(embeddings, template_matrix)

        return similarities
    
    def validate(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        if self.val_loader is None:
            self.train_logger.warning("没有提供验证数据集，跳过验证")
            return 0.0, {}
        self.model.eval()
        self._reset_running_stats()

        with torch.no_grad():
            val_loop = tqdm(self.val_loader, desc=f"Validating Epoch {epoch}", leave=False)
            for batch in val_loop:
                outputs, targets = self.model_out(batch)
                loss = self.get_loss(outputs, targets)
                tqdm_info = self.get_tqdm_info(outputs, loss)
                self._update_running_stats(loss, {})
                val_loop.set_postfix(tqdm_info)
        avg_loss, avg_metrics = self._get_average_stats()
        return avg_loss, avg_metrics

    def model_out(self, batch):
        batch = self.before_batch(batch)
        targets = self.get_batch_targets(batch)
        outputs = self.get_outputs(batch)
        return outputs, targets

    def enroll(self, loader=None):
        if loader is None:
            loader = self.train_loader
            if loader is None:
                print("没有提供注册集，跳过注册")
                return {}
        self.model.eval()
        data_dict = self.set_tracker(loader)
        self.enroll_data = data_dict

    def rigistered_test(self, loader=None):
        if loader is None:
            loader = self.test_loader
            if loader is None:
                self.test_logger.warning("没有提供测试集，跳过测试")
                return {}
        self.model.eval()

        # ---- 存储所有已注册数据 ----
        registered_logits_list = []
        registered_labels_list = []
        registered_feat_list = []
        registered_weights_list = []
        registered_session_list = []
        registered_k_index_list = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Registered Test"):
                out_dict = self.adapt_model_out(batch)
                registered_logits_list.append(out_dict["logit"])
                registered_labels_list.append(out_dict["target"])
                registered_feat_list.append(out_dict["embedding"].detach())
                registered_session_list.append(batch["session"])
                registered_k_index_list.append(batch["k_index"])
                
                if 'weight' in out_dict:
                    registered_weights_list.append(out_dict["weight"].detach())
        
        registered_logits_list = torch.cat(registered_logits_list, dim=0)
        registered_labels_list = torch.cat(registered_labels_list, dim=0)
        registered_feat_list = torch.cat(registered_feat_list, dim=0)
        registered_session_list = torch.cat(registered_session_list, dim=0)
        registered_k_index_list = torch.cat(registered_k_index_list, dim=0)
        
        if 'weight' in out_dict:
            registered_weights_list = torch.cat(registered_weights_list, dim=0)
        
        data = {
            "embeddings": registered_feat_list.cpu(),
            "labels": registered_labels_list.cpu(),
            "logits": registered_logits_list.cpu(),
            "session": registered_session_list.cpu(),
            "k_index": registered_k_index_list.cpu(),
        }
        if 'weight' in out_dict:
            data["weights"] = registered_weights_list.cpu()
        self.registeded_data = data

    def register_metrics(self,test_info = None,k = None):
        # 统计平均
        # 类别准确率（你原本的函数）
        test_logger = setup_logger(log_dir=self.checkpoint_dir,sub_path=f"test_info/", log_filename="test_logs.log")
        register_weight_list = self.registeded_data.get("weights", None)
        if 'weights' in self.registeded_data:
            for i in range(2):
                col = register_weight_list[:, i]
                print(f"weight[{i}]")
                print("mean   :", col.mean().item())
                print("std    :", col.std().item())
                print("min    :", col.min().item())
                print("25%    :", col.quantile(0.25).item())
                print("median :", col.median().item())
                print("75%    :", col.quantile(0.75).item())
                print("max    :", col.max().item())
                print("nonzero ratio:", (col > 0).float().mean().item())
        # 将标签重新映射为 0 ~ n-1 的连续整数
        enroll_label_list = self.enroll_data["labels"]
        unique_labels = torch.unique(enroll_label_list)
        old2new_map  = {old.item(): new for new, old in enumerate(unique_labels)}
        new2old_map  = {new: old for old, new in old2new_map.items()}
        enroll_label_list = torch.tensor([old2new_map[l.item()] for l in enroll_label_list], dtype=torch.long)
        register_label_list = torch.tensor([old2new_map[l.item()] for l in self.registeded_data["labels"]],dtype=torch.long,)
        register_session_list = self.registeded_data["session"]
        register_k_index_list = self.registeded_data["k_index"]
        register_logit_list = self.registeded_data["logits"]
        
        detail_logger = setup_logger(log_dir=self.checkpoint_dir,sub_path=f"detail_info/{test_info}/folder_{k}/", log_filename="detail_logs.log"
                                   ,overwrite_log=True,log_to_console=False)
        save2npy(register_label_list,os.path.join(self.checkpoint_dir, "detail_info",test_info,f'folder_{k}','register_label.npy'))
        save2npy(register_session_list,os.path.join(self.checkpoint_dir, "detail_info",test_info,f'folder_{k}','register_session.npy'))
        save2npy(self.registeded_data['embeddings'],os.path.join(self.checkpoint_dir, "detail_info",test_info,f'folder_{k}','register_embeddings.npy'))
        if 'weights' in self.registeded_data:
            save2npy(self.registeded_data['weights'],os.path.join(self.checkpoint_dir, "detail_info",test_info,f'folder_{k}','register_weights.npy'))
        detail_logger.info(self.format_register_output(register_logit_list=register_logit_list, register_label_list=register_label_list,
                                                       register_session_list=register_session_list,
                                                     register_k_index_list=register_k_index_list,new2old_map=new2old_map))
        # 合并 closed-set 指标
        macro_metrics = subject_accuracy(register_logit_list, register_label_list)
        closed_metrics = {
            **self.compute_metrics(register_logit_list, register_label_list),
            **macro_metrics,
        }
        # 1:1验证
        verify_metrics = eval_allpairs_eer(
            sim_matrix=register_logit_list, labels=register_label_list,save_folder= os.path.join(self.checkpoint_dir, "plt",test_info,f'folder_{k}')
        )
        all_metrics = {**verify_metrics, **closed_metrics}
        keys_to_extract = ['top1', 'top3', 'top5','precision','recall','f1', 'verify_eer']
        all_metrics = {k: v for k, v in all_metrics.items() if k in keys_to_extract}
        value_list = []
        for k in keys_to_extract:
            v = all_metrics.get(k)
            if isinstance(v, (float, int)):
                value_list.append(f"{v:.4f}")
        test_logger.info(",".join(value_list))
        
        subject_close_pd = pd.DataFrame(macro_metrics["class_accuracy"])
        subject_close_pd['Subject_ID'] = subject_close_pd['Subject_ID'].map(new2old_map)
        result_clean =subject_close_pd.round(4)
        person_info = result_clean.to_csv(index=False, sep=',', quoting=csv.QUOTE_NONE, escapechar=" ")
        subject_logger = setup_logger(log_dir=self.checkpoint_dir, sub_path=f"subject_info/{test_info}/folder_{k}/", log_filename="subject_logs.log")
        subject_logger.info(person_info)

    def adapt_model_out(self, batch):
        batch = self.before_batch(batch)
        targets = batch["label"].view(-1).to(self.device)
        seg = batch["seg"]
        weights = None
        if seg.ndim > 3:
            B, C, num, D = seg.shape
            lengths = batch["lengths"]
            seg_reshaped = seg.transpose(1, 2).reshape(B * num, C, D)
            outputs = self.get_outputs({"seg": seg_reshaped})
            embedding_dim = outputs["embedding"].shape[-1]
            embeddings = outputs["embedding"].view(B, num, embedding_dim)
            mask = (torch.arange(num, device=self.device)[None, :] < lengths[:, None]).unsqueeze(-1)
            embedding = (embeddings * mask).sum(dim=1) / (lengths.unsqueeze(-1) + 1e-8)
            if 'weight' in outputs:
                mask = mask.unsqueeze(-1)
                weights = outputs["weight"].view(B, num, 2,-1)
                weights = (weights * mask).sum(dim=1) / (lengths.unsqueeze(-1).unsqueeze(-1) + 1e-8)
            
            
        else:
            outputs = self.get_outputs(batch)
            embedding = outputs["embedding"]
            if 'weight' in outputs:
                weights = outputs["weight"]

        logits = self.predict(embedding)
        data_dict = {"logit": logits, "target": targets, "embedding": embedding}
        if 'weight' in outputs:
            data_dict["weight"] = weights
        return data_dict

    def compute_metrics(self, logits, targets):
        metrics_results = {}
        preds = torch.argmax(logits, dim=-1)
        for name, metric_func in self.metrics.items():
            metrics_results[name] = metric_func(
                logits=logits, preds=preds, targets=targets
            ).item()
        return metrics_results
    
    def format_register_output(
        self,
        register_label_list,
        register_session_list,
        register_k_index_list,
        register_logit_list,
        new2old_map,
        topk=5,
    ):
        labels = register_label_list.cpu()
        logits = register_logit_list.cpu()

        sessions = (
            register_session_list.cpu().tolist()
            if isinstance(register_session_list, torch.Tensor)
            else list(register_session_list)
        )
        k_indices = (
            register_k_index_list.cpu().tolist()
            if isinstance(register_k_index_list, torch.Tensor)
            else list(register_k_index_list)
        )
        
        n = labels.shape[0]
        assert len(sessions) == n, "session 长度不一致"
        assert len(k_indices) == n, "k_index 长度不一致"
        assert logits.shape[0] == n, "logits.shape[0] 不一致"

        topk_vals, topk_idx = torch.topk(logits, k=topk, dim=1)

        records = []
        for i in range(n):
            label = new2old_map[labels[i].item()]
            session = sessions[i]
            k_index = k_indices[i]
            top5_list = [new2old_map[x] for x in topk_idx[i].tolist()]
            pred_top1 = top5_list[0]
            is_correct = (pred_top1 == label)

            records.append({
                "label": label,
                "session": session,
                "k_index": k_index,
                "is_correct": is_correct,
                "top5": top5_list,
                "logit": logits[i].tolist()
            })

        records.sort(key=lambda x: (x["label"], not x["is_correct"],x['session'],x['k_index']))

        header = (
            f"{'label':<8} "
            f"{'session':<12} "
            f"{'k_index':<10} "
            f"{'correct':<10} "
            f"{'top5':<30} "
            f"{'logit':<30} "
        )
        lines = [header, "-" * len(header)]

        for r in records:
            lines.append(
                f"{str(r['label']):<8} "
                f"{str(r['session']):<12} "
                f"{str(r['k_index']):<10} "
                f"{str(r['is_correct']):<10} "
                f"{str(r['top5']):<30} "
                f"{str(r['logit'])} "
            )

        return "\n".join(lines)


    