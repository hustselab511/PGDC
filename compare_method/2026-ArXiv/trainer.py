from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from utils import get_metrics

class Trainer:
    def __init__(
        self,
        model,
        train_loader = None,
        val_loader=None,
        model_name="HPAFModel",
        checkpoint_dir="metrics",
        device="cuda:0",
        miner=None,
        loss_func=None,
        num_prototypes=3,
    ):
        self.model = model.to(device)
        self.model_name = model_name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
        self.miner = miner
        self.loss_func = loss_func
        self.num_prototypes = num_prototypes
        self.best_val_metric = -1e18
        self.gallery = None
        self.test_result = None

    # =========================================================
    # 基础工具
    # =========================================================
    def before_batch(self, batch):
        def move_to_device(x):
            if torch.is_tensor(x):
                return x.to(self.device, non_blocking=True)
            if isinstance(x, dict):
                return {k: move_to_device(v) for k, v in x.items()}
            if isinstance(x, list):
                return [move_to_device(v) for v in x]
            if isinstance(x, tuple):
                return tuple(move_to_device(v) for v in x)
            return x

        return move_to_device(batch)

    def get_outputs(self, batch):
        return self.model(batch)

    def compute_loss(self, outputs, batch):
        if self.loss_func is None:
            raise ValueError("loss_func 不能为空")

        embeddings = outputs["embedding"]
        labels = batch["label"]

        if self.miner is not None:
            mined = self.miner(embeddings, labels)
            loss = self.loss_func(embeddings, labels, mined)
        else:
            loss = self.loss_func(embeddings, labels)

        return loss

    @staticmethod
    def cosine_distance(a, b, eps=1e-8):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        if a.ndim == 1:
            a = a[None, :]
        a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + eps)
        b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + eps)

        dist = 1.0 - np.matmul(a, b.T)
        return dist[0] if dist.shape[0] == 1 else dist

    # =========================================================
    # train / val
    # =========================================================
    def train_one_epoch(self, epoch,loader):
        self.model.train()

        total_loss = 0.0
        total_num = 0

        pbar = tqdm(loader, desc=f"Training Epoch {epoch}", leave=True)
        for batch in pbar:
            batch = self.before_batch(batch)

            self.optimizer.zero_grad()
            outputs = self.get_outputs(batch)
            loss = self.compute_loss(outputs, batch)
            loss.backward()
            self.optimizer.step()

            bs = batch["label"].shape[0]
            total_loss += loss.item() * bs
            total_num += bs

            pbar.set_postfix(loss=f"{total_loss / max(total_num, 1):.4f}")

        if self.scheduler is not None:
            self.scheduler.step()
        
        return total_loss / max(total_num, 1)

    @torch.no_grad()
    def validate(self, epoch=0,loader=None):
        if loader is None:
            return {}

        self.model.eval()

        total_loss = 0.0
        total_num = 0
        pbar = tqdm(loader, desc=f"Validate Epoch {epoch}", leave=True)
        for batch in pbar:
            batch = self.before_batch(batch)
            outputs = self.get_outputs(batch)
            loss = self.compute_loss(outputs, batch)

            bs = batch["label"].shape[0]
            total_loss += loss.item() * bs
            total_num += bs

        val_loss = total_loss / max(total_num, 1)
        return {"val_loss": val_loss,}

    def train(self,train_loader,val_loader=None, num_epochs=10,save_path=None):
        for epoch in range(num_epochs):
            train_loss = self.train_one_epoch(epoch,train_loader)
            val_metrics = self.validate(epoch,val_loader)

            msg = f"[Epoch {epoch + 1}/{num_epochs}] train_loss={train_loss:.4f}"
            if len(val_metrics) > 0:
                msg += (
                    f", val_loss={val_metrics['val_loss']:.4f}"
                )
            print(msg)
            if save_path is not None:
                torch.save(self.model.state_dict(),save_path)
                print(f"model saved to {save_path}")

    @torch.no_grad()
    def extract_embeddings(self, loader, desc="Extract"):
        self.model.eval()

        all_embeddings = []
        all_labels = []

        for batch in tqdm(loader, desc=desc, leave=True):
            batch = self.before_batch(batch)
            outputs = self.get_outputs(batch)
            all_embeddings.append(outputs["embedding"].detach().cpu())
            all_labels.append(batch["label"].detach().cpu())

        all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        return all_embeddings, all_labels

    def build_gallery_prototypes(self, embeddings, labels, k, normalize=True):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        labels = np.asarray(labels)

        proto_list = []
        proto_label_list = []

        for sid in np.unique(labels):
            feats = embeddings[labels == sid]

            if len(feats) < k:
                continue

            group = feats[:k]
            proto = group.mean(axis=0)

            if normalize:
                proto = proto / (np.linalg.norm(proto) + 1e-12)

            proto_list.append(proto.astype(np.float32))
            proto_label_list.append(int(sid))

        if len(proto_list) == 0:
            raise ValueError(f"No gallery prototypes can be built with k={k}")

        proto_matrix = np.stack(proto_list, axis=0)   # (num_class, D)
        proto_labels = np.asarray(proto_label_list)

        return proto_matrix, proto_labels

    def enroll(self, loader=None):
        embeddings, labels = self.extract_embeddings(loader, desc="Enroll")
        proto_matrix, proto_labels = self.build_gallery_prototypes(
            embeddings, labels, k=self.num_prototypes, normalize=True
        )

        self.gallery = {
            "proto_matrix": proto_matrix,
            "proto_labels": proto_labels,
        }

        print(f"Enroll done: {len(proto_labels)} identities, k={self.num_prototypes}")

    def build_query_prototypes_sliding(self, embeddings, labels, k, normalize=True):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        labels = np.asarray(labels)

        proto_list = []
        proto_label_list = []

        for sid in np.unique(labels):
            feats = embeddings[labels == sid]
            n = len(feats)

            if n < k:
                continue

            # 滑动窗口: 窗口大小 k, 步长 1
            for start in range(0, n - k + 1):
                group = feats[start:start + k]
                proto = group.mean(axis=0)
                if normalize:
                    proto = proto / (np.linalg.norm(proto) + 1e-12)
                proto_list.append(proto.astype(np.float32))
                proto_label_list.append(int(sid))

        if len(proto_list) == 0:
            raise ValueError(f"No query prototypes can be built with k={k}")

        proto_matrix = np.stack(proto_list, axis=0)   # (N_query, D)
        proto_labels = np.asarray(proto_label_list)

        return proto_matrix, proto_labels

    @torch.no_grad()
    def test(self, loader, test_info="", dataset_name="", k=None):
        assert self.gallery is not None, "请先调用 enroll()"
        if k is None:
            k = self.num_prototypes

        query_embeddings, query_labels = self.extract_embeddings(loader, desc="Registered Test")

        # 测试端：滑动窗口，窗口大小 k，步长 1
        query_embeddings, query_labels = self.build_query_prototypes_sliding(
            query_embeddings, query_labels, k=k, normalize=True
        )

        proto_matrix = self.gallery["proto_matrix"]   # (M, D)
        proto_labels = self.gallery["proto_labels"]   # (M,)

        class_ids = np.unique(proto_labels)
        num_class = len(class_ids)
        class_to_col = {int(cid): i for i, cid in enumerate(class_ids)}
        logit_rows = []

        for emb in query_embeddings:
            dists = self.cosine_distance(emb, proto_matrix)   # (M,)
            proto_scores = -dists

            class_scores = np.full(num_class, -1e9, dtype=np.float32)

            for cid in class_ids:
                mask = (proto_labels == cid)
                class_scores[class_to_col[int(cid)]] = proto_scores[mask].max()
            logit_rows.append(class_scores)

        logit_list = torch.tensor(np.stack(logit_rows, axis=0), dtype=torch.float32)
        label_list = torch.tensor(query_labels, dtype=torch.long)

        query_labels = np.asarray(query_labels)
        save_folder = f"metrics/{test_info}/{dataset_name}/"
        get_metrics(logit_list,label_list,test_info=test_info,save_folder=save_folder,)
