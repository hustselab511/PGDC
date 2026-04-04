import torch
import numpy as np
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from config import SETUP_SEED,TRAIN_FOLD
from utils.dataset.meta import get_meta
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from model import EDITHIdentification, EDITHSiamese
from utils.dataset.segment_dataset import find_indices,find_worst_st_dict
from utils.util.metrics import get_metrics
from Dataset import ECGHeartbeatDataset, SiameseFeatureDataset, extract_embeddings, build_siamese_dataset
from collections import defaultdict, deque



root_path = os.path.dirname(os.path.abspath(__file__))
static_path = os.path.join(root_path, "static")
os.makedirs(static_path, exist_ok=True)

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    model.to(device)
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    return acc

def evaluate_final_model_fused(model, loader, device='cuda:0', num_beats=5):
    model.eval()
    model.to(device)

    logit_list = []
    label_list = []

    # 每个 id 一个滑动窗口
    id_to_logits = defaultdict(lambda: deque(maxlen=num_beats))

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            logits = model(inputs)   # (B, C)
            logits = logits.detach().cpu()
            labels = labels.detach().cpu()

            for i in range(logits.size(0)):
                sid = labels[i].item()
                id_to_logits[sid].append(logits[i])

                # 该 id 的窗口满了：做一次融合
                if len(id_to_logits[sid]) == num_beats:
                    window_logits = torch.stack(list(id_to_logits[sid]), dim=0)   # (num_beats, C)
                    fused_logit = window_logits.mean(dim=0)
                    logit_list.append(fused_logit)
                    label_list.append(sid)
                    id_to_logits[sid].popleft()

    if not logit_list:
        return torch.tensor([]), torch.tensor([])

    logit_tensor = torch.stack(logit_list, dim=0)                 # (N, C)
    label_tensor = torch.tensor(label_list, dtype=torch.long)     # (N,)

    print(label_tensor.shape, logit_tensor.shape)
    return logit_tensor, label_tensor

def train_identification_model(model, train_loader, val_loader, num_epochs = 500, device='cuda:0'):
    """
    第一阶段：训练闭集识别模型，加入学习率调度
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
    criterion = torch.nn.CrossEntropyLoss() 
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    for _ in range(num_epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                outputs = model(x_val)
                _, predicted = torch.max(outputs.data, 1)
                total += y_val.size(0)
                correct += (predicted == y_val).sum().item()
    return model

def train_verification_model(model, train_loader, num_epochs = 75, device='cuda:0'):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for u, v, label in train_loader:
            u, v, label = u.to(device), v.to(device), label.to(device).unsqueeze(1)
            optimizer.zero_grad()
            
            score = model(u, v) 
            loss = criterion(score, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] | Train MSE: {train_loss/len(train_loader):.4f}")
    return model

def evaluate_1to1_verification(
    base_model,
    siamese_net,
    enroll_loader,
    test_loader,
    device='cuda:0',
    num_beats=5,
    step=1
):
    base_model.eval()
    siamese_net.eval()
    base_model.to(device)
    siamese_net.to(device)

    # =========================================================
    # 1. 提取注册集 embedding，并按用户收集
    # =========================================================
    enroll_dict = defaultdict(list)

    with torch.no_grad():
        for x, y in enroll_loader:
            x = x.to(device)
            emb = base_model.features(x).detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            for i in range(len(y)):
                enroll_dict[int(y[i])].append(emb[i])

    for user in enroll_dict:
        enroll_dict[user] = np.asarray(enroll_dict[user], dtype=np.float32)

    # =========================================================
    # 2. 每个用户只取第一个模板
    # =========================================================
    templates = {}   # user -> Tensor(D,)
    for user, user_embs in enroll_dict.items():
        if len(user_embs) < num_beats:
            continue

        fused_template = user_embs[:num_beats].mean(axis=0).astype(np.float32)
        templates[user] = torch.from_numpy(fused_template).to(device)

    if len(templates) == 0:
        return [], []

    valid_users = sorted(list(templates.keys()))

    # =========================================================
    # 3. 提取测试集 embedding，并按用户收集
    # =========================================================
    test_dict = defaultdict(list)

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            emb = base_model.features(x).detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            for i in range(len(y)):
                uid = int(y[i])
                if uid in templates:
                    test_dict[uid].append(emb[i])

    for user in test_dict:
        test_dict[user] = np.asarray(test_dict[user], dtype=np.float32)

    # =========================================================
    # 4. 测试端按多心拍滑窗形成融合测试样本
    # =========================================================
    evaluation_samples = []   # [(Tensor(D,), true_user), ...]

    for user, user_embs in test_dict.items():
        if len(user_embs) < num_beats:
            continue

        for start in range(0, len(user_embs) - num_beats + 1, step):
            fused_eval = user_embs[start:start + num_beats].mean(axis=0).astype(np.float32)
            evaluation_samples.append((torch.from_numpy(fused_eval).to(device), user))

    if len(evaluation_samples) == 0:
        return [], []

    # =========================================================
    # 5. 每个测试融合样本与所有注册模板做 1:1 打分
    # =========================================================
    y_true = []
    scores = []

    with torch.no_grad():
        for eval_emb, true_user in evaluation_samples:
            for template_user in valid_users:
                s = siamese_net(
                    eval_emb.unsqueeze(0),
                    templates[template_user].unsqueeze(0)
                )

                if s.numel() != 1:
                    raise ValueError(f"Unexpected siamese_net output shape: {s.shape}")

                scores.append(s.item())
                y_true.append(1 if true_user == template_user else 0)

    return y_true, scores
if __name__ == "__main__":
    SETUP_SEED(seed = 42)
    batch_size = 128
    first_epochs = 120
    second_epochs = 75
    device = 'cuda:0'
    info = "2022-TETCI"
    # TRAIN_FOLD = {'AF':{'k':[None]},'PAC':{'k':[None]},'PVC':{'k':[None]}}
    dataset_name = "SRRSH"
    TRAIN_FOLD = {dataset_name:TRAIN_FOLD[dataset_name]}
    for dataset_name,fold_dict in TRAIN_FOLD.items():
        for k in fold_dict['k']:
            # 训练集
            first_segments, first_labels = get_meta(dataset_name=dataset_name, k=k,session=0,process_des=info)
            X_train,x_val,y_train,y_val = train_test_split(first_segments,first_labels, test_size=0.2, random_state=42,stratify=first_labels)
            train_dataset = ECGHeartbeatDataset(X_train,y_train)
            val_dataset = ECGHeartbeatDataset(x_val,y_val)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,pin_memory=True,num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,pin_memory=True,num_workers=4)
            n_classes = len(np.unique(first_labels))
            
            base_model = EDITHIdentification(num_classes=n_classes)
            # 训练模型
            train_identification_model(base_model, train_loader, val_loader, num_epochs=first_epochs, device=device)
            train_embeddings, train_labels = extract_embeddings(base_model, train_loader, device=device)
            train_embeddings, train_labels = build_siamese_dataset(train_embeddings, train_labels)
            train_siamese_dataset = SiameseFeatureDataset(train_embeddings, train_labels)
            train_siamese_loader = DataLoader(train_siamese_dataset, batch_size=512, shuffle=True,pin_memory=True)
            siamse_model = EDITHSiamese()
            train_verification_model(siamse_model, train_siamese_loader, num_epochs=second_epochs, device=device)
            # 测试
            second_segments, second_labels = get_meta(dataset_name=dataset_name, k=k,session=1,process_des=info)
            enroll_loader = train_loader
            for mode in ['Pearson','MAE',]:
                for selection in ['top','middle','bottom']:
                        test_indices = find_indices(X_train,y_train,second_segments,second_labels,mode=mode,ratio=0.25,selection=selection)
                        new_second_segments = second_segments[test_indices]
                        new_second_labels = second_labels[test_indices]
                        test_dataset = ECGHeartbeatDataset(new_second_segments,new_second_labels)
                        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,pin_memory=True)
                        test_logits, test_labels = evaluate_final_model_fused(base_model, test_loader, device=device)
                        y_true, scores = evaluate_1to1_verification(base_model, siamse_model,enroll_loader, test_loader, device=device)
                        get_metrics(test_logits, test_labels, y_true=y_true, y_scores=scores, test_info=info,save_folder=f"metrics/{info}/{dataset_name}/{k}/")
            # ST-T最差的P-QRS分类
            for mode in ['Pearson','MAE',]:
                pqrs_dict = find_worst_st_dict(X_train,y_train,second_segments,second_labels,mode=mode,info=info)
                for selection,data_dict in pqrs_dict.items():
                    new_second_segments = data_dict['X']
                    new_second_labels = data_dict['y']
                    test_dataset = ECGHeartbeatDataset(new_second_segments,new_second_labels)
                    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,pin_memory=True)
                    test_logits, test_labels = evaluate_final_model_fused(base_model, test_loader, device=device)
                    y_true, scores = evaluate_1to1_verification(base_model, siamse_model,enroll_loader, test_loader, device=device)
                    get_metrics(test_logits, test_labels, y_true=y_true, y_scores=scores, test_info=info,save_folder=f"metrics/{info}/{dataset_name}/{k}/")
            test_dataset = ECGHeartbeatDataset(second_segments,second_labels)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,pin_memory=True)
            test_logits, test_labels = evaluate_final_model_fused(base_model, test_loader, device=device)
            y_true, scores = evaluate_1to1_verification(base_model, siamse_model,enroll_loader, test_loader, device=device)
            get_metrics(test_logits, test_labels, y_true=y_true, y_scores=scores, test_info=info,save_folder=f"metrics/{info}/{dataset_name}/{k}/")
