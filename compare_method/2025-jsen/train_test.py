import torch
import numpy as np
import os
import sys

from config import SETUP_SEED,TRAIN_FOLD,TRAIN_DATASET_LIST
from utils.dataset import get_meta,pretrain_dataset
from utils.dataset.segment_dataset import find_indices,find_worst_st_dict
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import copy

from utils import get_metrics
from PreMDataset import PreMDataset
from model import CESTNet,CESTNetConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
root_path = os.path.dirname(os.path.abspath(__file__))
static_path = os.path.join(root_path, "static")
os.makedirs(static_path, exist_ok=True)

def evaluate(model, data_loader, device='cuda:0'):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for raw_signals, spec_signals, labels in data_loader:
            raw_signals = raw_signals.to(device, non_blocking=True)
            spec_signals = spec_signals.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(raw_signals, spec_signals)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100.0 * correct / total if total > 0 else 0.0
    return acc


def train_model(
    model,
    train_loader,
    val_loader=None,
    device='cuda:0',
    num_epochs=200,
    lr=1e-4,
    weight_decay=1e-4,
    save_path=None,
    patience=20,
    use_amp=True
):
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )

    amp_enabled = use_amp and ('cuda' in str(device))
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

    best_val_acc = 0.0
    best_epoch = 0
    early_stop_counter = 0

    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for raw_signals, spec_signals, labels in train_loader:
            raw_signals = raw_signals.to(device, non_blocking=True)
            spec_signals = spec_signals.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=amp_enabled):
                outputs = model(raw_signals, spec_signals)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * labels.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total if total > 0 else 0.0
        train_acc = 100.0 * correct / total if total > 0 else 0.0
        current_lr = optimizer.param_groups[0]['lr']

        if val_loader is not None:
            val_acc = evaluate(model, val_loader, device)
            scheduler.step(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                if save_path is not None:
                    torch.save(model.state_dict(), save_path)
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            print(
                f"Epoch [{epoch+1:03d}/{num_epochs:03d}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.2f}% | "
                f"Val Acc: {val_acc:.2f}% | "
                f"LR: {current_lr:.6e}"
            )

            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best Val Acc: {best_val_acc:.2f}% (Epoch {best_epoch})")
                break
        else:
            print(
                f"Epoch [{epoch+1:03d}/{num_epochs:03d}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.2f}% | "
                f"LR: {current_lr:.6e}"
            )

    return model

def evaluate_final_model(model, loader, device, dataset_name=None, k=None, test_info='2025-jsen'):
    model.eval()
    logit_list = []
    label_list = []

    with torch.no_grad():
        for raw_signals, spec_signals, labels in loader:
            raw_signals = raw_signals.to(device, non_blocking=True)
            spec_signals = spec_signals.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(raw_signals, spec_signals)
            probs = torch.softmax(outputs, dim=1)   # 如果 get_metrics 需要概率，就保留

            logit_list.append(probs.cpu())
            label_list.append(labels.cpu())

    logit_list = torch.cat(logit_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    get_metrics(
        logit_list,
        label_list,
        test_info=test_info,
        save_folder=f'metrics/{test_info}/{dataset_name}/{k}'
    )

if __name__ == "__main__":
    SETUP_SEED(seed = 42)
    batch_size = 128
    device = 'cuda:0'
    info = "2025-jsen"
    # TRAIN_FOLD = {'AF':{'k':[None]},'PAC':{'k':[None]},'PVC':{'k':[None]}}
    dataset_list = ['SRRSH']
    TRAIN_FOLD = {dataset_name:TRAIN_FOLD[dataset_name] for dataset_name in dataset_list}
    path = f"metrics/{info}/CESTNet_pretrian.pth"
    for dataset_name,fold_dict in TRAIN_FOLD.items():
        for k in fold_dict['k']:
            #预训练
            # dataset_dict = pretrain_dataset(TRAIN_DATASET_LIST,k = 0,process_des=info)
            # X_train, x_val, y_train, y_val,*_ = train_test_split(dataset_dict["all_data"]['X'], dataset_dict["all_data"]['y'], test_size=0.2, random_state=42, stratify=dataset_dict["all_data"]['y'])
            # pre_train_dataset = PreMDataset(X_train,y_train,mmap_path = f'pre_train.mmap')
            # pre_val_dataset = PreMDataset(x_val,y_val,mmap_path = f'pre_val.mmap')
            # train_loader = DataLoader(pre_train_dataset, batch_size=batch_size, shuffle=True,pin_memory=True)
            # val_loader = DataLoader(pre_val_dataset, batch_size=batch_size, shuffle=False,pin_memory=True)
            # cfg = CESTNetConfig(num_classes=1268)
            # model = CESTNet(cfg)
            # train_model(model, train_loader, val_loader, device=device, num_epochs=30,save_path=path)
            #加载模型权重
            cfg = CESTNetConfig(num_classes=1268)
            model = CESTNet(cfg)
            model.load_state_dict(torch.load(path,weights_only=True))
            #训练集
            first_segments, first_labels = get_meta(dataset_name=dataset_name,k = k,session=0,process_des=info)
            X_train,x_val,y_train,y_val = train_test_split(first_segments,first_labels, test_size=0.2, random_state=42,stratify=first_labels)
            train_dataset = PreMDataset(X_train,y_train,mmap_path = f'{dataset_name}_train.mmap')
            val_dataset = PreMDataset(x_val,y_val,mmap_path = f'{dataset_name}_val.mmap')
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,pin_memory=True)
            # 微调
            num_classes = train_dataset.class_num
            model.modify_head(num_classes=num_classes)
            train_model(model, train_loader, val_loader, device=device, num_epochs=30)
            # 端到端训练
            cfg = CESTNetConfig(num_classes=train_dataset.class_num)
            model = CESTNet(cfg)
            train_model(model, train_loader, val_loader, device=device, num_epochs=50)
            # 测试
            second_segments, second_labels = get_meta(dataset_name=dataset_name, k=k,session=1,process_des=info)
            # 形态学测试
            for mode in ['Pearson','MAE',]:
                for selection in ['top','middle','bottom']:
                    test_indices = find_indices(X_train,y_train,second_segments,second_labels,mode=mode,ratio=0.25,selection=selection)
                    new_second_segments = second_segments[test_indices]
                    new_second_labels = second_labels[test_indices]
                    test_dataset = PreMDataset(new_second_segments,new_second_labels,mmap_path = f'{dataset_name}_test_{mode}_{selection}.mmap')
                    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,pin_memory=True)
                    evaluate_final_model(model, test_loader, device=device,dataset_name = dataset_name,k = k, test_info = info)
            # ST-T最差的P-QRS分类
            for mode in ['Pearson','MAE',]:
                pqrs_dict = find_worst_st_dict(X_train,y_train,second_segments,second_labels,mode=mode,info=info)
                for selection,data_dict in pqrs_dict.items():
                    new_second_segments = data_dict['X']
                    new_second_labels = data_dict['y']
                    test_dataset = PreMDataset(new_second_segments,new_second_labels,mmap_path = f'{dataset_name}_test_worst_st_{mode}_{selection}.mmap')
                    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,pin_memory=True)
                    test_acc = evaluate_final_model(model, test_loader, device=device,test_info = info,k=k,dataset_name=dataset_name)
            test_dataset = PreMDataset(second_segments,second_labels,mmap_path = f'{dataset_name}_test.mmap')
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,pin_memory=True)
            evaluate_final_model(model, test_loader, device=device,dataset_name = dataset_name,k = k, test_info = info)
            