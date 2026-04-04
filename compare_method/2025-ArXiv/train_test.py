import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import math
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import SETUP_SEED, TRAIN_FOLD,TRAIN_DATASET_LIST
from TimeDataset import TimeDatasetBuilder, TimeDataset,get_meta,pretrain_dataset
from model import TransECG
from model.ViT import ViTClassify
from utils.util.metrics import get_metrics

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, train_loader, val_loader, lr_init=1e-4, device='cuda:0', num_epochs=45,save_path=None):
    optimizer = optim.AdamW(model.parameters(), lr=lr_init)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        val_acc,val_loss  = evaluate(model, val_loader, device,criterion=criterion)
        if save_path is not None:
            torch.save(model.state_dict(), save_path)
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Loss: {running_loss/len(train_loader):.4f}, "
              f"Val Acc: {val_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, "
              f"ValLR: {current_lr:.6f}")

def evaluate(model, loader, device,criterion,dataset_name=None,test_info = '2025-ArXiv',k=None,is_out = False):
    model.eval()
    logits_list = []
    label_list = []
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            logits_list.append(logits)
            label_list.append(labels)
    acc = 100 * correct / total        
    if is_out:
        logits_list = torch.cat(logits_list, dim=0).cpu()
        label_list = torch.cat(label_list, dim=0).cpu()
        get_metrics(logits_list,label_list,test_info = test_info,save_folder=f"metrics/{test_info}/{dataset_name}/{k}/")
    return acc,loss.item()
SETUP_SEED(seed=42)
info = "2025-ArXiv"
train_config = {
    "batch_size": 256,
    "device": "cuda:0",
    "info": info,
    "num_epochs": 100,
    "isPretrain": False,
    "save_path": f'metrics/{info}/',
}
path = train_config["save_path"]+ViTClassify.__name__+'_pretrain.pth'
os.makedirs(os.path.dirname(path), exist_ok=True)
dataset_list = ['CYBHi','heartprint']
TRAIN_FOLD = {dataset_name:TRAIN_FOLD[dataset_name] for dataset_name in dataset_list}
pre_train_num_classes = 1315
dataloader_func = lambda dataset,shuffle=False: DataLoader(dataset, batch_size=train_config["batch_size"], shuffle=shuffle, pin_memory=True, num_workers=4)
if __name__ == "__main__":
    for dataset_name,fold_dict in TRAIN_FOLD.items():
        for k in fold_dict['k']:
            # 预训练
            # dataset_dict = pretrain_dataset(TRAIN_DATASET_LIST,k = k,process_des=info)
            # pretrain_segments, pretrain_labels = dataset_dict["all_data"]['X'], dataset_dict["all_data"]['y']
            # X_pretrain,X_preval,y_pretrain,y_preval = train_test_split(pretrain_segments,pretrain_labels, test_size=0.2, random_state=42,stratify=pretrain_labels)
            # pre_train_dataset = TimeDataset(X_pretrain,y_pretrain)
            # pre_train_loader = dataloader_func(pre_train_dataset,shuffle=True)   
            # pre_val_dataset = TimeDataset(X_preval,y_preval)
            # pre_val_loader = dataloader_func(pre_val_dataset,shuffle=False)
            # model = ViTClassify(num_classes=pre_train_num_classes, seq_len=1000, patch_size=10,hidden_size=256,depth=6, heads=8)
            # train_model(model, pre_train_loader, pre_val_loader, device=train_config["device"], num_epochs=train_config["num_epochs"],
            #             save_path=path)
            # 加载预训练权重
            model = ViTClassify(num_classes=pre_train_num_classes, seq_len=1000, patch_size=10,hidden_size=256,depth=6, heads=8)
            model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
            # 训练集
            first_segments, first_labels = get_meta(dataset_name=dataset_name, k=k,session=0,process_des=info)
            X_train,x_val,y_train,y_val = train_test_split(first_segments,first_labels, test_size=0.2, random_state=42,stratify=first_labels)
            train_dataset = TimeDataset(X_train,y_train)
            val_dataset = TimeDataset(x_val,y_val)
            train_loader = dataloader_func(train_dataset,shuffle=True)
            val_loader = dataloader_func(val_dataset,shuffle=False)
            #微调
            model.modify_num_classes(train_dataset.num_classes)
            train_model(model, train_loader, val_loader, device=train_config["device"],lr_init=1e-3, num_epochs=train_config["num_epochs"],)
            #端到端训练
            # model = ViTClassify(num_classes=train_dataset.num_classes, seq_len=1000, patch_size=10,hidden_size=256,depth=6, heads=8)
            # train_model(model, train_loader, val_loader, device=train_config["device"], num_epochs=train_config["num_epochs"],)
            #测试
            second_segments, second_labels = get_meta(dataset_name=dataset_name, k=k,session=1,process_des=info)
            test_dataset = TimeDataset(second_segments,second_labels)
            test_loader = dataloader_func(test_dataset,shuffle=False)
            acc,loss = evaluate(model, test_loader, device=train_config["device"],criterion=nn.CrossEntropyLoss(),dataset_name=dataset_name,k=k,test_info=info,is_out=True)
            print(f"Test Acc: {acc:.2f}%, "
                  f"Test Loss: {loss:.4f}, ")
