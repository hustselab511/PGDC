import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import math
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import SETUP_SEED, TRAIN_FOLD,TRAIN_DATASET_LIST
from utils.dataset.segment_dataset import pretrain_dataset,  find_worst_st_dict
from MutiDataset import ECGDataset
from utils.dataset.meta import get_meta
from model import EfficientNet1D, IntegratedEfficientNet1D
from utils.util.metrics import get_metrics

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

class DIEN_Scheduler:
    def __init__(self, optimizer, lr_init=0.01, lrf=0.01, total_epochs=100):
        self.optimizer = optimizer
        self.lr_init = lr_init
        self.lrf = lrf
        self.total_epochs = total_epochs

    def step(self, epoch):
        cos_val = math.cos(math.pi * epoch / self.total_epochs)
        lr = ((1 + cos_val) / 2 * (1 - self.lrf) + self.lrf) * self.lr_init
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
def train_dien_model(model, train_loader, val_loader,lr_init=0.01, device='cuda:0', num_epochs=100,save_path=None):
    
    # 1. 配置优化器：SGD，动量 0.9 
    optimizer = optim.SGD(model.parameters(), lr=lr_init, momentum=0.9)
    scheduler = DIEN_Scheduler(optimizer, lr_init=lr_init, lrf=0.01, total_epochs=num_epochs)
    criterion = nn.CrossEntropyLoss()
    
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
        # 更新学习率
        current_lr = scheduler.step(epoch)
        # 验证阶段（每个 epoch 结束后）
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2f}%, LR: {current_lr:.6f}")
        if save_path:
            torch.save(model.state_dict(), save_path)

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    return acc

def evaluate_integrated_model(model, loader,k=None,dataset_name=None, device='cuda:0',test_info = '2023-TOMM'):
    model.eval()
    logit_list = []
    label_list = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logit = model.get_logit(inputs)
            logit_list.append(logit)
            label_list.append(labels)
    logit_list = torch.cat(logit_list, dim=0).cpu()
    label_list = torch.cat(label_list, dim=0).cpu()
        
    get_metrics(logit_list,label_list,test_info = test_info,save_folder=f"metrics/{test_info}/{dataset_name}/{k}/")

if __name__ == "__main__":
    SETUP_SEED(seed=42)
    device = 'cuda:0'
    batch_size = 256
    # TRAIN_FOLD = {'AF':{'k':[None]},'PAC':{'k':[None]},'PVC':{'k':[None]}}
    info = '2023-TOMM'
    dataset_name = 'SRRSH'
    TRAIN_FOLD = {dataset_name:TRAIN_FOLD[dataset_name]}
    # dataset_list = ['ecg_id','SRRSH']
    # TRAIN_FOLD = {dataset_name:TRAIN_FOLD[dataset_name] for dataset_name in dataset_list}
    pre_train_path = lambda i: f"metrics/{info}/EfficientNet1D_pretrain_{i}.pth"
    
    for dataset_name,fold_dict in TRAIN_FOLD.items():
        for k in fold_dict['k']:
            # 预训练
            # pre_train_dataset_dict = pretrain_dataset(TRAIN_DATASET_LIST,k = 0,process_des=info)
            # data = pre_train_dataset_dict["all_data"]
            # X_pre_train, x_pre_val, y_pre_train, y_pre_val,*_ = train_test_split(data['X'], data['y'], test_size=0.2, random_state=42, stratify=data['y'])
            # pre_train_dataset = ECGDataset(X_pre_train,y_pre_train,n_concat=4)
            # pre_val_dataset = ECGDataset(x_pre_val,y_pre_val,n_concat=4)
            # pre_train_loader = DataLoader(pre_train_dataset, batch_size=batch_size, shuffle=True)
            # preval_loader = DataLoader(pre_val_dataset, batch_size=batch_size, shuffle=False)
            # model_param = [{'sigma_d': 1.0, 'sigma_w': 1.0},{'sigma_d': 1.1, 'sigma_w': 1.2},{'sigma_d': 1.2, 'sigma_w': 1.4}]
            # model_list = [EfficientNet1D(**param, num_classes=pre_train_dataset.class_num) for param in model_param]
            # for i,model in enumerate(model_list):
            #     train_dien_model(model, pre_train_loader, preval_loader,lr_init=0.01,  device=device, num_epochs=30,save_path=pre_train_path(i))
            # 预加载模型权重
            # for i,model in enumerate(model_list):
            #     model.load_state_dict(torch.load(pre_train_path(i)))
            # 训练集
            first_segments, first_labels = get_meta(dataset_name=dataset_name, k=k,session=0,process_des='2023-TOMM')
            X_train,x_val,y_train,y_val = train_test_split(first_segments,first_labels, test_size=0.2, random_state=42,stratify=first_labels)
            train_dataset = ECGDataset(X_train,y_train,n_concat=4)
            val_dataset = ECGDataset(x_val,y_val,n_concat=4)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            n_classes = train_dataset.class_num
            # 微调
            # for i,model in enumerate(model_list):
            #     n_classes = train_dataset.class_num
            #     model.modify_classifier(n_classes)
            #     train_dien_model(model, train_loader, val_loader,lr_init=0.001,  device=device, num_epochs=50)
            # 训练端到端模型
            model_param = [{'sigma_d': 1.0, 'sigma_w': 1.0},{'sigma_d': 1.1, 'sigma_w': 1.2},{'sigma_d': 1.2, 'sigma_w': 1.4}]
            model_list = [EfficientNet1D(**param, num_classes=n_classes) for param in model_param]
            for model in model_list:
                train_dien_model(model, train_loader, val_loader, device=device, num_epochs=100)
            # 测试模型
            inter_model = IntegratedEfficientNet1D(model_list)
            second_segments, second_labels = get_meta(dataset_name=dataset_name, k=k,session=1,process_des='2023-TOMM')
            # 完整形态学
            for mode in ['Pearson','MAE',]:
                for selection in ['top','middle','bottom']:
                    test_indices = find_indices(X_train,y_train,second_segments,second_labels,mode=mode,ratio=0.25,selection=selection)
                    new_second_segments = second_segments[test_indices]
                    new_second_labels = second_labels[test_indices]
                    test_dataset = ECGDataset(new_second_segments,new_second_labels,n_concat=4)
                    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                    test_acc = evaluate_integrated_model(inter_model, test_loader, device=device,test_info = info,k=k,dataset_name=dataset_name)
            # ST-T最差的P-QRS分类
            for mode in ['Pearson','MAE',]:
                pqrs_dict = find_worst_st_dict(X_train,y_train,second_segments,second_labels,mode=mode,info=info)
                for selection,data_dict in pqrs_dict.items():
                    new_second_segments = data_dict['X']
                    new_second_labels = data_dict['y']
                    test_dataset = ECGDataset(new_second_segments,new_second_labels,n_concat=4)
                    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                    test_acc = evaluate_integrated_model(inter_model, test_loader, device=device,test_info = info,k=k,dataset_name=dataset_name)
            test_dataset = ECGDataset(second_segments,second_labels,n_concat=4)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            test_acc = evaluate_integrated_model(inter_model, test_loader, device=device,test_info = info,k=k,dataset_name=dataset_name)

            
            