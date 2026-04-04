import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from pytorch_metric_learning import losses, miners,distances

# 你自己的工具
from config import SETUP_SEED, TRAIN_FOLD
from utils.dataset import get_meta
from MutiRoi_Dataset import MutiRoiDataset
from model import HPAFModel
from trainer import Trainer
from utils.dataset import pretrain_dataset
from utils.dataset.segment_dataset import find_worst_st_dict,find_indices
from config import TRAIN_DATASET_LIST


SETUP_SEED(seed=42)
info = "2026-ArXiv"
distance = distances.CosineSimilarity()
train_config = {
    "batch_size": 512,
    "device": "cuda:0",
    "info": info,
    "num_epochs": 50,
    "miner": None,
    "loss_func": losses.TripletMarginLoss(margin=0.2,distance=distance),
    "isPretrain": False,
    "TRAIN_FOLD": {'SRRSH': TRAIN_FOLD['SRRSH']},
    "num_prototypes": 5,
    "checkpoint_dir": f"metrics/{info}",
}
dataloader_func = lambda dataset,shuffle=False: DataLoader(dataset, batch_size=train_config["batch_size"], shuffle=shuffle, pin_memory=True, num_workers=4)
TRAIN_FOLD = {'SRRSH': TRAIN_FOLD['SRRSH']}
if __name__ == "__main__":
    for dataset_name, fold_dict in TRAIN_FOLD.items():
        for k in fold_dict["k"]:
            # 默认配置
            model = HPAFModel(in_ch=1,embed_dim=256,final_dim=256,dropout=0.1,normalize_output=True,)
            trainer = Trainer(model=model,model_name=model.__class__.__name__,checkpoint_dir=train_config["checkpoint_dir"],
                device=train_config["device"],
                miner=train_config["miner"],
                loss_func=train_config["loss_func"],
                num_prototypes=train_config["num_prototypes"]
            )
            pre_path = trainer.checkpoint_dir / f"{trainer.model_name}_pretrain.pth"
            # 预训练过程
            # dataset_dict = pretrain_dataset(TRAIN_DATASET_LIST,k = k,process_des=info)
            # data = dataset_dict["all_data"]
            # pretrain_segments, pretrain_labels = data["X"], data["y"]
            # X_pre_train, x_pre_val, y_pre_train, y_pre_val,*_ = train_test_split(pretrain_segments, pretrain_labels, test_size=0.2, random_state=42, stratify=pretrain_labels)
            # pretrain_train_dataset = MutiRoiDataset(X_pre_train, y_pre_train)
            # pretrain_val_dataset = MutiRoiDataset(x_pre_val, y_pre_val)
            # pretrain_train_loader = dataloader_func(pretrain_train_dataset,shuffle=True)
            # pretrain_val_loader = dataloader_func(pretrain_val_dataset,shuffle=False)
            # trainer.train(train_loader=pretrain_train_loader,val_loader=pretrain_val_loader,num_epochs=train_config["num_epochs"],save_path=pre_path)
            # 预训练后加载模型
            # model.load_state_dict(torch.load(pre_path,map_location="cpu",weights_only=True))
            # 训练端到端模型
            first_segments, first_labels = get_meta(dataset_name=dataset_name,k=k,session=0,process_des=info)
            X_train,x_val,y_train,y_val = train_test_split(first_segments,first_labels,test_size=0.2,random_state=42,stratify=first_labels)
            second_segments, second_labels = get_meta(dataset_name=dataset_name,k=k,session=1,process_des=info)
            train_dataset = MutiRoiDataset(X_train, y_train)
            train_loader = dataloader_func(train_dataset,shuffle=True)
            val_dataset = MutiRoiDataset(x_val, y_val)
            val_loader = dataloader_func(val_dataset,shuffle=False)
            trainer.train(train_loader=train_loader,val_loader=val_loader,num_epochs=train_config["num_epochs"])
            first_segments, first_labels = get_meta(dataset_name=dataset_name,k=k,session=0,process_des=info)
            enroll_dataset = MutiRoiDataset(first_segments, first_labels)
            enroll_loader = dataloader_func(enroll_dataset,shuffle=False)
            trainer.enroll(loader=enroll_loader)
            second_segments, second_labels = get_meta(dataset_name=dataset_name, k=k,session=1,process_des=info)
            test_dataset = MutiRoiDataset(second_segments, second_labels)
            test_loader = dataloader_func(test_dataset,shuffle=False)
            trainer.test(loader=test_loader,test_info=info,dataset_name=dataset_name)
            # 形态学测试
            for mode in ['Pearson','MAE',]:
                for selection in ['top','middle','bottom']:
                    test_indices = find_indices(X_train,y_train,second_segments,second_labels,mode=mode,ratio=0.25,selection=selection)
                    new_second_segments = second_segments[test_indices]
                    new_second_labels = second_labels[test_indices]
                    test_dataset = MutiRoiDataset(new_second_segments, new_second_labels)
                    test_loader = dataloader_func(test_dataset,shuffle=False)
                    trainer.test(loader=test_loader,test_info=info,dataset_name=dataset_name)
            # ST-T最差的P-QRS分类
            for mode in ['Pearson','MAE',]:
                pqrs_dict = find_worst_st_dict(X_train,y_train,second_segments,second_labels,mode=mode,info=info)
                for selection,data_dict in pqrs_dict.items():
                    new_second_segments = data_dict['X']
                    new_second_labels = data_dict['y']
                    test_dataset = MutiRoiDataset(new_second_segments,new_second_labels)    
                    test_loader = dataloader_func(test_dataset,shuffle=False)
                    trainer.test(loader=test_loader,test_info = info,dataset_name=dataset_name)
            test_dataset = MutiRoiDataset(second_segments,second_labels)
            test_loader = dataloader_func(test_dataset,shuffle=False)
            trainer.test(loader=test_loader,test_info = info,dataset_name=dataset_name)