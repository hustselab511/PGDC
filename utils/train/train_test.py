# -*- coding: UTF-8 -*-
"""
@Project ：ECG identify 
@File    ：triplet_moe_train.py
@Author  ：yankangli
@Date    ：2025/11/14 19:31 
"""
import os
import sys
import torch
# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from config import (
    GET_FOLDER,
    GET_EXPERT_NAME,
    SETUP_SEED,
    TEST_DATASET_NAME,
    MOE_NAME,
    MODEL_NAME,
    MODEL_FOLDER
)
from utils.trainer import Triplet_MoE_Trainer,TripletTrainer
from utils.train import get_train_dataloader,get_moe,get_expert,TRAIN_CONFIG
from utils.dataset import find_indices,find_worst_st_dict,WindowsDataset,padding_collate_fn,WindowsDatasetBuilder
from torch.utils.data import DataLoader




def train(num_epochs = 20,k = 0,n_sampes = 8,train_transforms = None,seg = "seg",gpu_id = 0,step = 0,**kwargs):
    loader_dict = get_train_dataloader(k,n_sampes = n_sampes,batch_size=kwargs["batch_size"],train_transforms = train_transforms)
    if step == 1:
        model = get_moe(seg = seg,k = k,isTrained = False,num_classes = loader_dict["num_classes"])
        trainer = Triplet_MoE_Trainer(
        model=model,
        **loader_dict,
        model_name=GET_EXPERT_NAME(model.num_experts),
        checkpoint_dir=GET_FOLDER(model_folder=MOE_NAME, model_name=MODEL_NAME),
        device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu",
        **kwargs,
    )
    else:
        model = get_expert(seg = seg, isTrained=False,k=k,
                           num_classes = loader_dict["num_classes"])
        trainer = TripletTrainer(model=model,**loader_dict,model_name=model.seg,checkpoint_dir=GET_FOLDER(model_folder=MODEL_FOLDER, model_name=MODEL_NAME),
                                     device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu",**kwargs,)
        
    trainer.train(num_epochs=num_epochs,k=k)

def test(dataset_name = TEST_DATASET_NAME,ids = None,k = 0,enroll_time = 10,test_time = 5,
         seg="seg",test_first = False,step = 0,**kwargs): 
    datasetBulider = WindowsDatasetBuilder(dataset_name = dataset_name,enroll_time = enroll_time,test_time = test_time)
    data_dict = datasetBulider.enroll_test(k = k,is_plt = False,test_first = test_first,ids = ids)
    enroll_data = data_dict['enroll_data']
    test_data = data_dict['test_data']
    num_classes = max(set(enroll_data['labels']))+1
    model = get_moe(seg = seg,k = k,isTrained = True,num_classes = num_classes)
    trainer = TripletTrainer(
        model=model,
        is_test=True,
        model_name=GET_EXPERT_NAME(model.num_experts),
        checkpoint_dir=GET_FOLDER(model_folder=MOE_NAME, model_name=MODEL_NAME),
        **kwargs,
    )
    # 正常注册
    enroll_dataset = WindowsDataset(**enroll_data)
    enroll_loader = DataLoader(
        enroll_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True
    )
    trainer.enroll(loader=enroll_loader)
    # 形态学测试
    X_enroll = enroll_data['data']
    y_enroll = enroll_data['labels']
    X_test = test_data['data']
    y_test = test_data['labels']
    # 形态学测试
    for mode in ['Pearson','MAE',]:
        for selection in ['top','middle','bottom']:
            test_indices = find_indices(X_enroll,y_enroll,X_test,y_test,mode=mode,ratio=0.25,selection=selection)
            new_X_test = [X_test[i] for i in test_indices]
            new_y_test = [y_test[i] for i in test_indices]
            test_dataset = WindowsDataset(data = new_X_test,labels = new_y_test)
            test_loader = DataLoader(test_dataset,batch_size=512,shuffle=True,num_workers=0,pin_memory=True,collate_fn=padding_collate_fn,)
            trainer.rigistered_test(loader=test_loader)
            trainer.register_metrics(test_info = dataset_name,k = k)
    # ST-T最差的P-QRS分类
    for mode in ['Pearson','MAE',]:
        pqrs_dict = find_worst_st_dict(X_enroll,y_enroll,X_test,y_test,mode=mode)
        for selection,data_dict in pqrs_dict.items():
            new_X_test = data_dict['X']
            new_y_test = data_dict['y']
            test_dataset = WindowsDataset(data = new_X_test,labels = new_y_test)
            test_loader = DataLoader(test_dataset,batch_size=512,shuffle=True,num_workers=0,pin_memory=True,collate_fn=padding_collate_fn,)
            trainer.rigistered_test(loader=test_loader)
            trainer.register_metrics(test_info = dataset_name,k = k)
    #正常测试
    test_dataset = WindowsDataset(**test_data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=padding_collate_fn,
    )
    trainer.rigistered_test(loader=test_loader)
    trainer.register_metrics(test_info = dataset_name,k = k)
if __name__ == "__main__":
    SETUP_SEED(seed=42)
    kwargs = TRAIN_CONFIG.kwargs
    lt = TRAIN_CONFIG.segment_list
    train(lt = lt,**kwargs,step = 0)
    train(lt = lt,**kwargs,step = 1)
    test(lt = lt,**kwargs,step = 1)
    print(kwargs)