import os
import sys
from torch.utils.data import DataLoader
from torchvision import transforms

from config import (
    TRAIN_DATASET_LIST,
    HIDDEN_SIZE,
    GET_EXPERT_NAME,
    MODEL_FOLDER,
    MODEL_NAME,
    MOE_NAME,
    GET_FOLDER,
    PROJECT_PATH,
    TRAIN_FOLDER,
    GET_SEG_LEN,
)
from utils import AddGaussianNoise, ToTensor1D
from model import ExpertEncoder,MultiExpert,Pro_Classify,CrossAttention,ViT
from utils.dataset import padding_collate_fn,pretrain_dataset,WindowsDatasetBuilder,WeightedSessionBalancedSampler, SegmentDataset
from sklearn.model_selection import train_test_split
# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


class TRAIN_CONFIG:
    kwargs = {
        "batch_size": 512,
        "test_size": 0.0,
        "val_size": 0.2,
        "lr": 0.0001,
        "num_epochs": 50,
        "train_transforms": transforms.Compose([
        AddGaussianNoise(noise_std_range=(0.01, 0.1)),
        ToTensor1D(),]),
    }
    segment_list = ["seg","pqrs","st_t"]



def get_expert(
    seg,
    isTrained = False,
    k = 0,
    model_folder=MODEL_FOLDER,
    model_name=MODEL_NAME,
    num_classes=63,
    **kwargs,
):
    def get_encoder(seg):
        encoder = ViT(hidden_size=HIDDEN_SIZE,seq_len=GET_SEG_LEN(),patch_size=5)
        encoder = Pro_Classify(encoder=encoder, num_classes=num_classes,output_dim = 128)
        model = ExpertEncoder(encoder=encoder, seg=seg)
        return model
    if isinstance(seg, str):
        model = get_encoder(seg)
    else:
        model_list = [get_encoder(s) for s in seg]
        model = MultiExpert(model_list)
        seg = "seg_list"
    if isTrained:
        folder = os.path.join(PROJECT_PATH, TRAIN_FOLDER, GET_FOLDER(model_folder=model_folder, model_name=model_name))
        filename = f"{seg}_{k}.pth"
        fullpath = folder + filename
        ckpt = torch.load(fullpath, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("model_state_dict", ckpt)
        curr_state = model.state_dict()
        model.load_state_dict({
            k: v for k, v in state_dict.items() 
            if k in curr_state and v.shape == curr_state[k].shape
        }, strict=False)
        if isinstance(model, MultiExpert):
            model = model.getExperts()
        else:
            model.encoder = model.encoder.encoder
    return model

def get_moe_model(
    experts,
    k = 0,  
    isTrained=True,
    model_name=MODEL_NAME,
    model_folder=MOE_NAME,
    **kwargs,
):
    num_experts = len(experts)
    model = CrossAttention(experts=experts,feature_dim=HIDDEN_SIZE)
    num_experts = model.num_experts
    if isTrained:
        path = os.path.join(os.path.join(PROJECT_PATH, TRAIN_FOLDER), GET_FOLDER(model_folder=model_folder, model_name=model_name))
        sub_path = path + GET_EXPERT_NAME(num_experts) + f"_{k}.pth"
        print(f"Loading Mo  E model from: {sub_path}")
        model.load_state_dict(
                torch.load(sub_path, map_location="cpu", weights_only=True)[
                    "model_state_dict"
                ]
            )

    return model

def get_moe(seg,k,isTrained = False,num_classes = 63):
    experts = get_expert(seg,isTrained = True,k = k,num_classes = num_classes)
    model = get_moe_model(
        experts=experts,
        k = k,
        isTrained=isTrained,
        model_folder=MOE_NAME, 
        model_name=MODEL_NAME,
    )
    return model


def get_train_dataloader(k,n_sampes,batch_size,train_transforms):
    dataset_dict = pretrain_dataset(TRAIN_DATASET_LIST,k = k)
    data = dataset_dict["all_data"]
    X_train, X_tmp, y_train, y_tmp, attr_train, attr_tmp,*_ = train_test_split(
        data['X'], data['y'], data['attr'], data['time'],data['session'], test_size=0.2, random_state=42, stratify=data['y']
    )
    train_dataset = SegmentDataset(X_train, y_train, attr_train)
    val_dataset = SegmentDataset(X_tmp, y_tmp, attr_tmp)
    train_dataset.transform = train_transforms
    sampler = WeightedSessionBalancedSampler(train_dataset, n_classes=batch_size//n_sampes, 
                                             n_samples=n_sampes,class_weights=1/dataset_dict['label_count'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=sampler,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return {"train_loader": train_loader, "val_loader": val_loader,"num_classes": max(train_dataset.labels) + 1,'session_count':dataset_dict['session_count']}
