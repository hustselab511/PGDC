import torch
from torch.utils.data import Dataset
from imblearn.over_sampling import SMOTE
import numpy as np

class ECGHeartbeatDataset(Dataset):
    def __init__(self, data_list, label_list):
        self.data = data_list
        self.labels = label_list
        self.class_num = len(np.unique(label_list))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

def extract_embeddings(model, dataloader, device):
    model.eval()
    all_embs = []
    all_labels = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            features = model.features(x)
            all_embs.append(features.cpu().numpy())
            all_labels.append(y.numpy())
    return np.concatenate(all_embs), np.concatenate(all_labels)



def build_siamese_dataset(embeddings, labels, window_size=200):
    n = len(embeddings)
    idx1 = []
    idx2 = []
    for i in range(n):
        end = min(i + window_size, n)
        idx1.extend([i] * (end - (i + 1)))
        idx2.extend(range(i + 1, end))
    
    feat1 = embeddings[idx1]
    feat2 = embeddings[idx2]
    X = np.hstack([feat1, feat2])
    y = (labels[idx1] == labels[idx2]).astype(int)
    
    smote = SMOTE(sampling_strategy='auto')
    return smote.fit_resample(X, y)

class SiameseFeatureDataset(Dataset):
    def __init__(self, resampled_features, resampled_labels):
        self.features = torch.from_numpy(resampled_features).float()
        self.labels = torch.from_numpy(resampled_labels).float()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        f = self.features[idx]
        u = f[:128]
        v = f[128:]
        label = self.labels[idx]
        return u, v, label