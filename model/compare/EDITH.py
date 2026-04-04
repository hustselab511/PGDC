import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiResBlock1d(nn.Module):
    def __init__(self, in_channels):
        super(MultiResBlock1d, self).__init__()
        # 并行路径以捕获多尺度特征
        self.path1 = nn.Sequential(nn.Conv1d(in_channels, 32, 15, padding=7), nn.BatchNorm1d(32), nn.ReLU())
        self.path2 = nn.Sequential(nn.Conv1d(32, 64, 15, padding=7), nn.BatchNorm1d(64), nn.ReLU())
        self.path3 = nn.Sequential(nn.Conv1d(64, 128, 15, padding=7), nn.BatchNorm1d(128), nn.ReLU())
        self.res = nn.Conv1d(in_channels, 32+64+128, 1)

    def forward(self, x):
        p1 = self.path1(x)
        p2 = self.path2(p1)
        p3 = self.path3(p2)
        out = torch.cat([p1, p2, p3], dim=1)
        return F.relu(out + self.res(x))

class SPP1d(nn.Module):
    def __init__(self):
        super(SPP1d, self).__init__()
        self.pools = nn.ModuleList([nn.AdaptiveMaxPool1d(w) for w in [8, 16, 32]])
    def forward(self, x):
        return torch.cat([p(x).flatten(1) for p in self.pools], dim=1)

class EDITHBase(nn.Module):
    def __init__(self):
        super(EDITHBase, self).__init__()
        self.features = MultiResBlock1d(1)
        self.spp = SPP1d()
        # SPP后的维度：(32+64+128) * (8+16+32) = 12544 
        self.fc = nn.Linear(12544, 128) 

    def forward(self, x, get_embedding=False):
        x = self.features(x)
        x = self.spp(x)
        x = self.fc(x)
        if get_embedding:
            return torch.sigmoid(x)
        return F.relu(x)

class EDITHIdentification(nn.Module):
    def __init__(self, num_classes):
        super(EDITHIdentification, self).__init__()
        self.base = EDITHBase()
        self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Linear(128, num_classes) # 闭集识别使用 Softmax 层

    def features(self, x):
        return self.base(x)
    
    def forward(self, x):
        emb = self.base(x)
        return self.classifier(self.dropout(emb))

class EDITHSiamese(nn.Module):
    def __init__(self):
        super(EDITHSiamese, self).__init__()
        self.decision = nn.Sequential(
            nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(64, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, 1), nn.Sigmoid()
        )

    def forward(self, u, v):
        sq_diff = torch.pow(u - v, 2)
        prod_prox = u * v
        combined = torch.cat([sq_diff, prod_prox], dim=1)
        return self.decision(combined)