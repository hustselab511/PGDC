import torch
import torch.nn as nn
from einops import repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F

class SequenceToPatches(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        B, C, L = x.shape
        num_patches = L // self.patch_size
        return x.view(B, num_patches, self.patch_size)


class ViT(nn.Module):
    def __init__(self, 
                 seq_len=120,
                 patch_size=5,     
                 hidden_size=256,   
                 depth=4,          
                 heads=8,         
                 dropout=0.1):
        super().__init__()
        
        self.patch_size = patch_size
        self.seq_len = seq_len

        self.num_patches = (seq_len-1+patch_size) // patch_size

        self.to_patch_embedding = nn.Sequential(
            SequenceToPatches(patch_size),
            nn.Linear(patch_size, hidden_size),
            Rearrange('b n h -> b h n')
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, hidden_size))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=heads, 
            dim_feedforward=hidden_size * 4, 
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.output_dim = hidden_size

    def forward(self, x):
        b, c, current_len = x.shape
        remainder = current_len % self.patch_size
        if remainder != 0:
            pad_len = self.patch_size - remainder
            x = F.pad(x, (0, pad_len), "constant", 0)

        x = self.to_patch_embedding(x).transpose(1, 2)
        b, n, _ = x.shape

        # 拼接 CLS token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        x += self.pos_embedding[:, :n+1, :]
        
        x = self.dropout(x)
        x = self.transformer_encoder(x)

        # 5. 特征提取
        embedding = x[:, 0]
        
        return {'embedding': embedding}

# https://arxiv.org/pdf/2503.13495
# 将6头改为8头
class ViTClassify(nn.Module):
    def __init__(self, num_classes=2, seq_len=2000, patch_size=20,hidden_size=256,depth=6, heads=8,dropout=0.1):
        super().__init__()
        self.vit = ViT(seq_len=seq_len, patch_size=patch_size, hidden_size=hidden_size, depth=depth, heads=heads, dropout=dropout)
        # 论文指出 MLP Head 维度为 128 
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def modify_num_classes(self, num_classes):
        for param in self.vit.parameters():
            param.requires_grad = False
        self.classifier[2] = nn.Linear(128, num_classes)
    def forward(self, x):
        features = self.vit(x)
        return self.classifier(features['embedding'])
# --- 测试运行 ---
if __name__ == "__main__":
    sample_heartbeat = torch.randn(8, 1, 2000)
    
    # 实例化模型
    vit_classifier = ViTClassify()
    
    # 预测
    predictions = vit_classifier(sample_heartbeat)
    
    print(f"输入形状: {sample_heartbeat.shape}") # [8, 1, 2000]
    print(f"提取出的特征维度: {predictions.shape}") # [8, 512]
    print(f"提取出的分类维度: {predictions.shape}") # [8, 2]
