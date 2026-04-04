import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, experts, feature_dim=256, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.dim = feature_dim

        for e in experts:
            for p in e.parameters():
                p.requires_grad = False
        
        self.local_experts = nn.ModuleList(
            [e for e in experts if getattr(e, 'seg', '') != 'seg']
        )
        self.global_expert = next(
            e for e in experts if getattr(e, 'seg', '') == 'seg'
        )
        self.num_experts = len(self.local_experts)

        self.weight_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.dim * 2, self.dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.dim, self.dim),
            ) for _ in range(self.num_experts)
        ])
        
        self.sigmoid = nn.Sigmoid()
        
        self.norm = nn.LayerNorm(self.dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        f_global = F.normalize(self.global_expert(inputs)['embedding'], p=2, dim=1)

        # 2. 提取并融合所有局部特征
        weighted_locals = []
        gate_weights_list = []
        
        for i, expert in enumerate(self.local_experts):
            # 提取当前局部特征 [B, D]
            f_local = F.normalize(expert(inputs)['embedding'], p=2, dim=1)
            
            combined = torch.cat([f_global, f_local], dim=-1)
            T = self.weight_nets[i](combined)
            T_new = self.sigmoid(T) 
            weighted_local = T_new * f_local 
    
            weighted_locals.append(weighted_local)
            gate_weights_list.append(T_new)
            
        weighted_locals_sum = torch.stack(weighted_locals, dim=1).sum(dim=1) 
        
        f_local_fused = f_global + weighted_locals_sum
        
        prototype = self.dropout(f_local_fused)
        prototype = self.norm(prototype)
        gate_weights = torch.stack(gate_weights_list, dim=1)
        return {
            'embedding': prototype,                 # [B, D]
            'weight': gate_weights                  # [B, N, D]
        }

    def train(self, mode=True):
        super(CrossAttention, self).train(mode)
        self.global_expert.eval()
        for e in self.local_experts:
            e.eval()
        return self